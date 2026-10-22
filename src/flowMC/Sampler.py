import io
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, PyTree

from flowMC.resource.base import Resource
from flowMC.resource.states import State
from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle
from flowMC.strategy.base import Strategy

logger = logging.getLogger(__name__)


@runtime_checkable
class _CursorStrategy(Protocol):
    """The checkpointable cursor interface implemented by step strategies."""

    current_position: int

    def set_current_position(self, current_position: int) -> None: ...


@runtime_checkable
class _LogPDFResource(Protocol):
    """The minimal interface needed to fingerprint a log-PDF checkpoint."""

    def log_pdf(self, x: Float[Array, " n_dim"], data: PyTree) -> Float[Array, "1"]: ...


class Sampler:
    """Top level API that the users primarily interact with.

    Args:
        n_dim (int): Dimension of the parameter space.
        n_chains (int): Number of chains to sample.
        rng_key (Key): JAX PRNGKey.
        resources (dict[str, Resource]): Resources to be used by the sampler.
        strategies (dict[str, Strategy]): Strategies to be used by the sampler.
        strategy_order (list[str]): Order of strategies to execute.
        resource_strategy_bundles (ResourceStrategyBundle): Pre-configured bundle
            containing resources and strategies.
        checkpoint_interval (float): Minimum wall-clock seconds that must elapse
            since the previous write before a new checkpoint is written.
            Default ``600`` (10 minutes).  Set to ``0`` to disable checkpointing
            entirely.
    """

    # Essential parameters
    n_dim: int
    n_chains: int
    rng_key: Key[Array, ""]
    resources: dict[str, Resource]
    strategies: dict[str, Strategy]
    strategy_order: Optional[list[str]]

    outdir: str = "./outdir/"

    # Checkpoint / resume
    checkpoint_interval: float = 600.0
    _prev_elapsed: float  # cumulative wall-clock seconds from all prior runs

    def __init__(
        self,
        *,
        n_dim: int,
        n_chains: int,
        rng_key: Key[Array, ""],
        resources: Optional[dict[str, Resource]] = None,
        strategies: Optional[dict[str, Strategy]] = None,
        strategy_order: Optional[list[str]] = None,
        resource_strategy_bundles: Optional[ResourceStrategyBundle] = None,
        outdir: str = "./outdir/",
        checkpoint_interval: float = 600.0,
    ) -> None:
        """Initialize the sampler.

        Provide *either* ``resources`` + ``strategies`` + ``strategy_order``
        *or* a ``resource_strategy_bundles`` pre-configured bundle.

        Args:
            n_dim (int): Dimension of the parameter space.
            n_chains (int): Number of parallel chains.
            rng_key (Key): JAX PRNG key.
            resources (Optional[dict[str, Resource]]): Dictionary of named resources
                (kernels, buffers, etc.). Must be paired with ``strategies``.
            strategies (Optional[dict[str, Strategy]]): Dictionary of named strategies.
                Must be paired with ``resources``.
            strategy_order (Optional[list[str]]): Ordered list of strategy names to
                execute each call to ``sample``.
            resource_strategy_bundles (Optional[ResourceStrategyBundle]): Pre-configured
                bundle that provides resources, strategies, and ordering.
            outdir (str): Directory where ``checkpoint.pkl`` is written.
                Created automatically if it does not exist.  Default ``"./outdir/"``.
            checkpoint_interval (float): Minimum wall-clock seconds that must elapse
                since the previous write before a new checkpoint is written.
                Default ``600`` (10 minutes).  Set to ``0`` to disable checkpointing.
                When checkpointing is enabled the JAX XLA compilation cache is also
                activated at ``{outdir}/jax_cache/`` so that compiled step functions
                are reused across processes (no recompilation on resume).
        Raises:
            ValueError: If neither ``resources``/``strategies`` nor
                ``resource_strategy_bundles`` is provided.
        """
        self.n_dim = n_dim
        self.n_chains = n_chains
        self.rng_key = rng_key
        self.outdir = outdir
        self.checkpoint_interval = checkpoint_interval
        self._prev_elapsed = 0.0

        if checkpoint_interval > 0:
            cache_dir = Path(outdir) / "jax_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            current_cache = getattr(jax.config, "jax_compilation_cache_dir", None)
            if current_cache is None:
                jax.config.update("jax_compilation_cache_dir", str(cache_dir))
                jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            elif current_cache != str(cache_dir):
                logger.warning(
                    "JAX compilation cache is already set to '%s' (this Sampler "
                    "would use '%s'). JAX cache is process-global; only one "
                    "Sampler per process should configure it. "
                    "The cache will not be reconfigured.",
                    current_cache,
                    cache_dir,
                )

        if resources is not None and strategies is not None:
            logger.info(
                "Resources and strategies provided. Ignoring resource strategy bundles."
            )
            self.resources = resources
            self.strategies = strategies
            self.strategy_order = strategy_order

        else:
            logger.info(
                "Resources or strategies not provided. Using resource strategy bundles."
            )
            if resource_strategy_bundles is None:
                raise ValueError(
                    "Resource strategy bundles not provided."
                    "Please provide either resources and strategies or resource strategy bundles."
                )
            self.resources = resource_strategy_bundles.resources
            self.strategies = resource_strategy_bundles.strategies
            self.strategy_order = resource_strategy_bundles.strategy_order

    def _training_loop_end_indices(self) -> set[int]:
        """Return the set of strategy_order indices that are the last step of each training loop.

        Scans the training section (everything before ``"reset_steppers"``) for the
        first strategy name that repeats.  The gap between its first and second
        occurrence is the training-loop period; everything before it is a one-time
        preamble (e.g. ``"initialize_tempered_positions"`` in PT bundles).

        Returns an empty set when ``"reset_steppers"`` is absent or there are no
        training strategies.

        Assumes each strategy name appears at most once within a single training loop.
        """
        assert isinstance(self.strategy_order, list)
        try:
            reset_idx = self.strategy_order.index("reset_steppers")
        except ValueError:
            return set()
        training_strategies = self.strategy_order[:reset_idx]
        if not training_strategies:
            return set()

        # Find the period of the repeating block and the length of any one-time preamble.
        # Strategies that never repeat are part of the preamble.
        phase_len = reset_idx  # fallback: treat entire training section as one block
        preamble_len = 0
        for i, name in enumerate(training_strategies):
            for j in range(i + 1, len(training_strategies)):
                if training_strategies[j] == name:
                    phase_len = j - i
                    preamble_len = i
                    break
            else:
                continue  # name doesn't repeat — part of preamble, keep scanning
            break  # period found

        n_loops = (reset_idx - preamble_len) // phase_len
        return {preamble_len + phase_len * (k + 1) - 1 for k in range(n_loops)}

    def _resume_from_checkpoint(
        self, ckpt_path: Path, data: dict
    ) -> tuple[int, Key[Array, ""], Float[Array, "n_chains n_dim"]]:
        """Load and validate a checkpoint; restore sampler state in-place.

        Mutates ``self.resources`` and ``self.strategies`` with the saved state.

        Args:
            ckpt_path: Path to the ``.pkl`` checkpoint file.
            data: The ``data`` dict the caller intends to pass to ``sample``,
                used to verify the logpdf fingerprint.

        Returns:
            ``(start_idx, rng_key, last_step)`` — the strategy index to resume
            from and the restored PRNG key and chain positions.

        Raises:
            ValueError: If the checkpoint config is inconsistent with the
                current sampler, or if the logpdf fingerprint has changed.
        """
        try:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
        except Exception as e:
            raise ValueError(
                f"Checkpoint file at {ckpt_path} is corrupt or unreadable ({e}). "
                "Delete it to start a fresh run."
            ) from e

        meta = ckpt["_meta"]
        if meta["n_dim"] != self.n_dim or meta["n_chains"] != self.n_chains:
            raise ValueError(
                f"Checkpoint was created with n_dim={meta['n_dim']}, "
                f"n_chains={meta['n_chains']}, but current sampler has "
                f"n_dim={self.n_dim}, n_chains={self.n_chains}. "
                "Delete the checkpoint file to start a fresh run."
            )
        if meta["strategy_order"] != self.strategy_order:
            raise ValueError(
                "Checkpoint strategy_order does not match the current sampler. "
                "The sampler configuration has changed since the checkpoint was saved. "
                "Delete the checkpoint file to start a fresh run."
            )

        ckpt_fp = meta["logpdf_fingerprint"]
        logpdf_resource = self.resources.get("logpdf")
        if (
            ckpt_fp is not None
            and logpdf_resource is not None
            and isinstance(logpdf_resource, _LogPDFResource)
        ):
            current_fp = float(
                logpdf_resource.log_pdf(jnp.asarray(ckpt["last_step"][0]), data)
            )
            if abs(current_fp - ckpt_fp) > 1e-6:
                raise ValueError(
                    f"logpdf or data has changed since checkpoint was saved "
                    f"(checkpoint fingerprint: {ckpt_fp:.6g}, "
                    f"current: {current_fp:.6g}). "
                    "Delete the checkpoint file to start a fresh run."
                )

        for k, entry in ckpt["resources"].items():
            if isinstance(entry, tuple) and len(entry) == 2:
                method, payload = entry
                if method == "pkl":
                    self.resources[k] = payload
                elif method == "eqx" and k in self.resources:
                    buf = io.BytesIO(payload)
                    self.resources[k] = eqx.tree_deserialise_leaves(
                        buf, self.resources[k]
                    )
                else:
                    payload_info = (
                        len(payload)
                        if isinstance(payload, (bytes, bytearray))
                        else repr(type(payload))
                    )
                    logger.warning(
                        "Cannot restore resource '%s' (method=%r, payload=%s) — removing from resources.",
                        k,
                        method,
                        payload_info,
                    )
                    self.resources.pop(k, None)
            elif isinstance(entry, Resource):
                self.resources[k] = entry
            else:
                logger.warning(
                    "Cannot restore resource '%s' with unsupported checkpoint entry %r.",
                    k,
                    type(entry),
                )
        for name, cursor in ckpt["stepper_cursors"].items():
            strategy = self.strategies.get(name)
            if isinstance(strategy, _CursorStrategy) and isinstance(cursor, int):
                strategy.set_current_position(cursor)
        for name, attrs in ckpt["strategy_states"].items():
            if name in self.strategies:
                for attr, val in attrs.items():
                    setattr(self.strategies[name], attr, val)

        self._prev_elapsed = float(ckpt["elapsed_time"])
        start_idx = ckpt["strategy_idx"] + 1
        logger.info(
            "Resumed from checkpoint at strategy_idx=%d (%s)",
            start_idx - 1,
            ckpt_path,
        )
        return start_idx, ckpt["rng_key"], ckpt["last_step"]

    def _save_checkpoint(
        self,
        ckpt_path: Path,
        strategy_idx: int,
        rng_key: Key[Array, ""],
        last_step: Float[Array, "n_chains n_dim"],
        data: dict,
        elapsed_time: float,
    ) -> None:
        """Atomically write a checkpoint to ``ckpt_path``.

        Writes to a ``.tmp`` file first, then renames — so a crash mid-write
        leaves the previous checkpoint intact.

        The ``"logpdf"`` resource is excluded because it is a user-supplied
        callable that strategies read but never mutate; it is already present
        in ``self.resources`` when ``sample`` is called again.

        Args:
            ckpt_path: Destination path for the ``.pkl`` checkpoint.
            strategy_idx: Index of the last completed strategy in ``strategy_order``.
            rng_key: Current JAX PRNG key.
            last_step: Current chain positions, shape ``(n_chains, n_dim)``.
            data: Data dict passed to the strategy calls, used to record the
                logpdf fingerprint.
            elapsed_time: Cumulative wall-clock seconds across all runs up to
                this checkpoint write (``_prev_elapsed`` + seconds since
                ``sample`` was called).
        """
        resources_to_save: dict[str, tuple[str, Resource | bytes]] = {}
        for k, v in self.resources.items():
            if k == "logpdf":
                continue
            try:
                pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                resources_to_save[k] = ("pkl", v)
            # Pickling may execute arbitrary user-defined code; every failure must
            # use the equivalent Equinox serialisation fallback.
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "pickle.dumps failed for resource '%s' (%s); falling back to eqx serialisation.",
                    k,
                    exc,
                )
                buf = io.BytesIO()
                eqx.tree_serialise_leaves(buf, v)
                resources_to_save[k] = ("eqx", buf.getvalue())

        stepper_cursors = {
            name: strategy.current_position
            for name, strategy in self.strategies.items()
            if isinstance(strategy, _CursorStrategy)
        }

        state_attrs = (
            "_call_count",
            "_prev_acceptance",
            "_prev_cov",
            "_patience_count",
        )
        strategy_states = {
            name: {attr: getattr(s, attr) for attr in state_attrs if hasattr(s, attr)}
            for name, s in self.strategies.items()
            if any(hasattr(s, attr) for attr in state_attrs)
        }

        logpdf_resource = self.resources.get("logpdf")
        logpdf_fp = (
            float(logpdf_resource.log_pdf(jnp.asarray(last_step[0]), data))
            if isinstance(logpdf_resource, _LogPDFResource)
            else None
        )

        ckpt_data = {
            "_meta": {
                "n_dim": self.n_dim,
                "n_chains": self.n_chains,
                "strategy_order": self.strategy_order,
                "logpdf_fingerprint": logpdf_fp,
            },
            "strategy_idx": strategy_idx,
            "rng_key": rng_key,
            "last_step": last_step,
            "elapsed_time": elapsed_time,
            "resources": resources_to_save,
            "stepper_cursors": stepper_cursors,
            "strategy_states": strategy_states,
        }

        tmp = ckpt_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(ckpt_data, f)
        tmp.replace(ckpt_path)
        logger.debug("Checkpoint saved at strategy_idx=%d", strategy_idx)

    def sample(self, initial_position: Float[Array, "n_chains n_dim"], data: dict):
        """Execute the strategy loop and populate resource buffers with samples.

        If ``outdir`` is set and ``checkpoint_interval > 0``, the sampler writes
        a checkpoint to ``{outdir}/checkpoint.pkl`` atomically after each complete
        training loop (when at least ``checkpoint_interval`` seconds have elapsed
        since the previous write).  On the next call with the same ``outdir`` the
        sampler detects the existing file, validates it against the current
        configuration, and resumes from the strategy immediately after the last
        checkpointed one.

        Checkpoint validation raises ``ValueError`` if:

        * ``n_dim`` or ``n_chains`` differ from the checkpoint.
        * ``strategy_order`` differs from the checkpoint.
        * The logpdf fingerprint (evaluated at the first chain's position) differs
          by more than ``1e-6``, indicating a change in the likelihood or data.

        Args:
            initial_position: Starting chain positions, shape
                ``(n_chains, n_dim)`` or broadcastable.  Ignored when resuming
                from a checkpoint (the checkpointed positions are used instead).
            data: Arbitrary data dict forwarded to every strategy call.  Must be
                consistent with any checkpoint on disk; changes are detected via
                the logpdf fingerprint when available.
        """
        initial_position = jnp.atleast_2d(initial_position)
        assert isinstance(self.strategy_order, list)

        rng_key = self.rng_key
        last_step = initial_position
        start_idx = 0
        _t0 = time.perf_counter()

        # Check for an existing checkpoint and resume from it if found.
        ckpt_path: Optional[Path] = (
            Path(self.outdir) / "checkpoint.pkl"
            if self.checkpoint_interval > 0
            else None
        )
        if ckpt_path is not None:
            if ckpt_path.exists():
                start_idx, rng_key, last_step = self._resume_from_checkpoint(
                    ckpt_path, data
                )
            else:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # early_stopped in any State resource means we should skip remaining training loops.
        skip_to_production = any(
            isinstance(r, State) and r.data.get("early_stopped", False)
            for r in self.resources.values()
        )

        training_end_indices = self._training_loop_end_indices()
        last_ckpt_t = time.perf_counter()

        for idx, strategy in enumerate(self.strategy_order):
            if idx < start_idx:
                continue

            if skip_to_production:
                if strategy == "reset_steppers":
                    skip_to_production = False
                    for resource in self.resources.values():
                        if (
                            isinstance(resource, State)
                            and "early_stopped" in resource.data
                        ):
                            resource.data["early_stopped"] = False
                    logger.info(
                        "[Early stop] Remaining training loops skipped. "
                        "Starting production phase."
                    )
                else:
                    continue

            if strategy not in self.strategies:
                raise ValueError(
                    f"Invalid strategy name '{strategy}' provided. "
                    f"Available strategies are: {list(self.strategies.keys())}."
                )
            rng_key, self.resources, last_step = self.strategies[strategy](
                rng_key, self.resources, last_step, data
            )

            if not skip_to_production:
                for resource in self.resources.values():
                    if isinstance(resource, State) and resource.data.get(
                        "early_stopped", False
                    ):
                        skip_to_production = True
                        logger.info(
                            "[Early stop] Early stop triggered — "
                            "skipping remaining training loops."
                        )
                        break

            if (
                ckpt_path is not None
                and idx in training_end_indices
                and time.perf_counter() - last_ckpt_t >= self.checkpoint_interval
            ):
                self._save_checkpoint(
                    ckpt_path,
                    idx,
                    rng_key,
                    last_step,
                    data,
                    elapsed_time=self._prev_elapsed + (time.perf_counter() - _t0),
                )
                last_ckpt_t = time.perf_counter()

        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        if self.checkpoint_interval > 0:
            shutil.rmtree(Path(self.outdir) / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

    # TODO: Implement quick access and summary functions that operates on buffer

    def serialize(self):
        """Serialize the sampler object."""
        raise NotImplementedError

    def deserialize(self):
        """Deserialize the sampler object."""
        raise NotImplementedError

import logging

import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.strategy.base import Strategy
from flowMC.utils.logging import enable_verbose_logging

logger = logging.getLogger(__name__)


class CheckEarlyStop(Strategy):
    """Strategy that checks if global acceptance rate has stabilized.

    After each training loop iteration, this strategy reads the global
    acceptance rate from the buffer and compares it (relative change) against
    the previous loop. If the relative change falls below the tolerance for
    a specified number of consecutive checks, it sets ``"early_stopped": True``
    in the sampler state, which causes the Sampler to skip remaining training
    loops and jump directly to the production phase.

    The ``n_training_loops`` parameter effectively becomes the *maximum*
    number of training loops when this strategy is enabled.
    """

    state_name: str
    acceptance_buffer_key: str
    relative_tolerance: float
    min_acceptance_rate: float
    min_stable_acceptance: float
    acceptance_window: int
    n_loops_skip: int
    patience: int
    _call_count: int
    _prev_acceptance: float | None
    _prev_cov: float | None
    _patience_count: int

    def __init__(
        self,
        state_name: str,
        acceptance_buffer_key: str = "target_global_accs",
        relative_tolerance: float = 0.1,
        min_acceptance_rate: float = 0.1,
        min_stable_acceptance: float = 0.01,
        acceptance_window: int = 0,
        n_loops_skip: int = 3,
        patience: int = 3,
        verbose: bool = False,
    ):
        """Initialize the CheckEarlyStop strategy.

        Args:
            state_name: Name of the State resource (e.g., ``"sampler_state"``).
            acceptance_buffer_key: Key in the State that points to the global
                acceptance buffer name (e.g., ``"target_global_accs"``).
            relative_tolerance: Relative change threshold. Early stopping
                triggers when ``|current - prev| / max(prev, 1e-8) < tol``.
            min_acceptance_rate: Minimum global acceptance rate threshold.
                Early stopping triggers immediately when the current mean
                acceptance rate reaches or exceeds this value, regardless of
                stability. Set to 0 to disable this condition.
            min_stable_acceptance: Minimum acceptance rate required for the
                stability condition to trigger early stop. If the acceptance
                is below this value the sampler is considered stuck near zero
                and the stable condition is suppressed. Defaults to 0.01.
            acceptance_window: Number of most-recent thinned steps to use
                when computing the mean acceptance rate. If 0 (default), all
                available finite entries are used. Set this to
                ``n_global_steps // global_thinning`` to evaluate only the
                acceptance from the latest training loop iteration.
            n_loops_skip: Minimum number of training loops before early
                stopping can trigger. Must be >= 1 so that at least one
                comparison can be made.
            patience: Number of consecutive loops where the acceptance rate
                must be stable before triggering early stop. Defaults to 3.
            verbose: Whether to log detailed information.
        """
        self.state_name = state_name
        self.acceptance_buffer_key = acceptance_buffer_key
        self.relative_tolerance = relative_tolerance
        self.min_acceptance_rate = min_acceptance_rate
        self.min_stable_acceptance = min_stable_acceptance
        self.acceptance_window = acceptance_window
        self.n_loops_skip = max(n_loops_skip, 1)
        self.patience = max(patience, 1)
        self._call_count = 0
        self._prev_acceptance = None
        self._prev_cov = None
        self._patience_count = 0
        if verbose:
            enable_verbose_logging(logger)

    def __call__(
        self,
        rng_key: Key[Array, ""],
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        Key[Array, ""],
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        """Check if global acceptance rate has stabilized.

        If stable for ``patience`` consecutive loops after the warmup period,
        sets ``"early_stopped": True`` in the sampler state.

        Args:
            rng_key: JAX PRNGKey.
            resources: Dictionary of resources.
            initial_position: Current positions of chains.
            data: Additional data.

        Returns:
            Tuple of (rng_key, resources, initial_position).
        """
        self._call_count += 1

        state = resources[self.state_name]
        assert isinstance(state, State), f"Resource {self.state_name} must be a State"

        # Read the global acceptance buffer
        buffer_name = state.data.get(self.acceptance_buffer_key)
        assert isinstance(buffer_name, str), (
            f"State key {self.acceptance_buffer_key} must point to a string "
            f"(buffer name)"
        )

        acceptance_buffer = resources[buffer_name]
        assert isinstance(acceptance_buffer, Buffer), (
            f"Resource {buffer_name} must be a Buffer"
        )

        # Compute current mean acceptance from finite entries
        all_accs = acceptance_buffer.data
        finite_mask = jnp.all(jnp.isfinite(all_accs), axis=0)
        finite_accs = all_accs[:, finite_mask]

        if self.acceptance_window > 0:
            finite_accs = finite_accs[:, -self.acceptance_window :]

        if finite_accs.shape[1] == 0:
            logger.debug("No acceptance data recorded yet — skipping early-stop check.")
            return rng_key, resources, initial_position

        current_acceptance = float(jnp.mean(finite_accs))
        per_chain_mean = jnp.mean(finite_accs, axis=1)
        current_cov = float(
            jnp.std(per_chain_mean) / max(float(jnp.mean(per_chain_mean)), 1e-8)
        )

        # During warmup, just record acceptance and return
        if self._call_count <= self.n_loops_skip:
            logger.debug(
                f"[Early stop] Warmup loop {self._call_count}/{self.n_loops_skip}: "
                f"global acceptance = {current_acceptance:.4f}, "
                f"CoV = {current_cov:.4f} (not checking yet)"
            )
            self._prev_acceptance = current_acceptance
            self._prev_cov = current_cov
            return rng_key, resources, initial_position

        # Compute relative changes for both mean acceptance and CoV
        assert self._prev_acceptance is not None
        assert self._prev_cov is not None
        mean_change = abs(current_acceptance - self._prev_acceptance) / max(
            abs(self._prev_acceptance), 1e-8
        )
        cov_change = abs(current_cov - self._prev_cov) / max(abs(self._prev_cov), 1e-8)

        logger.debug(
            f"[Early stop] Loop {self._call_count}: "
            f"acceptance = {current_acceptance:.4f} (prev = {self._prev_acceptance:.4f}, "
            f"relative change = {mean_change:.4%}), "
            f"CoV = {current_cov:.4f} (prev = {self._prev_cov:.4f}, "
            f"relative change = {cov_change:.4%}), "
            f"threshold = {self.relative_tolerance:.4%}"
        )

        mean_stable = mean_change < self.relative_tolerance
        cov_stable = cov_change < self.relative_tolerance
        rate_reached = (
            self.min_acceptance_rate > 0
            and current_acceptance >= self.min_acceptance_rate
        )
        acceptance_too_low = current_acceptance < self.min_stable_acceptance
        stable_condition = mean_stable and cov_stable and not acceptance_too_low

        if acceptance_too_low and mean_stable and cov_stable:
            logger.debug(
                f"[Early stop] Acceptance mean and CoV are stable but acceptance "
                f"{current_acceptance:.4f} is below min stable threshold "
                f"{self.min_stable_acceptance:.4f} — not triggering stable early stop."
            )

        if stable_condition or rate_reached:
            self._patience_count += 1
            if rate_reached:
                logger.debug(
                    f"[Early stop] Acceptance {current_acceptance:.4f} >= "
                    f"min target {self.min_acceptance_rate:.4f} "
                    f"({self._patience_count}/{self.patience} consecutive loop(s) required)"
                )
            else:
                logger.debug(
                    f"[Early stop] Both acceptance mean and CoV are stable "
                    f"({self._patience_count}/{self.patience} consecutive loop(s) required)"
                )
            if self._patience_count >= self.patience:
                if rate_reached:
                    logger.info(
                        f"[Early stop] Training stopped early at loop {self._call_count}: "
                        f"global acceptance {current_acceptance:.4f} reached the "
                        f"minimum target {self.min_acceptance_rate:.4f} for "
                        f"{self.patience} consecutive loop(s). "
                        f"Proceeding to production phase."
                    )
                else:
                    logger.info(
                        f"[Early stop] Training stopped early at loop {self._call_count}: "
                        f"global acceptance mean and CoV have both been stable for "
                        f"{self.patience} consecutive loop(s) "
                        f"(mean change {mean_change:.4%}, CoV change {cov_change:.4%}, "
                        f"threshold {self.relative_tolerance:.4%}). "
                        f"Proceeding to production phase."
                    )
                state.data["early_stopped"] = True
        else:
            if self._patience_count > 0:
                reason = []
                if not mean_stable:
                    reason.append(f"mean changed by {mean_change:.4%}")
                if not cov_stable:
                    reason.append(f"CoV changed by {cov_change:.4%}")
                if acceptance_too_low:
                    reason.append(
                        f"acceptance {current_acceptance:.4f} below "
                        f"min stable threshold {self.min_stable_acceptance:.4f}"
                    )
                logger.debug(
                    f"[Early stop] Patience counter reset ({', '.join(reason)})."
                )
            self._patience_count = 0

        self._prev_acceptance = current_acceptance
        self._prev_cov = current_cov
        return rng_key, resources, initial_position

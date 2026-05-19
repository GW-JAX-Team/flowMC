import logging

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.kernel.base import ProposalBase
from flowMC.resource.states import State
from flowMC.strategy.base import Strategy
from flowMC.utils.logging import enable_verbose_logging

logger = logging.getLogger(__name__)


class AdaptStepSize(Strategy):
    """Strategy to adapt the step size of a local sampler based on acceptance rates.

    This strategy computes the acceptance rate from a specified buffer and calls
    the kernel's adapt_step_size method.

    The strategy is generic and works with any kernel that implements the
    adapt_step_size(acceptance_rate, target_rate) method.
    """

    kernel_name: str
    state_name: str
    acceptance_buffer_key: str
    target_acceptance_rate: float
    acceptance_window: int
    n_loops_skip: int
    _call_count: int

    def __init__(
        self,
        kernel_name: str,
        state_name: str,
        acceptance_buffer_key: str,
        target_acceptance_rate: float,
        acceptance_window: int = 0,
        n_loops_skip: int = 0,
        verbose: bool = False,
    ):
        """Initialize the AdaptStepSize strategy.

        Args:
            kernel_name: Name of the kernel resource to adapt.
            state_name: Name of the State resource that tracks sampler state.
            acceptance_buffer_key: Key in the State resource that points to the
                acceptance buffer name (e.g., "target_local_accs").
            target_acceptance_rate: Target acceptance rate for adaptation.
                Common values: 0.234 (Random Walk), 0.574 (MALA), 0.65 (HMC).
            acceptance_window: Number of recent samples to use for computing
                acceptance rate. If 0, uses all available samples.
            n_loops_skip: Skip adaptation for the first N calls to allow chains
                to reach a stable state before adapting (warmup period).
            verbose: Whether to log adaptation information.
        """
        self.kernel_name = kernel_name
        self.state_name = state_name
        self.acceptance_buffer_key = acceptance_buffer_key
        self.target_acceptance_rate = target_acceptance_rate
        self.acceptance_window = acceptance_window
        self.n_loops_skip = n_loops_skip
        self._call_count = 0
        if verbose:
            enable_verbose_logging(logger)

    def __call__(
        self,
        rng_key: Key,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        Key,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        """Adapt the kernel's step size based on recent acceptance rates.

        Args:
            rng_key: JAX PRNGKey.
            resources: Dictionary of resources.
            initial_position: Current positions of chains.
            data: Additional data.

        Returns:
            Tuple of (rng_key, resources, initial_position).
        """
        # Skip adaptation during warmup period
        if self._call_count < self.n_loops_skip:
            self._call_count += 1
            logger.debug(
                f"Skipping adaptation for {self.kernel_name} "
                f"(warmup {self._call_count}/{self.n_loops_skip})"
            )
            return rng_key, resources, initial_position

        self._call_count += 1

        assert isinstance(state := resources[self.state_name], State), (
            f"Resource {self.state_name} must be a State"
        )

        # Get the acceptance buffer name from state
        assert isinstance(
            buffer_name := state.data.get(self.acceptance_buffer_key), str
        ), (
            f"State key {self.acceptance_buffer_key} must point to a string "
            f"(buffer name)"
        )

        assert isinstance(acceptance_buffer := resources[buffer_name], Buffer), (
            f"Resource {buffer_name} must be a Buffer"
        )

        # Compute acceptance rate from buffer
        all_accs = acceptance_buffer.data

        # Filter out steps (columns) where all chains have -inf (global steps)
        finite_steps_accs = all_accs[:, jnp.all(jnp.isfinite(all_accs), axis=0)]

        # Window the last N steps
        windowed_accs = finite_steps_accs[:, -self.acceptance_window :]

        acceptance_rate = float(jnp.mean(windowed_accs))

        logger.debug(f"Adapting {self.kernel_name} step size:")
        logger.debug(f"  Current acceptance rate: {acceptance_rate:.4f}")
        logger.debug(f"  Target acceptance rate: {self.target_acceptance_rate:.4f}")

        # Get and validate the local sampler
        assert self.kernel_name in resources, (
            f"Local sampler '{self.kernel_name}' not found in resources"
        )
        local_sampler = resources[self.kernel_name]
        assert isinstance(local_sampler, ProposalBase), (
            f"Resource '{self.kernel_name}' must be a ProposalBase (local sampler kernel), "
            f"got {type(local_sampler)}"
        )

        # Call the local sampler's adapt_step_size method
        assert hasattr(local_sampler, "adapt_step_size"), (
            f"Local sampler '{self.kernel_name}' must implement adapt_step_size() method"
        )

        resources[self.kernel_name] = local_sampler.adapt_step_size(
            acceptance_rate, target_rate=self.target_acceptance_rate
        )

        return rng_key, resources, initial_position


class AdaptStepSizePerDim(Strategy):
    """Strategy to adapt the per-dimension step size profile using empirical scale estimates.

    Computes the sample standard deviation of the most-recent ``window`` chain
    positions per chain and sets the per-dimension step size profile so that each
    dimension's effective step is proportional to its posterior scale.  The geometric mean
    of the step sizes is preserved, so this strategy controls the *shape* of the step size
    vector while ``AdaptStepSize`` continues to control its *global scale*.

    The positions buffer is fully overwritten every training loop (written from offset 0),
    so all stored positions are already from the most recent loop — no burn-in bias.
    Pass ``window=n_local_steps // local_thinning`` so the window covers exactly one
    training loop's worth of stored positions.  The fixed shape limits the array that
    enters the JIT-compiled variance kernel, keeping per-call runtime low.

    Each call drives the kernel's effective per-dim profile exactly to the current scale
    estimate in one step (no damping, no clipping).  Adaptation starts from the first call.

    Run this after AdaptStepSize in the training phase.
    Requires the kernel to implement get_effective_dim_profile() and apply_per_dim_scaling().
    """

    kernel_name: str
    state_name: str
    positions_buffer_key: str
    window: int
    _call_count: int

    def __init__(
        self,
        kernel_name: str,
        state_name: str,
        positions_buffer_key: str,
        window: int = 50,
        verbose: bool = False,
    ):
        """Initialize AdaptStepSizePerDim.

        Args:
            kernel_name: Name of the kernel resource to adapt.
            state_name: Name of the State resource that tracks sampler state.
            positions_buffer_key: Key in State that points to the positions buffer name
                (e.g. "target_positions").
            window: Number of most-recent steps per chain to use when estimating
                per-dimension scales. Pass ``n_local_steps // local_thinning`` so the
                window exactly covers the positions written in one training loop.
            verbose: Log per-dim step sizes after each adaptation.
        """
        self.kernel_name = kernel_name
        self.state_name = state_name
        self.positions_buffer_key = positions_buffer_key
        self.window = window
        self._call_count = 0
        if verbose:
            enable_verbose_logging(logger)

    @staticmethod
    @jax.jit
    def _compute_sigma(
        flat: Float[Array, "N n_dim"],
        mask: Array,
    ) -> Float[Array, " n_dim"]:
        n = jnp.sum(mask)
        safe_n = jnp.maximum(n, 1.0)
        valid = jnp.where(mask[:, None], flat, 0.0)
        mean = jnp.sum(valid, axis=0) / safe_n
        sq_dev = jnp.where(mask[:, None], (flat - mean) ** 2, 0.0)
        var = jnp.sum(sq_dev, axis=0) / jnp.maximum(safe_n - 1.0, 1.0)
        return jnp.sqrt(jnp.maximum(var, jnp.finfo(flat.dtype).eps))

    def __call__(
        self,
        rng_key: Key,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        Key,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        """Adapt per-dimension step sizes based on the empirical std of recent chain positions.

        Args:
            rng_key: JAX PRNGKey.
            resources: Dictionary of resources.
            initial_position: Current positions of chains.
            data: Additional data (unused).

        Returns:
            Tuple of (rng_key, resources, initial_position).
        """
        self._call_count += 1

        assert isinstance(state := resources[self.state_name], State), (
            f"Resource {self.state_name} must be a State"
        )

        assert isinstance(
            buffer_name := state.data.get(self.positions_buffer_key), str
        ), f"State key {self.positions_buffer_key} must point to a string (buffer name)"

        assert isinstance(positions_buffer := resources[buffer_name], Buffer), (
            f"Resource {buffer_name} must be a Buffer"
        )

        # positions: (n_chains, n_steps, n_dims)
        positions = positions_buffer.data

        # Slice the last `window` steps per chain — fixed shape, JIT-cache-friendly.
        # Python slicing handles buffers smaller than window gracefully (returns all).
        recent = positions[:, -self.window :, :]  # (n_chains, W, n_dims)

        finite_mask = jnp.all(jnp.isfinite(recent), axis=-1)  # (n_chains, W)
        flat = recent.reshape(-1, recent.shape[-1])  # (n_chains * W, n_dims)
        mask = finite_mask.reshape(-1)  # (n_chains * W,)

        if int(jnp.sum(mask)) < 2:
            return rng_key, resources, initial_position

        sigma = self._compute_sigma(flat, mask)

        # Target profile: normalised to geometric mean 1
        log_sigma = jnp.log(sigma)
        target_profile = jnp.exp(log_sigma - jnp.mean(log_sigma))

        assert self.kernel_name in resources, (
            f"Kernel '{self.kernel_name}' not found in resources"
        )
        kernel = resources[self.kernel_name]
        assert isinstance(kernel, ProposalBase), (
            f"Resource '{self.kernel_name}' must be a ProposalBase, got {type(kernel)}"
        )
        assert hasattr(kernel, "get_effective_dim_profile") and hasattr(
            kernel, "apply_per_dim_scaling"
        ), (
            f"Kernel '{self.kernel_name}' must implement get_effective_dim_profile() "
            f"and apply_per_dim_scaling(). Got {type(kernel)}."
        )

        # Current per-dim profile from the kernel (geomean=1, kernel-specific)
        current_profile = getattr(kernel, "get_effective_dim_profile")()
        safe_current = jnp.maximum(
            current_profile, jnp.finfo(current_profile.dtype).eps
        )

        # Exact ratio to move current profile to target profile in one step
        ratios = target_profile / safe_current

        logger.debug(f"Adapting per-dim step sizes for {self.kernel_name}:")
        logger.debug(f"  Estimated sigma: {sigma}")
        logger.debug(f"  Applied ratios: {ratios}")

        resources[self.kernel_name] = getattr(kernel, "apply_per_dim_scaling")(ratios)

        return rng_key, resources, initial_position

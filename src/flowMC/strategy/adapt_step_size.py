import logging

import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

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
    the kernel's adapt_step_size method. It can optionally only run during the
    training phase by checking a State resource.

    The strategy is generic and works with any kernel that implements the
    adapt_step_size(acceptance_rate, target_rate) method.
    """

    kernel_name: str
    state_name: str
    acceptance_buffer_key: str
    target_acceptance_rate: float
    training_only: bool
    acceptance_window: int

    def __init__(
        self,
        kernel_name: str,
        state_name: str,
        acceptance_buffer_key: str,
        target_acceptance_rate: float,
        training_only: bool = True,
        acceptance_window: int = -1,
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
            training_only: If True, only adapt during training phase (when
                State["training"] is True). If False, adapt in all phases.
            acceptance_window: Number of recent samples to use for computing
                acceptance rate. If -1, uses all available samples.
            verbose: Whether to log adaptation information.
        """
        self.kernel_name = kernel_name
        self.state_name = state_name
        self.acceptance_buffer_key = acceptance_buffer_key
        self.target_acceptance_rate = target_acceptance_rate
        self.training_only = training_only
        self.acceptance_window = acceptance_window
        if verbose:
            enable_verbose_logging(logger)

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
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
        assert isinstance(state := resources[self.state_name], State), (
            f"Resource {self.state_name} must be a State"
        )

        # Check if we should skip adaptation (training_only mode + not in training)
        if self.training_only and not state.data.get("training", False):
            logger.debug(
                f"Skipping step size adaptation for {self.kernel_name} "
                f"(training_only=True, training={state.data.get('training', False)})"
            )
            return rng_key, resources, initial_position

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

        # Filter out non-finite values (e.g., -inf from global steps) and compute mean
        if self.acceptance_window > 0:
            # Use only the most recent window of samples
            window_accs = all_accs[:, -self.acceptance_window :]
            finite_mask = jnp.isfinite(window_accs)
            finite_accs = jnp.where(finite_mask, window_accs, 0.0)
        else:
            # Use all available samples
            finite_mask = jnp.isfinite(all_accs)
            finite_accs = jnp.where(finite_mask, all_accs, 0.0)

        # Compute mean acceptance rate across all chains and samples
        n_finite = jnp.sum(finite_mask)
        acceptance_rate = float(jnp.sum(finite_accs) / jnp.maximum(n_finite, 1))

        logger.debug(f"Adapting {self.kernel_name} step size:")
        logger.debug(f"  Current acceptance rate: {acceptance_rate:.4f}")
        logger.debug(f"  Target acceptance rate: {self.target_acceptance_rate:.4f}")
        logger.debug(f"  Number of finite samples: {n_finite}")

        # Get kernel and adapt
        kernel = resources[self.kernel_name]
        assert isinstance(kernel, ProposalBase), (
            f"Resource {self.kernel_name} must be a ProposalBase"
        )

        # Call the kernel's adapt_step_size method
        if hasattr(kernel, "adapt_step_size"):
            resources[self.kernel_name] = kernel.adapt_step_size(
                acceptance_rate, target_rate=self.target_acceptance_rate
            )
        else:
            logger.warning(
                f"Kernel {self.kernel_name} does not implement adapt_step_size. "
                f"Skipping adaptation."
            )

        return rng_key, resources, initial_position

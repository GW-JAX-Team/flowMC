"""Early stopping strategy for NF training based on global acceptance rates."""

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from jaxtyping import Array, Float, PRNGKeyArray
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)


class EarlyStoppingCheck(Strategy):
    """Strategy that checks global acceptance rates and triggers early stopping.

    This strategy monitors the global acceptance rate during training and sets
    a flag when the acceptance rate exceeds a specified threshold. This allows
    the training loop to terminate early and proceed to the production phase.

    Args:
        acceptance_criterion: Target global acceptance rate (0.0-1.0). If None or >= 1.0,
            early stopping is disabled. When the acceptance rate exceeds this
            threshold, training will stop early. Default is None.
        window_size: Number of recent steps to average for acceptance calculation.
            Default is 100 steps.
        state_name: Name of the State resource that holds the early stopping flag.
            Default is "early_stopping_state".
    """

    def __init__(
        self,
        acceptance_criterion: float | None = None,
        window_size: int = 100,
        state_name: str = "early_stopping_state",
    ):
        """Initialize the early stopping check strategy."""
        self.acceptance_criterion = acceptance_criterion
        self.window_size = window_size
        self.state_name = state_name

        if acceptance_criterion is not None:
            if acceptance_criterion < 0.0:
                raise ValueError(
                    f"acceptance_criterion must be non-negative, "
                    f"got {acceptance_criterion}"
                )
            if acceptance_criterion >= 1.0:
                logger.info(
                    f"Early stopping disabled: acceptance_criterion = {acceptance_criterion} >= 1.0"
                )

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
        """Check global acceptance rate and set early stopping flag if criterion is met.

        Args:
            rng_key: JAX PRNG key
            resources: Dictionary of resources including acceptance buffers and state
            initial_position: Current positions of chains
            data: Additional data (unused)

        Returns:
            Tuple of (rng_key, resources, initial_position) - unchanged except for
            the early stopping flag in the state resource if criterion is met
        """
        # If early stopping is disabled, return immediately
        if self.acceptance_criterion is None or self.acceptance_criterion >= 1.0:
            return rng_key, resources, initial_position

        # Get the global acceptance buffer for training
        if "global_accs_training" not in resources:
            logger.warning(
                "global_accs_training buffer not found in resources. "
                "Early stopping check skipped."
            )
            return rng_key, resources, initial_position

        # Cast to Buffer type for type checking
        global_acc_resource = resources["global_accs_training"]
        if not isinstance(global_acc_resource, Buffer):
            logger.warning(
                "global_accs_training is not a Buffer resource. "
                "Early stopping check skipped."
            )
            return rng_key, resources, initial_position

        global_acc_buffer: Buffer = global_acc_resource
        global_acc_data = global_acc_buffer.data  # Shape: (n_chains, n_steps)

        # Get the current cursor position (number of steps filled)
        current_step = global_acc_buffer.cursor

        # Calculate mean acceptance rate over recent window
        if current_step >= self.window_size:
            # Use the most recent window_size steps
            recent_window = global_acc_data[:, current_step - self.window_size : current_step]
        elif current_step > 0:
            # Use all available steps if we haven't filled the window yet
            recent_window = global_acc_data[:, :current_step]
        else:
            # No data yet, skip check
            return rng_key, resources, initial_position

        # Calculate mean acceptance across all chains and steps in the window
        mean_acceptance = float(jnp.mean(recent_window))

        # Check if we've exceeded the criterion
        if mean_acceptance >= self.acceptance_criterion:
            logger.info(
                f"Early stopping triggered: global acceptance rate "
                f"{mean_acceptance:.3f} exceeds criterion {self.acceptance_criterion:.3f}"
            )

            # Set the early stopping flag in the state resource
            if self.state_name in resources:
                state_resource = resources[self.state_name]
                if isinstance(state_resource, State):
                    state: State = state_resource
                    state.update(["triggered"], [True])
                else:
                    logger.warning(
                        f"State resource '{self.state_name}' is not a State. "
                        "Early stopping flag not set."
                    )
            else:
                logger.warning(
                    f"State resource '{self.state_name}' not found. "
                    "Early stopping flag not set."
                )

        return rng_key, resources, initial_position

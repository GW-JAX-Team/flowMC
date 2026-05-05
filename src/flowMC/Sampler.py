import jax.numpy as jnp
from jaxtyping import Array, Float, Key
from typing import Optional
import logging

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource.states import State
from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle

logger = logging.getLogger(__name__)


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
    """

    # Essential parameters
    n_dim: int
    n_chains: int
    rng_key: Key
    resources: dict[str, Resource]
    strategies: dict[str, Strategy]
    strategy_order: Optional[list[str]]

    # Logging hyperparameters
    logging: bool = True
    outdir: str = "./outdir/"

    def __init__(
        self,
        n_dim: int,
        n_chains: int,
        rng_key: Key,
        resources: Optional[dict[str, Resource]] = None,
        strategies: Optional[dict[str, Strategy]] = None,
        strategy_order: Optional[list[str]] = None,
        resource_strategy_bundles: Optional[ResourceStrategyBundle] = None,
        **kwargs,
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
                execute each call to :meth:`sample`.
            resource_strategy_bundles (Optional[ResourceStrategyBundle]): Pre-configured
                bundle that provides resources, strategies, and ordering.
            **kwargs: Additional keyword arguments that override class-level attributes
                (e.g. ``logging``, ``outdir``).

        Raises:
            ValueError: If neither ``resources``/``strategies`` nor
                ``resource_strategy_bundles`` is provided.
        """
        # Copying input into the model

        self.n_dim = n_dim
        self.n_chains = n_chains
        self.rng_key = rng_key

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

        # Set and override any given hyperparameters
        class_keys = list(self.__class__.__dict__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def sample(self, initial_position: Float[Array, "n_chains n_dim"], data: dict):
        """Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.
            data (dict): Data to be used by the likelihood functions
        """

        initial_position = jnp.atleast_2d(initial_position)  # type: ignore
        rng_key = self.rng_key
        last_step = initial_position
        assert isinstance(self.strategy_order, list)

        skip_to_production = False

        for strategy in self.strategy_order:
            # Early-stop skip: jump over remaining training strategies
            # until we reach "reset_steppers" (the training→production boundary)
            if skip_to_production:
                if strategy == "reset_steppers":
                    skip_to_production = False
                    # Clear the early_stopped flag on all State resources so the
                    # post-strategy scan does not re-enable skip_to_production for
                    # update_state and every subsequent production strategy.
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
            (
                rng_key,
                self.resources,
                last_step,
            ) = self.strategies[strategy](rng_key, self.resources, last_step, data)

            # Check if any State resource has early_stopped flag set
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

    # TODO: Implement quick access and summary functions that operates on buffer

    def serialize(self):
        """Serialize the sampler object."""
        raise NotImplementedError

    def deserialize(self):
        """Deserialize the sampler object."""
        raise NotImplementedError

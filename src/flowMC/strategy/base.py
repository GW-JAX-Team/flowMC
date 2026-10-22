from abc import ABC, abstractmethod

from jaxtyping import Array, Float, Key

from flowMC.resource.base import Resource


class Strategy(ABC):
    """Base class for strategies, which are basically wrapper blocks that modify the
    state of the sampler.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        """Initialize the strategy."""
        raise NotImplementedError

    @abstractmethod
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
        """Execute one step of the strategy.

        Args:
            rng_key (Key): JAX PRNG key (consumed and split internally).
            resources (dict[str, Resource]): Mutable resource dictionary shared
                across all strategies.
            initial_position (Float[Array, "n_chains n_dim"]): Current chain
                positions of shape ``(n_chains, n_dim)``.
            data (dict): Auxiliary data passed to log-pdf calls.

        Returns:
            tuple: A 3-tuple of:
                - Key: Updated PRNG key.
                - dict[str, Resource]: Updated resources.
                - Float[Array, "n_chains n_dim"]: Updated chain positions.
        """
        raise NotImplementedError

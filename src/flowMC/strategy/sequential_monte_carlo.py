from jaxtyping import Array, Float, Key

from flowMC.resource.base import Resource


class SequentialMonteCarlo(Resource):
    def __init__(self):
        raise NotImplementedError

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
        raise NotImplementedError

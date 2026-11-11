from abc import abstractmethod
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
from jaxtyping import Array, Bool, Float, Key, PyTree

from flowMC.resource.base import Resource
from flowMC.resource.logPDF import LogPDF


class ProposalBase(eqx.Module, Resource):
    @abstractmethod
    def __init__(
        self,
    ):
        """Initialize the sampler class."""

    @abstractmethod
    def kernel(
        self,
        rng_key: Key[Array, ""],
        position: Float[Array, "nstep n_dim"],
        log_prob: Float[Array, "nstep 1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[
        Float[Array, "nstep n_dim"], Float[Array, "nstep 1"], Bool[Array, "n_step 1"]
    ]:
        """Kernel for one step in the proposal cycle."""

    def sample_chain(
        self,
        rng_key: Key[Array, ""],
        position: Float[Array, " n_dim"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
        n_steps: int,
    ) -> tuple[
        Float[Array, "n_steps n_dim"], Float[Array, " n_steps"], Bool[Array, " n_steps"]
    ]:
        """Take ``n_steps`` steps of `kernel` from ``position``.

        Returns the position, log-probability and acceptance flag after each step.
        """

        def body(carry, _):
            key, position, log_prob = carry
            key, subkey = jax.random.split(key)
            position, log_prob, do_accept = self.kernel(
                subkey, position, log_prob, logpdf, data
            )
            return (key, position, log_prob), (position, log_prob, do_accept)

        _, steps = jax.lax.scan(
            body, (rng_key, position, logpdf(position, data)), length=n_steps
        )
        return steps

    def adapt_step_size(self, acceptance_rate: float, target_rate: float) -> Self:
        """Adapt the step size based on acceptance rate.

        This method should be implemented by kernels that support step size adaptation.
        If not implemented, the default behavior is to return self unchanged.

        Args:
            acceptance_rate: The observed acceptance rate.
            target_rate: The target acceptance rate.

        Returns:
            A new instance of the kernel with adapted step size.
        """
        return self

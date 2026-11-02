from abc import abstractmethod
from collections.abc import Callable
from typing import Self

import equinox as eqx
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

    def init_cache(
        self,
        position: Float[Array, " n_dim"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, "1"], PyTree]:
        """Evaluate a chain's starting point once for `step`.

        Returns the log-probability at ``position`` and the values `step` reuses
        from one step to the next (``None`` unless a kernel needs them).
        """
        return logpdf(position, data), None

    def step(
        self,
        rng_key: Key[Array, ""],
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        cache: PyTree,
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], PyTree, Bool[Array, "1"]]:
        """`kernel` reusing ``cache`` from `init_cache` or the previous step.

        Returns the new position, log-probability, cache and acceptance flag.
        """
        position, log_prob, do_accept = self.kernel(
            rng_key, position, log_prob, logpdf, data
        )
        return position, log_prob, cache, do_accept

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

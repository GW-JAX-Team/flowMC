import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from typing import Callable
import logging
from equinox import tree_at

from flowMC.resource.kernel.base import ProposalBase
from flowMC.resource.logPDF import LogPDF

logger = logging.getLogger(__name__)


class GaussianRandomWalk(ProposalBase):
    """Gaussian random walk sampler class."""

    step_size: Float[Array, " n_dim"]
    ADAPTATION_RATE: float = 0.1

    def __repr__(self):
        return "Gaussian Random Walk with step size " + str(self.step_size)

    def __init__(
        self,
        step_size: Float[Array, " n_dim"],
    ):
        """Initialize Gaussian Random Walk sampler.

        Args:
            step_size: Step size for the random walk as a 1D array representing
                      diagonal elements of the proposal covariance matrix.
        """
        super().__init__()
        self.step_size = step_size

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        """Random walk gaussian kernel. This is a kernel that only evolve a single
        chain.

        Args:
            rng_key (PRNGKeyArray): Jax PRNGKey
            position (Float[Array, "n_dim"]): current position of the chain
            log_prob (Float[Array, "1"]): current log-probability of the chain
            data (PyTree): data to be passed to the logpdf function

        Returns:
            position (Float[Array, "n_dim"]): new position of the chain
            log_prob (Float[Array, "1"]): new log-probability of the chain
            do_accept (Int[Array, "1"]): whether the new position is accepted
        """

        key1, key2 = jax.random.split(rng_key)
        move_proposal: Float[Array, " n_dim"] = (
            jax.random.normal(key1, shape=position.shape) * self.step_size
        )

        proposal = position + move_proposal
        proposal_log_prob: Float[Array, " n_dim"] = logpdf(proposal, data)

        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept = log_uniform < proposal_log_prob - log_prob

        position = jnp.where(do_accept, proposal, position)
        log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
        return position, log_prob, do_accept

    def adapt_step_size(self, acceptance_rate: float, target_rate: float = 0.234):
        """Adapt step size based on acceptance rate.

        Args:
            acceptance_rate: The current acceptance rate.
            target_rate: The target acceptance rate (default: 0.234 for RWM).

        Returns:
            A new GaussianRandomWalk instance with updated step_size.
        """
        diff = acceptance_rate - target_rate
        new_step_size = self.step_size * (1.0 + self.ADAPTATION_RATE * diff)
        return tree_at(lambda k: k.step_size, self, new_step_size)

    def print_parameters(self):
        logger.debug("Gaussian Random Walk parameters:")
        logger.debug(f"  - step_size: {self.step_size}")

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path):
        raise NotImplementedError

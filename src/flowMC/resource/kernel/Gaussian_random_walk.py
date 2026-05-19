import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key, PyTree
from typing import Callable, Optional
from typing_extensions import Self
import logging
from equinox import tree_at

from flowMC.resource.kernel.base import ProposalBase
from flowMC.resource.logPDF import LogPDF

logger = logging.getLogger(__name__)


class GaussianRandomWalk(ProposalBase):
    """Gaussian random walk sampler class."""

    step_size: Float[Array, " n_dim"]
    periodic_mask: Bool[Array, " n_dim"]
    periodic_bounds: Float[Array, "n_dim 2"]
    ADAPTATION_RATE: float = 0.5

    def __repr__(self):
        return "Gaussian Random Walk with step size " + str(self.step_size)

    def __init__(
        self,
        step_size: Float[Array, " n_dim"],
        periodic_mask: Optional[Bool[Array, " n_dim"]] = None,
        periodic_bounds: Optional[Float[Array, "n_dim 2"]] = None,
    ):
        """Initialize Gaussian Random Walk sampler.

        Args:
            step_size: Step size for the random walk as a 1D array representing
                      diagonal elements of the proposal covariance matrix.
            periodic_mask: Boolean mask indicating which dimensions are periodic.
                If None, no periodic boundaries are applied.
            periodic_bounds: Array of shape (n_dim, 2) with [lower, upper] bounds
                for each periodic dimension. Only used where periodic_mask is True.
                If None, no periodic boundaries are applied.
        """
        super().__init__()
        self.step_size = step_size
        n_dim = (
            jnp.asarray(step_size).shape[0] if jnp.asarray(step_size).ndim > 0 else 1
        )
        if periodic_mask is None:
            periodic_mask = jnp.zeros(n_dim, dtype=bool)
        if periodic_bounds is None:
            periodic_bounds = jnp.zeros((n_dim, 2))
        self.periodic_mask = periodic_mask
        self.periodic_bounds = periodic_bounds

    def kernel(
        self,
        rng_key: Key,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        """Random walk gaussian kernel. This is a kernel that only evolve a single
        chain.

        Args:
            rng_key (Key): JAX PRNGKey
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

        # Wrap periodic dimensions
        lower = self.periodic_bounds[:, 0]
        upper = self.periodic_bounds[:, 1]
        period = upper - lower
        wrapped = lower + jnp.mod(proposal - lower, period)
        proposal = jnp.where(self.periodic_mask, wrapped, proposal)

        proposal_log_prob: Float[Array, " n_dim"] = logpdf(proposal, data)

        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept = log_uniform < proposal_log_prob - log_prob

        position = jnp.where(do_accept, proposal, position)
        log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
        return position, log_prob, do_accept

    def adapt_step_size(
        self, acceptance_rate: float, target_rate: float = 0.234
    ) -> Self:
        """Adapt step size based on acceptance rate.

        Args:
            acceptance_rate: The current acceptance rate.
            target_rate: The target acceptance rate (default: 0.234 for RWM).

        Returns:
            GaussianRandomWalk: A new instance with updated step_size.
        """
        diff = acceptance_rate - target_rate
        new_step_size = self.step_size * (1.0 + self.ADAPTATION_RATE * diff)
        return tree_at(lambda k: k.step_size, self, new_step_size)

    def get_effective_dim_profile(self) -> Float[Array, " n_dim"]:
        """Return the current per-dim step size profile normalised to geometric mean 1."""
        safe = jnp.maximum(self.step_size, jnp.finfo(self.step_size.dtype).eps)
        log_step = jnp.log(safe)
        return jnp.exp(log_step - jnp.mean(log_step))

    def apply_per_dim_scaling(self, ratios: Float[Array, " n_dim"]) -> Self:
        """Multiply step_size element-wise by ratios (returns a new GaussianRandomWalk).

        Args:
            ratios: Per-dimension multiplicative scaling factors, shape (n_dim,).

        Returns:
            GaussianRandomWalk: A new instance with updated step_size.
        """
        new_step_size = self.step_size * ratios
        return tree_at(lambda k: k.step_size, self, new_step_size)

    def print_parameters(self):
        logger.debug("Gaussian Random Walk parameters:")
        logger.debug(f"  - step_size: {self.step_size}")

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path):
        raise NotImplementedError

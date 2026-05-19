import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Bool, Float, Int, Key, PyTree
from typing import Callable, Optional
from typing_extensions import Self
import logging
from equinox import tree_at

from flowMC.resource.logPDF import LogPDF
from flowMC.resource.kernel.base import ProposalBase

logger = logging.getLogger(__name__)


class MALA(ProposalBase):
    """Metropolis-adjusted Langevin algorithm sampler class."""

    step_size: Float[Array, " n_dim"]
    periodic_mask: Bool[Array, " n_dim"]
    periodic_bounds: Float[Array, "n_dim 2"]
    ADAPTATION_RATE: float = 0.5

    def __repr__(self):
        return "MALA with step size " + str(self.step_size)

    def __init__(
        self,
        step_size: Float[Array, " n_dim"],
        periodic_mask: Optional[Bool[Array, " n_dim"]] = None,
        periodic_bounds: Optional[Float[Array, "n_dim 2"]] = None,
    ):
        """Initialize MALA sampler.

        Args:
            step_size: Step size for the MALA sampler as a 1D array representing
                      diagonal elements of the step size matrix.
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
        """Metropolis-adjusted Langevin algorithm kernel for a single chain.

        Args:
            rng_key (Key): JAX PRNGKey for stochastic operations.
            position (Float[Array, "n_dim"]): Current position of the chain.
            log_prob (Float[Array, "1"]): Current log-probability of the chain.
            logpdf: Log probability density function to evaluate.
            data (PyTree): Additional data to pass to the logpdf function.
        Returns:
            Tuple of (new_position, new_log_prob, acceptance_flag):
            - new_position: New position of the chain.
            - new_log_prob: New log-probability of the chain.
            - acceptance_flag: Whether the new position is accepted.
        """

        periodic_mask = self.periodic_mask
        periodic_bounds = self.periodic_bounds
        lower = periodic_bounds[:, 0]
        upper = periodic_bounds[:, 1]
        period = upper - lower

        def wrap_periodic(x: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
            """Wrap periodic dimensions into [lower, upper)."""
            wrapped = lower + jnp.mod(x - lower, period)
            return jnp.where(periodic_mask, wrapped, x)

        def body(
            carry: tuple[Float[Array, " n_dim"], Float[Array, " n_dim"], dict],
            this_key: Key,
        ) -> tuple[
            tuple[Float[Array, " n_dim"], Float[Array, " n_dim"], dict],
            tuple[Float[Array, " n_dim"], Float[Array, "1"], Float[Array, " n_dim"]],
        ]:
            logger.debug("Compiling MALA body")
            this_position, dt, data = carry
            dt2 = dt * dt
            this_log_prob, this_d_log = jax.value_and_grad(logpdf)(this_position, data)
            # MALA proposal: x' = x + (dt²/2) * ∇log p(x) + dt * ε, where ε ~ N(0, I)
            proposal = this_position + dt2 * this_d_log / 2
            proposal += dt * jax.random.normal(this_key, shape=this_position.shape)
            proposal = wrap_periodic(proposal)
            return (proposal, dt, data), (proposal, this_log_prob, this_d_log)

        key1, key2 = jax.random.split(rng_key)

        dt: Float[Array, " n_dim"] = self.step_size
        dt2 = dt * dt

        # Use scan to iterate twice: first to generate proposal from current position
        # and compute its log_prob and gradient, then to compute log_prob and gradient
        # at the proposed position. Note: proposal[1] from the second iteration is
        # discarded; we only need logprob[1] and d_logprob[1] for the acceptance ratio.
        # Using the same key twice is fine since proposal[1] is unused.
        _, (proposal, logprob, d_logprob) = jax.lax.scan(
            body, (position, dt, data), jnp.array([key1, key1])
        )

        def periodic_diff(
            a: Float[Array, " n_dim"], b: Float[Array, " n_dim"]
        ) -> Float[Array, " n_dim"]:
            """Compute minimum-image difference (a - b) for periodic dimensions."""
            raw_diff = a - b
            # For periodic dims, use nearest image convention
            periodic_diff_val = raw_diff - period * jnp.round(raw_diff / period)
            return jnp.where(periodic_mask, periodic_diff_val, raw_diff)

        # Metropolis-Hastings ratio: log[p(proposal)/p(position)] + log[q(position|proposal)/q(proposal|position)]
        # For periodic dimensions, use minimum-image distances in the proposal density
        ratio = logprob[1] - logprob[0]

        # Forward proposal: q(proposal | position) — how likely is the proposal given current position
        fwd_mean = position + dt2 * d_logprob[0] / 2
        fwd_diff = periodic_diff(proposal[0], fwd_mean)
        ratio -= multivariate_normal.logpdf(
            fwd_diff, jnp.zeros_like(fwd_diff), jnp.diag(dt2)
        )

        # Backward proposal: q(position | proposal) — how likely is the current position given the proposal
        bwd_mean = proposal[0] + dt2 * d_logprob[1] / 2
        bwd_diff = periodic_diff(position, bwd_mean)
        ratio += multivariate_normal.logpdf(
            bwd_diff, jnp.zeros_like(bwd_diff), jnp.diag(dt2)
        )

        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept: Bool[Array, " n_dim"] = log_uniform < ratio

        position = jnp.where(do_accept, proposal[0], position)
        log_prob = jnp.where(do_accept, logprob[1], logprob[0])

        return position, log_prob, do_accept

    def adapt_step_size(
        self, acceptance_rate: float, target_rate: float = 0.574
    ) -> Self:
        """Adapt step size based on acceptance rate.

        Args:
            acceptance_rate: The current acceptance rate.
            target_rate: The target acceptance rate (default: 0.574 for MALA).

        Returns:
            MALA: A new instance with updated step_size.
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
        """Multiply step_size element-wise by ratios (returns a new MALA).

        Args:
            ratios: Per-dimension multiplicative scaling factors, shape (n_dim,).

        Returns:
            MALA: A new instance with updated step_size.
        """
        new_step_size = self.step_size * ratios
        return tree_at(lambda k: k.step_size, self, new_step_size)

    def print_parameters(self):
        logger.debug("MALA parameters:")
        logger.debug(f"  - step_size: {self.step_size}")

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path):
        raise NotImplementedError

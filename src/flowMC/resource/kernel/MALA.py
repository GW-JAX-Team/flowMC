import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from typing import Callable

from flowMC.resource.logPDF import LogPDF
from flowMC.resource.kernel.base import ProposalBase


class MALA(ProposalBase):
    """Metropolis-adjusted Langevin algorithm sampler class."""

    step_size: Float[Array, " n_dim"]
    periodic_mask: Bool[Array, " n_dim"]
    period: Float[Array, " n_dim"]
    periodic_lower_bound: Float[Array, " n_dim"]

    def __repr__(self):
        return "MALA with step size " + str(self.step_size)

    def __init__(
        self,
        step_size: Float[Array, " n_dim"],
        periodic_mask: Bool[Array, " n_dim"],
        periodic_bounds: Float[Array, " n_dim 2"],
    ):
        """Initialize MALA sampler.

        Args:
            step_size: Step size for the MALA sampler as a 1D array representing
                      diagonal elements of the step size matrix.
        """
        super().__init__()
        self.step_size = step_size
        self.periodic_mask = periodic_mask
        self.period = periodic_bounds[:, 1] - periodic_bounds[:, 0]
        self.period = jnp.where(
            self.periodic_mask, self.period, jnp.ones_like(self.period)
        )
        self.periodic_lower_bound = periodic_bounds[:, 0]

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        """Metropolis-adjusted Langevin algorithm kernel for a single chain.

        Args:
            rng_key (PRNGKeyArray): JAX PRNGKey for stochastic operations.
            position (Float[Array, " n_dim"]): Current position of the chain.
            log_prob (Float[Array, "1"]): Current log-probability of the chain.
            logpdf: Log probability density function to evaluate.
            data (PyTree): Additional data to pass to the logpdf function.
        Returns:
            Tuple of (new_position, new_log_prob, acceptance_flag):
            - new_position: New position of the chain.
            - new_log_prob: New log-probability of the chain.
            - acceptance_flag: Whether the new position is accepted.
        """

        def body(
            carry: tuple[Float[Array, " n_dim"], Float[Array, " n_dim"], dict],
            this_key: PRNGKeyArray,
        ) -> tuple[
            tuple[Float[Array, " n_dim"], Float[Array, " n_dim"], dict],
            tuple[Float[Array, " n_dim"], Float[Array, "1"], Float[Array, " n_dim"]],
        ]:
            print("Compiling MALA body")
            this_position, dt, data = carry
            dt2 = dt * dt
            this_log_prob, this_d_log = jax.value_and_grad(logpdf)(this_position, data)
            # MALA proposal: x' = x + (dt²/2) * ∇log p(x) + dt * ε, where ε ~ N(0, I)
            proposal = this_position + dt2 * this_d_log / 2
            proposal += dt * jax.random.normal(this_key, shape=this_position.shape)
            proposal = jnp.where(
                self.periodic_mask,
                jnp.mod(proposal - self.periodic_lower_bound, self.period)
                + self.periodic_lower_bound,
                proposal,
            )
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

        # Metropolis-Hastings ratio: log[p(proposal)/p(position)] + log[q(position|proposal)/q(proposal|position)]
        ratio = logprob[1] - logprob[0]
        ratio -= multivariate_normal.logpdf(
            proposal[0], position + dt2 * d_logprob[0] / 2, jnp.diag(dt2)
        )
        ratio += multivariate_normal.logpdf(
            position, proposal[0] + dt2 * d_logprob[1] / 2, jnp.diag(dt2)
        )

        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept: Bool[Array, " n_dim"] = log_uniform < ratio

        position = jnp.where(do_accept, proposal[0], position)
        log_prob = jnp.where(do_accept, logprob[1], logprob[0])

        return position, log_prob, do_accept

    def print_parameters(self):
        print("MALA parameters:")
        print(f"step_size: {self.step_size}")

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path):
        raise NotImplementedError

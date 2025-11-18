import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from typing import Callable, Union

from flowMC.resource.logPDF import LogPDF
from flowMC.resource.kernel.base import ProposalBase


class MALA(ProposalBase):
    """Metropolis-adjusted Langevin algorithm sampler class."""

    step_size: Float

    def __repr__(self):
        return "MALA with step size " + str(self.step_size)

    def __init__(
        self,
        step_size: Union[float, Float[Array, " n_dim n_dim"]],
    ):
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
        """Metropolis-adjusted Langevin algorithm kernel. This is a kernel that only
        evolve a single chain.

        Args:
            rng_key (PRNGKeyArray): Jax PRNGKey
            position (Float[Array, " n_dim"]): current position of the chain
            log_prob (Float[Array, "1"]): current log-probability of the chain
            data (PyTree): data to be passed to the logpdf function

        Returns:
            position (Float[Array, " n_dim"]): new position of the chain
            log_prob (Float[Array, "1"]): new log-probability of the chain
            do_accept (Int[Array, "1"]): whether the new position is accepted
        """

        def body(
            carry: tuple[
                Float[Array, " n_dim"], Union[float, Float[Array, " n_dim n_dim"]], dict
            ],
            this_key: PRNGKeyArray,
        ) -> tuple[
            tuple[
                Float[Array, " n_dim"], Union[float, Float[Array, " n_dim n_dim"]], dict
            ],
            tuple[Float[Array, " n_dim"], Float[Array, "1"], Float[Array, " n_dim"]],
        ]:
            print("Compiling MALA body")
            this_position, dt, data = carry
            dt2 = dt * dt
            this_log_prob, this_d_log = jax.value_and_grad(logpdf)(this_position, data)
            proposal = this_position + jnp.dot(dt2, this_d_log) / 2
            proposal += jnp.dot(
                dt, jax.random.normal(this_key, shape=this_position.shape)
            )
            return (proposal, dt, data), (proposal, this_log_prob, this_d_log)

        key1, key2 = jax.random.split(rng_key)

        dt: Union[float, Float[Array, " n_dim n_dim"]] = self.step_size
        dt2 = dt * dt

        # Use scan to iterate twice: first to generate proposal from current position
        # and compute its log_prob and gradient, then to compute log_prob and gradient
        # at the proposed position. Note: proposal[1] from the second iteration is
        # discarded; we only need logprob[1] and d_logprob[1] for the acceptance ratio.
        # Using the same key twice is fine since proposal[1] is unused.
        _, (proposal, logprob, d_logprob) = jax.lax.scan(
            body, (position, dt, data), jnp.array([key1, key1])
        )

        # Compute Metropolis-Hastings acceptance ratio for MALA
        # ratio = log(p(proposal) / p(position)) + log(q(position|proposal) / q(proposal|position))
        ratio = logprob[1] - logprob[0]
        # Subtract log of forward proposal density q(proposal|position)
        ratio -= multivariate_normal.logpdf(
            proposal[0], position + jnp.dot(dt2, d_logprob[0]) / 2, dt2
        )
        # Add log of reverse proposal density q(position|proposal)
        ratio += multivariate_normal.logpdf(
            position, proposal[0] + jnp.dot(dt2, d_logprob[1]) / 2, dt2
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

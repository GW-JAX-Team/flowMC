import logging
from collections.abc import Callable
from typing import Optional, Self

import jax
import jax.numpy as jnp
from equinox import tree_at
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Bool, Float, Key, PyTree

from flowMC.resource.kernel.base import ProposalBase
from flowMC.resource.logPDF import LogPDF

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
        rng_key: Key[Array, ""],
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Bool[Array, "1"]]:
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
        # The second key draws a proposal that is never evaluated
        positions, log_probs, do_accepts = self._run_chain(
            jnp.array([rng_key, rng_key]), position, logpdf, data
        )
        return positions[0], log_probs[0], do_accepts[0]

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
        """Take ``n_steps`` MALA steps from ``position``.

        Returns the position, log-probability and acceptance flag after each step.
        """

        def split(key, _):
            key, subkey = jax.random.split(key)
            return key, subkey

        # Step ``i`` uses the ``i``-th key split off ``rng_key``; the last is unused
        _, step_keys = jax.lax.scan(split, rng_key, length=n_steps + 1)
        return self._run_chain(step_keys, position, logpdf, data)

    def _run_chain(
        self,
        step_keys: Key[Array, " n_steps_plus_1"],
        position: Float[Array, " n_dim"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[
        Float[Array, "n_steps n_dim"], Float[Array, " n_steps"], Bool[Array, " n_steps"]
    ]:
        """Take one MALA step per key in ``step_keys`` but the last.

        Each scan iteration evaluates the log-pdf and its gradient at one point,
        the start and then each proposal, so the gradient is compiled once. The
        evaluation decides the pending proposal and gives the gradient that draws
        the next one.
        """
        logger.debug("Compiling MALA chain")
        periodic_mask = self.periodic_mask
        periodic_bounds = self.periodic_bounds
        lower = periodic_bounds[:, 0]
        upper = periodic_bounds[:, 1]
        period = upper - lower

        def wrap_periodic(x: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
            """Wrap periodic dimensions into [lower, upper)."""
            wrapped = lower + jnp.mod(x - lower, period)
            return jnp.where(periodic_mask, wrapped, x)

        def periodic_diff(
            a: Float[Array, " n_dim"], b: Float[Array, " n_dim"]
        ) -> Float[Array, " n_dim"]:
            """Compute minimum-image difference (a - b) for periodic dimensions."""
            raw_diff = a - b
            # For periodic dims, use nearest image convention
            periodic_diff_val = raw_diff - period * jnp.round(raw_diff / period)
            return jnp.where(periodic_mask, periodic_diff_val, raw_diff)

        dt: Float[Array, " n_dim"] = self.step_size
        dt2 = dt * dt

        def body(carry, xs):
            accept_key, position, log_prob, grad, proposal = carry
            first, step_key = xs
            proposal_log_prob, proposal_grad = jax.value_and_grad(logpdf)(
                proposal, data
            )

            # Metropolis-Hastings ratio: log[p(proposal)/p(position)] + log[q(position|proposal)/q(proposal|position)]
            # For periodic dimensions, use minimum-image distances in the proposal density
            ratio = proposal_log_prob - log_prob

            # Forward proposal: q(proposal | position) — how likely is the proposal given current position
            fwd_mean = position + dt2 * grad / 2
            fwd_diff = periodic_diff(proposal, fwd_mean)
            ratio -= multivariate_normal.logpdf(
                fwd_diff, jnp.zeros_like(fwd_diff), jnp.diag(dt2)
            )

            # Backward proposal: q(position | proposal) — how likely is the current position given the proposal
            bwd_mean = proposal + dt2 * proposal_grad / 2
            bwd_diff = periodic_diff(position, bwd_mean)
            ratio += multivariate_normal.logpdf(
                bwd_diff, jnp.zeros_like(bwd_diff), jnp.diag(dt2)
            )

            # The first iteration evaluates the start, which is always kept
            log_uniform = jnp.log(jax.random.uniform(accept_key))
            do_accept = first | (log_uniform < ratio)

            position = jnp.where(do_accept, proposal, position)
            log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
            grad = jnp.where(do_accept, proposal_grad, grad)

            # MALA proposal: x' = x + (dt²/2) * ∇log p(x) + dt * ε, where ε ~ N(0, I)
            noise_key, accept_key = jax.random.split(step_key)
            proposal = position + dt2 * grad / 2
            proposal += dt * jax.random.normal(noise_key, shape=position.shape)
            proposal = wrap_periodic(proposal)
            return (accept_key, position, log_prob, grad, proposal), (
                position,
                log_prob,
                do_accept,
            )

        # The first iteration sets the start's log-prob and gradient, so these are
        # placeholders. Deriving them from ``position`` makes them batched under
        # vmap, which then batches the scan body once instead of twice.
        zeros = 0 * position
        log_prob = jax.eval_shape(logpdf, position, data)
        carry = (
            step_keys[0],
            position,
            jnp.broadcast_to(zeros[0], log_prob.shape).astype(log_prob.dtype),
            zeros,
            position,
        )
        _, (positions, log_probs, do_accepts) = jax.lax.scan(
            body, carry, (jnp.arange(step_keys.shape[0]) == 0, step_keys)
        )
        return positions[1:], log_probs[1:], do_accepts[1:]

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

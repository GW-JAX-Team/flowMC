import logging
from collections.abc import Callable
from typing import Optional, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import tree_at
from jaxtyping import Array, Bool, Float, Key, PyTree

from flowMC.resource.kernel.base import ProposalBase
from flowMC.resource.logPDF import LogPDF

logger = logging.getLogger(__name__)


class HMC(ProposalBase):
    """Hamiltonian Monte Carlo sampler class builiding the hmc_sampler method from
    target logpdf.

    Args:
        condition_matrix (Array): Diagonal elements of the mass matrix.
        step_size (Float): Step size for leapfrog integration.
        n_leapfrog (Int): Number of leapfrog steps.
        periodic_mask (Array): Boolean mask for periodic dimensions.
        periodic_bounds (Array): Bounds for periodic dimensions.
    """

    condition_matrix: Float[Array, " n_dim"]
    step_size: float
    n_leapfrog: int = eqx.field(static=True)
    periodic_mask: Bool[Array, " n_dim"]
    periodic_bounds: Float[Array, "n_dim 2"]
    ADAPTATION_RATE: float = 0.5

    def __repr__(self):
        return (
            "HMC with step size "
            + str(self.step_size)
            + " and "
            + str(self.n_leapfrog)
            + " leapfrog steps"
        )

    def __init__(
        self,
        condition_matrix: Float[Array, " n_dim"],
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        periodic_mask: Optional[Bool[Array, " n_dim"]] = None,
        periodic_bounds: Optional[Float[Array, "n_dim 2"]] = None,
    ):
        """Initialize HMC sampler.

        Args:
            condition_matrix: Diagonal elements of the mass matrix as a 1D array.
            step_size: Step size for leapfrog integration.
            n_leapfrog: Number of leapfrog steps.
            periodic_mask: Boolean mask indicating which dimensions are periodic.
                If None, no periodic boundaries are applied.
            periodic_bounds: Array of shape (n_dim, 2) with [lower, upper] bounds
                for each periodic dimension. Only used where periodic_mask is True.
                If None, no periodic boundaries are applied.
        """
        self.condition_matrix = condition_matrix
        self.step_size = step_size

        self.n_leapfrog = n_leapfrog

        n_dim = (
            jnp.asarray(condition_matrix).shape[0]
            if jnp.asarray(condition_matrix).ndim > 0
            else 1
        )
        if periodic_mask is None:
            periodic_mask = jnp.zeros(n_dim, dtype=bool)
        if periodic_bounds is None:
            periodic_bounds = jnp.zeros((n_dim, 2))
        self.periodic_mask = periodic_mask
        self.periodic_bounds = periodic_bounds

    def get_initial_hamiltonian(
        self,
        potential: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        kinetic: Callable[
            [Float[Array, " n_dim"], Float[Array, " n_dim"]], Float[Array, "1"]
        ],
        rng_key: Key[Array, ""],
        position: Float[Array, " n_dim"],
        data: PyTree,
    ):
        """Compute the value of the Hamiltonian from positions with initial momentum
        draw at random from the standard normal distribution."""

        # Sample momentum from N(0, M^-1) where M is diagonal mass matrix
        momentum = jax.random.normal(rng_key, shape=position.shape) / jnp.sqrt(
            self.condition_matrix
        )
        return potential(position, data) + kinetic(momentum, self.condition_matrix)

    def init_cache(
        self,
        position: Float[Array, " n_dim"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, "1"], Float[Array, " n_dim"]]:
        """Return the log-probability at ``position`` and its gradient, the cache."""
        return jax.value_and_grad(logpdf)(position, data)

    def leapfrog(
        self,
        position: Float[Array, " n_dim"],
        momentum: Float[Array, " n_dim"],
        grad: Float[Array, " n_dim"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[
        Float[Array, " n_dim"],
        Float[Array, " n_dim"],
        Float[Array, "1"],
        Float[Array, " n_dim"],
    ]:
        """Integrate ``n_leapfrog`` leapfrog steps from ``position``.

        ``grad`` is the log-pdf gradient at ``position``.  Each step evaluates the
        gradient once, and the log-pdf is evaluated once at the end: a gradient
        alone is cheaper than ``value_and_grad``.  Returns the end position and
        momentum with the log-pdf and its gradient there.
        """
        logger.debug("Compiling leapfrog step")
        lower = self.periodic_bounds[:, 0]
        upper = self.periodic_bounds[:, 1]
        period = upper - lower

        def body(carry, _):
            position, momentum, grad = carry
            momentum = momentum + 0.5 * self.step_size * grad
            # dK/dp for the kinetic energy K(p) = (1/2) * p^T * M * p, M diagonal
            position = position + self.step_size * momentum * self.condition_matrix
            # Wrap periodic dimensions after position update
            wrapped = lower + jnp.mod(position - lower, period)
            position = jnp.where(self.periodic_mask, wrapped, position)
            grad = jax.grad(logpdf)(position, data)
            momentum = momentum + 0.5 * self.step_size * grad
            return (position, momentum, grad), None

        (position, momentum, grad), _ = jax.lax.scan(
            body, (position, momentum, grad), length=self.n_leapfrog
        )
        return position, momentum, logpdf(position, data), grad

    def kernel(
        self,
        rng_key: Key[Array, ""],
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Bool[Array, "1"]]:
        """Note that since the potential function is the negative log likelihood,
        hamiltonian is going down, but the likelihood value should go up.

        Evaluates the gradient at ``position`` before taking the step; chains that
        take many steps should use `step`, which reuses it.

        Args:
            rng_key (Key): Random key.
            position (Array): Current position.
            log_prob (Array): Log probability of the current position.
            logpdf (LogPDF | Callable): Log probability density function.
            data (PyTree): Additional data for the logpdf.

        Returns:
            tuple: (position, log_prob, do_accept).
        """
        _, grad = self.init_cache(position, logpdf, data)
        position, log_prob, _, do_accept = self.step(
            rng_key, position, log_prob, grad, logpdf, data
        )
        return position, log_prob, do_accept

    def step(
        self,
        rng_key: Key[Array, ""],
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        cache: Float[Array, " n_dim"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[
        Float[Array, " n_dim"],
        Float[Array, "1"],
        Float[Array, " n_dim"],
        Bool[Array, "1"],
    ]:
        """One HMC step, where ``cache`` is the log-pdf gradient at ``position``."""

        def kinetic(p: Float[Array, " n_dim"]) -> Float[Array, "1"]:
            # Kinetic energy: K(p) = (1/2) * p^T * M * p where M is diagonal
            return 0.5 * jnp.sum(p**2 * self.condition_matrix)

        key1, key2 = jax.random.split(rng_key)

        # Sample momentum from N(0, M^-1) where M is diagonal mass matrix
        momentum: Float[Array, " n_dim"] = jax.random.normal(
            key1, shape=position.shape
        ) / jnp.sqrt(self.condition_matrix)
        H = -log_prob + kinetic(momentum)
        proposed_position, proposed_momentum, proposed_log_prob, proposed_grad = (
            self.leapfrog(position, momentum, cache, logpdf, data)
        )
        proposed_ham = -proposed_log_prob + kinetic(proposed_momentum)
        log_acc = H - proposed_ham
        log_uniform = jnp.log(jax.random.uniform(key2))

        do_accept = log_uniform < log_acc

        position = jnp.where(do_accept, proposed_position, position)
        log_prob = jnp.where(do_accept, proposed_log_prob, log_prob)
        grad = jnp.where(do_accept, proposed_grad, cache)

        return position, log_prob, grad, do_accept

    def adapt_step_size(
        self, acceptance_rate: float, target_rate: float = 0.65
    ) -> Self:
        """Adapt step size based on acceptance rate.

        Args:
            acceptance_rate: The current acceptance rate.
            target_rate: The target acceptance rate (default: 0.65 for HMC).

        Returns:
            HMC: A new instance with updated step_size.
        """
        diff = acceptance_rate - target_rate
        new_step_size = self.step_size * (1.0 + self.ADAPTATION_RATE * diff)
        return tree_at(lambda k: k.step_size, self, new_step_size)

    def get_effective_dim_profile(self) -> Float[Array, " n_dim"]:
        """Return the current per-dim effective step profile normalised to geometric mean 1.

        The effective step size in dimension i is step_size / sqrt(condition_matrix[i]),
        so the per-dim profile is 1/sqrt(condition_matrix[i]) normalised to geomean 1.
        """
        safe = jnp.maximum(
            self.condition_matrix, jnp.finfo(self.condition_matrix.dtype).eps
        )
        log_eff = -0.5 * jnp.log(safe)
        return jnp.exp(log_eff - jnp.mean(log_eff))

    def apply_per_dim_scaling(self, ratios: Float[Array, " n_dim"]) -> Self:
        """Scale the effective per-dimension step size by ratios via the mass matrix.

        The effective step size in dimension i is step_size / sqrt(condition_matrix[i]).
        Multiplying that by ratios[i] is equivalent to dividing condition_matrix[i]
        by ratios[i]^2. The geometric mean of condition_matrix is preserved, so
        adapt_step_size retains control of the global scale.

        Args:
            ratios: Per-dimension scaling factors for the effective step size, shape
                (n_dim,). Values > 1 increase the effective step in that dimension.

        Returns:
            HMC: A new instance with updated condition_matrix.
        """
        new_condition_matrix = self.condition_matrix / (ratios * ratios)
        return tree_at(lambda k: k.condition_matrix, self, new_condition_matrix)

    def print_parameters(self):
        logger.debug("HMC parameters:")
        logger.debug(f"  - step_size: {self.step_size}")
        logger.debug(f"  - n_leapfrog: {self.n_leapfrog}")
        logger.debug(f"  - condition_matrix: {self.condition_matrix}")

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path):
        raise NotImplementedError

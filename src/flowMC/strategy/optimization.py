from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Key

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
import logging

logger = logging.getLogger(__name__)


class AdamOptimization(Strategy):
    """Optimize a set of chains using Adam optimization.

    .. note::
        If the posterior is unbounded, the optimization may return NaNs.

    Attributes:
        logpdf (Callable): Log-probability density function ``f(x, data) -> Float``.
        n_steps (int): Number of optimization steps.
        learning_rate (float): Adam learning rate.
        noise_level (float): Noise scale multiplied onto the gradient at each step.
        bounds (Float[Array, "n_dim 2"]): Box constraints; shape ``(n_dim, 2)`` or
            ``(1, 2)`` (broadcast to all dimensions).
    """

    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]
    n_steps: int = 100
    learning_rate: float = 1e-2
    noise_level: float = 10
    bounds: Float[Array, "n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]])

    def __repr__(self):
        return "AdamOptimization"

    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_steps: int = 100,
        learning_rate: float = 1e-2,
        noise_level: float = 10,
        bounds: Float[Array, "n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]]),
    ) -> None:
        """
        Args:
            logpdf (Callable): Log-PDF ``f(x, data) -> Float`` to maximise.
            n_steps (int): Number of Adam steps. Defaults to 100.
            learning_rate (float): Adam learning rate. Defaults to 1e-2.
            noise_level (float): Noise scale added to gradients each step.
                Defaults to 10.
            bounds (Float[Array, "n_dim 2"]): Box constraints of shape ``(n_dim, 2)``
                or ``(1, 2)`` (broadcast). Defaults to ``[[-inf, inf]]`` (unconstrained).
        """
        self.logpdf = logpdf
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        self.bounds = bounds

        # Validate bounds shape
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                f"bounds must have shape (n_dim, 2) or (1, 2), got {bounds.shape}"
            )
        # If bounds is (1, 2), it will be broadcast to all dimensions. If not, check compatibility.
        # Try to infer n_dim from logpdf signature or initial_position, but here we can't, so warn in runtime.

        self.solver = optax.chain(
            optax.adam(learning_rate=self.learning_rate),
        )

    def __call__(
        self,
        rng_key: Key,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chain n_dim"],
        data: dict,
    ) -> tuple[
        Key,
        dict[str, Resource],
        Float[Array, "n_chain n_dim"],
    ]:
        """Optimise all chains toward the mode of the log-PDF.

        Args:
            rng_key (Key): JAX PRNGKey (consumed and split internally).
            resources (dict[str, Resource]): Resource dictionary (returned unchanged).
            initial_position (Float[Array, "n_chain n_dim"]): Starting positions.
            data (dict): Auxiliary data passed to the log-PDF.

        Returns:
            tuple: ``(rng_key, resources, optimized_positions)``.
        """

        def loss_fn(params: Float[Array, " n_dim"], data: dict) -> Float:
            return -self.logpdf(params, data)

        rng_key, optimized_positions, _ = self.optimize(
            rng_key, loss_fn, initial_position, data
        )

        return rng_key, resources, optimized_positions

    def optimize(
        self,
        rng_key: Key,
        objective: Callable,
        initial_position: Float[Array, "n_chain n_dim"],
        data: dict,
    ) -> tuple[Key, Float[Array, "n_chain n_dim"], Float[Array, " n_chain"]]:
        """Optimization kernel that can be used independently of ``__call__``.

        Args:
            rng_key (Key): JAX PRNGKey for noise generation.
            objective (Callable): Scalar-valued objective to *minimise*
                (pass the negated log-PDF to maximise it).
            initial_position (Float[Array, "n_chain n_dim"]): Starting positions.
            data (dict): Auxiliary data passed to the objective.

        Returns:
            tuple:
                - rng_key (Key): Updated PRNGKey.
                - optimized_positions (Float[Array, "n_chain n_dim"]): Final positions.
                - final_log_prob (Float[Array, "n_chain"]): Log-PDF at final positions.
        """
        # Validate bounds shape against n_dim
        n_dim = initial_position.shape[-1]
        if not (self.bounds.shape[0] == 1 or self.bounds.shape[0] == n_dim):
            raise ValueError(
                f"bounds shape {self.bounds.shape} is incompatible with n_dim={n_dim}. "
                "Provide bounds of shape (1, 2) for broadcasting or (n_dim, 2) for per-dimension bounds."
            )

        grad_fn = jax.jit(jax.grad(objective))

        def _kernel(carry, _step):
            key, params, opt_state = carry

            key, subkey = jax.random.split(key)
            grad = grad_fn(params, data) * (
                1 + jax.random.normal(subkey) * self.noise_level
            )
            updates, opt_state = self.solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = optax.projections.projection_box(
                params, self.bounds[:, 0], self.bounds[:, 1]
            )
            return (key, params, opt_state), None

        def _single_optimize(
            key: Key,
            initial_position: Float[Array, "n_dim"],
        ) -> Float[Array, "n_dim"]:
            opt_state = self.solver.init(initial_position)

            (key, params, opt_state), _ = jax.lax.scan(
                _kernel,
                (key, initial_position, opt_state),
                jnp.arange(self.n_steps),
            )

            return params  # type: ignore

        logger.info("Using Adam optimization")
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, initial_position.shape[0])
        optimized_positions = jax.vmap(_single_optimize, in_axes=(0, 0))(
            keys, initial_position
        )

        final_log_prob = jax.vmap(self.logpdf, in_axes=(0, None))(
            optimized_positions, data
        )

        if jnp.isinf(final_log_prob).any() or jnp.isnan(final_log_prob).any():
            logger.warning("Optimization accessed infinite or NaN log-probabilities.")

        return rng_key, optimized_positions, final_log_prob

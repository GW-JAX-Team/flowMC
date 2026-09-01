import logging
from abc import abstractmethod
from typing import Optional, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Key
from tqdm import tqdm, trange

from flowMC.resource.base import Resource
from flowMC.typing import FloatLike, FloatScalar

logger = logging.getLogger(__name__)


class NFModel(eqx.Module, Resource):
    """Base class for normalizing flow models.

    This is an abstract template that should not be directly used.
    """

    _n_features: int
    _data_mean: Float[Array, " n_dim"]
    _data_cov: Float[Array, "n_dim n_dim"]

    @property
    def n_features(self):
        return self._n_features

    @property
    def data_mean(self):
        return jax.lax.stop_gradient(self._data_mean)

    @property
    def data_cov(self):
        return jax.lax.stop_gradient(jnp.atleast_2d(self._data_cov))

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(
        self, x: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_dim"], FloatScalar]:
        """Forward pass of the model.

        Args:
            x (Float[Array, "n_dim"]): Input data.

        Returns:
            tuple[Float[Array, "n_dim"], Float]:
                Output data and log determinant of the Jacobian.
        """
        return self.forward(x)

    @abstractmethod
    def log_prob(
        self, x: Float[Array, " n_dim"] | Float[Array, "n_sample n_dim"]
    ) -> FloatScalar | Float[Array, " n_sample"]:
        raise NotImplementedError

    @abstractmethod
    def sample(self, rng_key: Key, n_samples: int) -> Array:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, x: Float[Array, " n_dim"], key: Optional[Key] = None
    ) -> tuple[Float[Array, " n_dim"], FloatScalar]:
        """Forward pass of the model.

        Args:
            x (Float[Array, "n_dim"]): Input data.

        Returns:
            tuple[Float[Array, "n_dim"], Float]:
                Output data and log determinant of the Jacobian.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(
        self, x: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_dim"], FloatScalar]:
        """Inverse pass of the model.

        Args:
            x (Float[Array, "n_dim"]): Input data.

        Returns:
            tuple[Float[Array, "n_dim"], Float]:
                Output data and log determinant of the Jacobian.
        """
        raise NotImplementedError

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path + ".eqx", self)

    def load_model(self, path: str) -> Self:
        return eqx.tree_deserialise_leaves(path + ".eqx", self)

    @eqx.filter_value_and_grad
    def loss_fn(self, x: Float[Array, "n_batch n_dim"]) -> FloatScalar:
        return -jnp.mean(jax.vmap(self.log_prob)(x))

    @eqx.filter_jit
    def train_step(
        model: Self,
        x: Float[Array, "n_batch n_dim"],
        optim: optax.GradientTransformation,
        state: optax.OptState,
    ) -> tuple[FloatScalar, Self, optax.OptState]:
        """Train for a single step.

        Args:
            x (Array): Training data.
            optim (optax.GradientTransformation): Optimizer.
            state (optax.OptState): Optimizer state.

        Returns:
            tuple: (loss, model, state).
        """
        logger.debug("Compiling training step")
        loss, grads = model.loss_fn(x)
        updates, state = optim.update(
            grads, state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return loss, model, state

    def train_epoch(
        self: Self,
        rng: Key,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        data: Float[Array, "n_example n_dim"],
        batch_size: int,
    ) -> tuple[FloatLike, Self, optax.OptState]:
        """Train for a single epoch."""
        value = 1e9
        model = self
        train_ds_size = len(data)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = data[perm, ...]
                value, model, state = model.train_step(batch, optim, state)
        else:
            value, model, state = model.train_step(data, optim, state)

        return value, model, state

    def train(
        self: Self,
        rng: Key,
        data: Array,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        num_epochs: int,
        batch_size: int,
        verbose: bool = True,
    ) -> tuple[Key, Self, optax.OptState, Array]:
        """Train a normalizing flow model.

        Args:
            rng (Key): JAX PRNGKey.
            data (Array): Training data.
            optim (optax.GradientTransformation): Optimizer.
            state (optax.OptState): Optimizer state.
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            verbose (bool): Whether to print progress.

        Returns:
            rng (Key): Updated JAX PRNGKey.
            model (Self): Best model found during training.
            state (optax.OptState): Optimizer state corresponding to the best model.
            loss_values (Array): Loss values for each epoch.
        """
        loss_values = jnp.zeros(num_epochs)
        if verbose:
            pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        else:
            pbar = range(num_epochs)

        best_model = model = self
        best_state = state
        best_loss = 1e9
        model = eqx.tree_at(lambda m: m._data_mean, model, jnp.mean(data, axis=0))
        model = eqx.tree_at(lambda m: m._data_cov, model, jnp.cov(data.T))
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, model, state = model.train_epoch(
                input_rng, optim, state, data, batch_size
            )
            loss_values = loss_values.at[epoch].set(value)
            if loss_values[epoch] < best_loss:
                best_model = model
                best_state = state
                best_loss = loss_values[epoch]
            if verbose:
                assert isinstance(pbar, tqdm)
                if num_epochs > 10:
                    if epoch % int(num_epochs / 10) == 0:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")
                else:
                    if epoch == num_epochs:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")

        return rng, best_model, best_state, loss_values

    def to_precision(self, precision: str = "float32") -> Self:
        """Convert all parameters to a given precision.

        !!! warning
            This function is **experimental** and may change in the future.

        Args:
            precision (str): Precision to convert to.

        Returns:
            Self: Model with parameters converted to the given precision.
        """

        precisions_dict = {
            "float16": jnp.float16,
            "bfloat16": jnp.bfloat16,
            "float32": jnp.float32,
            "float64": jnp.float64,
        }
        try:
            precision_format = precisions_dict[precision.lower()]
        except KeyError:
            raise ValueError(
                f"Precision {precision} not supported.\
                Choose from {precisions_dict.keys()}"
            )
        dynamic_model, static_model = eqx.partition(self, eqx.is_array)
        dynamic_model = jax.tree.map(
            lambda x: x.astype(precision_format), dynamic_model
        )
        return eqx.combine(dynamic_model, static_model)

    save_resource = save_model
    load_resource = load_model

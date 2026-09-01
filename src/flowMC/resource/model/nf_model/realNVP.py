import logging
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from flowMC.resource.model.common import (
    MLP,
    Distribution,
    Gaussian,
    MaskedCouplingLayer,
    MLPAffine,
)
from flowMC.resource.model.nf_model.base import NFModel
from flowMC.typing import FloatScalar

logger = logging.getLogger(__name__)


class AffineCoupling(eqx.Module):
    """
    Affine coupling layer.
    (Defined in the RealNVP paper https://arxiv.org/abs/1605.08803)
    We use tanh as the default activation function.

    Args:
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        mask: (ndarray) Alternating mask for the affine coupling layer.
        dt: (Float) Scaling factor for the affine coupling layer.
    """

    _mask: Array
    scale_MLP: MLP
    translate_MLP: MLP
    dt: float = 1

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        mask: Array,
        key: Key,
        dt: float = 1,
        scale: float = 1e-4,
    ):
        self._mask = mask
        self.dt = dt
        key, scale_subkey, translate_subkey = jax.random.split(key, 3)
        features = [n_features, n_hidden, n_features]
        self.scale_MLP = MLP(features, key=scale_subkey, scale=scale)
        self.translate_MLP = MLP(features, key=translate_subkey, scale=scale)

    @property
    def mask(self):
        return jax.lax.stop_gradient(self._mask)

    @property
    def n_features(self):
        return self.scale_MLP.n_input

    def __call__(self, x: Array):
        return self.forward(x)

    def forward(self, x: Array) -> tuple[Array, Array]:
        """From latent space to data space.

        Args:
            x: (Array) Latent space.

        Returns:
            outputs: (Array) Data space.
            log_det: (Array) Log determinant of the Jacobian.
        """
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s) * self.dt
        t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt

        # Compute log determinant of the Jacobian
        log_det = s.sum()

        # Apply the transformation
        outputs = (x + t) * jnp.exp(s)
        return outputs, log_det

    def inverse(self, x: Array) -> tuple[Array, Array]:
        """From data space to latent space.

        Args:
            x: (Array) Data space.

        Returns:
            outputs: (Array) Latent space.
            log_det: (Array) Log determinant of the Jacobian.
        """
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s) * self.dt
        t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt
        log_det = -s.sum()
        outputs = x * jnp.exp(-s) - t
        return outputs, log_det


class RealNVP(NFModel):
    """
    RealNVP mode defined in the paper https://arxiv.org/abs/1605.08803.
    MLP is needed to make sure the scaling between layers are more or less the same.

    Args:
        n_features (int): The number of features in the input.
        n_layers (int): The number of affine coupling layers.
        n_hidden (int): The number of hidden units in the MLP.

    Properties:
        data_mean: (ndarray) Mean of Gaussian base distribution
        data_cov: (ndarray) Covariance of Gaussian base distribution
    """

    base_dist: Distribution
    affine_coupling: list[MaskedCouplingLayer]
    _n_features: int
    _data_mean: Float[Array, " n_dim"]
    _data_cov: Float[Array, "n_dim n_dim"]

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def data_mean(self):
        return jax.lax.stop_gradient(self._data_mean)

    @property
    def data_cov(self):
        return jax.lax.stop_gradient(jnp.atleast_2d(self._data_cov))

    def __init__(
        self, n_features: int, n_layers: int, n_hidden: int, key: Key, **kwargs
    ) -> None:
        """
        Args:
            n_features (int): Dimensionality of the data.
            n_layers (int): Number of affine coupling layers.
            n_hidden (int): Number of hidden units per coupling-layer MLP.
            key (Key): JAX PRNGKey for parameter initialisation.
            **kwargs: Optional overrides:
                - ``base_dist`` (Distribution): Custom base distribution.
                  Defaults to a standard Gaussian.
                - ``data_mean`` (Array): Initial data mean for whitening.
                  Updated during training.
                - ``data_cov`` (Array): Initial data covariance for whitening.
                  Updated during training.
        """
        base_dist = kwargs.get("base_dist")
        if base_dist is not None:
            if not isinstance(base_dist, Distribution):
                raise TypeError("base_dist must be a Distribution")
            self.base_dist = base_dist
        else:
            self.base_dist = Gaussian(
                jnp.zeros(n_features), jnp.eye(n_features), learnable=False
            )

        if kwargs.get("data_mean") is not None:
            data_mean = kwargs.get("data_mean")
            assert isinstance(data_mean, Array)
            self._data_mean = data_mean
        else:
            self._data_mean = jnp.zeros(n_features)

        if kwargs.get("data_cov") is not None:
            data_cov = kwargs.get("data_cov")
            assert isinstance(data_cov, Array)
            self._data_cov = data_cov
        else:
            self._data_cov = jnp.eye(n_features)

        self._n_features = n_features

        def make_layer(i: int, key: Key):
            key, scale_subkey, shift_subkey = jax.random.split(key, 3)
            mask = jnp.ones(n_features)
            mask = mask.at[: int(n_features / 2)].set(0)
            mask = jax.lax.cond(i % 2 == 0, lambda x: 1 - x, lambda x: x, mask)
            scale_MLP = MLP([n_features, n_hidden, n_features], key=scale_subkey)
            shift_MLP = MLP([n_features, n_hidden, n_features], key=shift_subkey)
            return MaskedCouplingLayer(MLPAffine(scale_MLP, shift_MLP), mask)

        keys = jax.random.split(key, n_layers)
        self.affine_coupling = eqx.filter_vmap(make_layer)(jnp.arange(n_layers), keys)

    def forward(
        self,
        x: Float[Array, " n_dim"],
        key: Optional[Key] = None,
        condition: Optional[Float[Array, " n_condition"]] = None,
    ) -> tuple[Float[Array, " n_dim"], FloatScalar]:
        log_det: FloatScalar = jnp.zeros(())
        dynamics, statics = eqx.partition(self.affine_coupling, eqx.is_array)

        def f(carry, data):
            x, log_det = carry
            layers = eqx.combine(data, statics)
            x, log_det_i = layers(x, condition)
            return (x, log_det + log_det_i), None

        (x, log_det), _ = jax.lax.scan(f, (x, log_det), dynamics)
        return x, log_det

    def inverse(
        self,
        x: Float[Array, " n_dim"],
        condition: Optional[Float[Array, " n_condition"]] = None,
    ) -> tuple[Float[Array, " n_dim"], FloatScalar]:
        """From latent space to data space."""
        log_det: FloatScalar = jnp.zeros(())
        dynamics, statics = eqx.partition(self.affine_coupling, eqx.is_array)

        def f(carry, data):
            x, log_det = carry
            layers = eqx.combine(data, statics)
            x, log_det_i = layers.inverse(x, condition)
            return (x, log_det + log_det_i), None

        (x, log_det), _ = jax.lax.scan(f, (x, log_det), dynamics, reverse=True)
        return x, log_det

    def sample(self, rng_key: Key, n_samples: int) -> Array:
        samples = self.base_dist.sample(rng_key, n_samples)
        samples = jax.vmap(self.inverse)(samples)[0]
        samples = samples * jnp.sqrt(jnp.diag(self.data_cov)) + self.data_mean
        return samples

    def _log_prob_single(self, x: Float[Array, " n_dim"]) -> FloatScalar:
        scale = jnp.sqrt(jnp.diag(self.data_cov))
        y, log_det = self.__call__((x - self.data_mean) / scale)
        log_det = log_det + self.base_dist.log_prob(y)
        return log_det - jnp.sum(jnp.log(scale))

    def log_prob(
        self, x: Float[Array, " n_dim"] | Float[Array, "n_sample n_dim"]
    ) -> FloatScalar | Float[Array, " n_sample"]:
        """Evaluate the normalized log density in data space."""
        if x.ndim == 2:
            return jax.vmap(self._log_prob_single)(x)
        return self._log_prob_single(x)

    def print_parameters(self):
        logger.debug("RealNVP parameters:")
        logger.debug(f"  - Data mean: {self.data_mean}")
        logger.debug(f"  - Data covariance: {self.data_cov}")

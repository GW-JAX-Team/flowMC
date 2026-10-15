from abc import abstractmethod
from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from flowMC.typing import FloatLike, FloatScalar


class Bijection(eqx.Module):
    """Base class for bijective transformations.

    Subclasses must implement ``forward`` and ``inverse``.
    The default ``__call__`` delegates to ``forward``.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        """Apply the forward transformation.

        Args:
            x (Float[Array, "n_dim"]): Input array.
            condition (Float[Array, "n_condition"]): Conditioning variables.

        Returns:
            tuple[Float[Array, "n_dim"], Float[Array, "n_dim"]]: Transformed output
                and per-dimension log-det Jacobian.
        """
        return self.forward(x, condition)

    @abstractmethod
    def forward(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        """Transform from input space to output space.

        Args:
            x (Float[Array, "n_dim"]): Input array.
            condition (Float[Array, "n_condition"]): Conditioning variables.

        Returns:
            tuple[Float[Array, "n_dim"], Float[Array, "n_dim"]]: Transformed output
                and per-dimension log-det Jacobian.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        """Transform from output space back to input space.

        Args:
            x (Float[Array, "n_dim"]): Array in the output (transformed) space.
            condition (Float[Array, "n_condition"]): Conditioning variables.

        Returns:
            tuple[Float[Array, "n_dim"], Float[Array, "n_dim"]]: Inverse output
                and per-dimension log-det Jacobian.
        """
        raise NotImplementedError


class Distribution(eqx.Module):
    """Base class for probability distributions.

    Subclasses must implement ``log_prob`` and ``sample``.
    The default ``__call__`` delegates to ``log_prob``.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(self, x: Array, key: Optional[Key[Array, ""]] = None) -> Array:
        """Evaluate the log-probability of ``x``.

        Args:
            x (Array): Input sample.
            key (Key, optional): Unused; reserved for subclass compatibility.

        Returns:
            Array: Log-probability of ``x``.
        """
        return self.log_prob(x)

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, rng_key: Key[Array, ""], n_samples: int
    ) -> Float[Array, "n_samples n_features"]:
        raise NotImplementedError


class MLP(eqx.Module):
    r"""Multilayer perceptron.

    Args:
        shape (List[int]): Shape of the MLP. The first element is the input dimension,
            the last element is the output dimension.
        key (Key): Random key.

    Attributes:
        layers (List): List of layers.
        activation (Callable): Activation function.
        use_bias (bool): Whether to use bias.
    """

    layers: list

    def __init__(
        self,
        shape: list[int],
        key: Key[Array, ""],
        scale: float = 1e-4,
        activation: Callable = jax.nn.relu,
        use_bias: bool = True,
    ):
        self.layers = []
        for i in range(len(shape) - 2):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            layer = eqx.nn.Linear(
                shape[i], shape[i + 1], key=subkey1, use_bias=use_bias
            )
            weight = jax.random.normal(subkey2, (shape[i + 1], shape[i])) * jnp.sqrt(
                scale / shape[i]
            )
            layer = eqx.tree_at(lambda layer: layer.weight, layer, weight)
            self.layers.append(layer)
            self.layers.append(activation)
        key, subkey = jax.random.split(key)
        self.layers.append(
            eqx.nn.Linear(shape[-2], shape[-1], key=subkey, use_bias=use_bias)
        )

    def __call__(self, x: Float[Array, " n_in"]) -> Float[Array, " n_out"]:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def n_input(self) -> int:
        return self.layers[0].in_features

    @property
    def n_output(self) -> int:
        return self.layers[-1].out_features

    @property
    def dtype(self) -> jnp.dtype:
        return self.layers[0].weight.dtype


class MaskedCouplingLayer(Bijection):
    r"""Masked coupling layer.

    f(x) = (1-m)*b(x;c(m*x;z)) + m*x
    where b is the inner bijector, m is the mask, and c is the conditioner.

    Args:
        bijector (Bijection): inner bijector in the masked coupling layer.
        mask (Array): Mask. 0 for the input variables that are transformed,
            1 for the input variables that are not transformed.
    """

    _mask: Float[Array, " n_dim"]
    bijector: Bijection

    @property
    def mask(self) -> Float[Array, " n_dim"]:
        return jax.lax.stop_gradient(self._mask)

    def __init__(self, bijector: Bijection, mask: Float[Array, " n_dim"]):
        self.bijector = bijector
        self._mask = mask

    def forward(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], FloatScalar]:
        y, log_det = self.bijector(x, x * self.mask)
        y = (1 - self.mask) * y + self.mask * x
        log_det = ((1 - self.mask) * log_det).sum()
        return y, log_det

    def inverse(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], FloatScalar]:
        y, log_det = self.bijector.inverse(x, x * self.mask)
        y = (1 - self.mask) * y + self.mask * x
        log_det = ((1 - self.mask) * log_det).sum()
        return y, log_det


class MLPAffine(Bijection):
    scale_MLP: MLP
    shift_MLP: MLP
    dt: float = 1

    def __init__(self, scale_MLP: MLP, shift_MLP: MLP, dt: float = 1):
        self.scale_MLP = scale_MLP
        self.shift_MLP = shift_MLP
        self.dt = dt

    def __call__(
        self, x: Float[Array, " n_dim"], condition: Float[Array, " n_cond"]
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        return self.forward(x, condition)

    def forward(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        # Returns per-dimension log-det so MaskedCouplingLayer can mask and sum.
        scale = jnp.tanh(self.scale_MLP(condition)) * self.dt
        shift = self.shift_MLP(condition) * self.dt
        log_det = scale
        y = (x + shift) * jnp.exp(scale)
        return y, log_det

    def inverse(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        scale = jnp.tanh(self.scale_MLP(condition)) * self.dt
        shift = self.shift_MLP(condition) * self.dt
        log_det = -scale
        y = x * jnp.exp(-scale) - shift
        return y, log_det


class ScalarAffine(Bijection):
    scale: Array
    shift: Array

    def __init__(self, scale: FloatLike, shift: FloatLike):
        self.scale = jnp.array(scale)
        self.shift = jnp.array(shift)

    def __call__(
        self, x: Float[Array, " n_dim"], condition: Float[Array, " n_cond"]
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        return self.forward(x, condition)

    def forward(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        y = (x + self.shift) * jnp.exp(self.scale)
        log_det = jnp.broadcast_to(self.scale, x.shape)
        return y, log_det

    def inverse(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        y = x * jnp.exp(-self.scale) - self.shift
        log_det = jnp.broadcast_to(-self.scale, x.shape)
        return y, log_det


class Gaussian(Distribution):
    r"""Multivariate Gaussian distribution.

    Args:
        mean (Array): Mean.
        cov (Array): Covariance matrix.
        learnable (bool):
            Whether the mean and covariance matrix are learnable parameters.

    Attributes:
        mean (Array): Mean.
        cov (Array): Covariance matrix.
    """

    _mean: Float[Array, " n_dim"]
    _cov: Float[Array, "n_dim n_dim"]
    learnable: bool = False

    @property
    def mean(self) -> Float[Array, " n_dim"]:
        if self.learnable:
            return self._mean
        else:
            return jax.lax.stop_gradient(self._mean)

    @property
    def cov(self) -> Float[Array, "n_dim n_dim"]:
        if self.learnable:
            return self._cov
        else:
            return jax.lax.stop_gradient(self._cov)

    def __init__(
        self,
        mean: Float[Array, " n_dim"],
        cov: Float[Array, "n_dim n_dim"],
        learnable: bool = False,
    ):
        self._mean = mean
        self._cov = cov
        self.learnable = learnable

    def log_prob(self, x: Float[Array, " n_dim"]) -> FloatScalar:
        return jnp.asarray(
            jax.scipy.stats.multivariate_normal.logpdf(x, self.mean, self.cov)
        )

    def sample(
        self, rng_key: Key[Array, ""], n_samples: int
    ) -> Float[Array, "n_samples n_features"]:
        return jax.random.multivariate_normal(
            rng_key, self.mean, self.cov, (n_samples,)
        )


class Composable(Distribution):
    distributions: list[Distribution]
    partitions: dict[str, tuple[int, int]]

    def __init__(self, distributions: list[Distribution], partitions: dict):
        self.distributions = distributions
        self.partitions = partitions

    def log_prob(self, x: Float[Array, " n_dim"]) -> FloatScalar:
        log_prob: FloatScalar = jnp.zeros(())
        for dist, (_, ranges) in zip(
            self.distributions, self.partitions.items(), strict=True
        ):
            log_prob = log_prob + dist.log_prob(x[ranges[0] : ranges[1]])
        return log_prob

    def sample(
        self, rng_key: Key[Array, ""], n_samples: int
    ) -> Float[Array, "n_samples n_features"]:
        samples = []
        for dist in self.distributions:
            rng_key, sub_key = jax.random.split(rng_key)
            samples.append(dist.sample(sub_key, n_samples=n_samples))
        return jnp.concatenate(samples, axis=-1)

from flowMC.resource.base import Resource
from typing import TypeVar
import numpy as np
from jaxtyping import Array, Float
import jax.numpy as jnp
import jax
import logging

logger = logging.getLogger(__name__)

TBuffer = TypeVar("TBuffer", bound="Buffer")


class Buffer(Resource):
    """A fixed-size ring-style buffer for storing MCMC chain positions, log-probs, or accept rates.

    Attributes:
        name (str): Name of the buffer.
        data (Float[Array, "..."]): Underlying data array, initialised to ``-inf``.
        cursor (int): Current write offset along ``cursor_dim``.
        cursor_dim (int): Dimension along which ``update_buffer`` writes.
    """

    name: str
    data: Float[Array, "..."]
    cursor: int = 0
    cursor_dim: int = 0

    def __repr__(self):
        return "Buffer " + self.name + " with shape " + str(self.data.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    def __init__(self, name: str, shape: tuple[int, ...], cursor_dim: int = 0) -> None:
        """
        Args:
            name (str): Name of the buffer.
            shape (tuple[int, ...]): Shape of the underlying array.
            cursor_dim (int): The dimension along which writes are appended.
                Defaults to 0.
        """
        self.cursor_dim = cursor_dim
        self.name = name
        self.data = jnp.zeros(shape) - jnp.inf

    def __call__(self) -> Float[Array, "..."]:
        """Return the underlying data array."""
        return self.data

    def update_buffer(self, updates: Array, start: int = 0) -> None:
        """Write new data into the buffer at the specified offset.

        Args:
            updates (Array): Data to write. The size of the first dimension of
                ``updates`` determines how many entries are written along
                ``cursor_dim``.
            start (int): Starting index along ``cursor_dim``. Defaults to 0.
        """
        self.data = jax.lax.dynamic_update_slice_in_dim(
            self.data, updates, start, self.cursor_dim
        )

    def print_parameters(self):
        logger.debug(
            f"Buffer: {self.name} with shape {self.data.shape} and cursor"
            f" {self.cursor} at dimension {self.cursor_dim}"
        )

    def get_distribution(self, n_bins: int = 100) -> tuple:
        """Compute a histogram of the flattened buffer data.

        Args:
            n_bins (int): Number of histogram bins. Defaults to 100.

        Returns:
            tuple: ``(counts, bin_edges)`` as returned by ``numpy.histogram``.
        """
        return np.histogram(self.data.flatten(), bins=n_bins)

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            data=self.data,
        )

    def load_resource(self: TBuffer, path: str) -> TBuffer:
        data = np.load(path)
        buffer: Float[Array, "..."] = data["data"]
        result = Buffer(data["name"], buffer.shape)
        result.data = buffer
        return result  # type: ignore

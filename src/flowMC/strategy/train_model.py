from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
import logging
from flowMC.resource.buffers import Buffer
from flowMC.resource.model.nf_model.base import NFModel
from flowMC.resource.optimizer import Optimizer
from flowMC.utils.logging import enable_verbose_logging
from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class TrainModel(Strategy):
    """Strategy that trains a normalizing flow model on buffered chain positions.

    Each call draws the most-recent ``history_window`` steps from the position
    buffer, sub-samples up to ``n_max_examples`` points, and runs ``n_epochs``
    of gradient descent via the associated ``flowMC.resource.optimizer.Optimizer``.

    Attributes:
        model_resource (str): Resource key for the ``flowMC.resource.model.nf_model.base.NFModel``.
        data_resource (str): Resource key for the position ``flowMC.resource.buffers.Buffer``.
        optimizer_resource (str): Resource key for the ``flowMC.resource.optimizer.Optimizer``.
        loss_buffer_name (str): Resource key for an optional
            ``flowMC.resource.buffers.Buffer`` to record loss values.
            Empty string disables loss recording.
        n_epochs (int): Number of training epochs per call.
        batch_size (int): Mini-batch size.
        n_max_examples (int): Maximum number of training examples sampled per call.
        history_window (int): Number of most-recent steps from the buffer used
            as training data.
        verbose (bool): Whether to print a training progress bar.
    """

    model_resource: str
    data_resource: str
    optimizer_resource: str
    loss_buffer_name: str
    n_epochs: int
    batch_size: int
    n_max_examples: int
    history_window: int
    verbose: bool

    def __repr__(self):
        return "Train " + self.model_resource

    def __init__(
        self,
        model_resource: str,
        data_resource: str,
        optimizer_resource: str,
        loss_buffer_name: str = "",
        n_epochs: int = 100,
        batch_size: int = 64,
        n_max_examples: int = 10000,
        history_window: int = 100,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            model_resource (str): Resource key for the normalizing flow model.
            data_resource (str): Resource key for the position buffer used as
                training data.
            optimizer_resource (str): Resource key for the optimizer.
            loss_buffer_name (str): Resource key for an optional
                ``flowMC.resource.buffers.Buffer`` to record loss values.
                Pass ``""`` to disable. Defaults to ``""``.
            n_epochs (int): Number of training epochs per call. Defaults to 100.
            batch_size (int): Mini-batch size. Defaults to 64.
            n_max_examples (int): Cap on training examples drawn per call.
                Defaults to 10000.
            history_window (int): Use only the last ``history_window`` thinned
                steps from the buffer. Defaults to 100.
            verbose (bool): Print training progress bar. Defaults to False.
        """
        self.model_resource = model_resource
        self.data_resource = data_resource
        self.optimizer_resource = optimizer_resource
        self.loss_buffer_name = loss_buffer_name

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_max_examples = n_max_examples
        self.history_window = history_window
        self.verbose = verbose
        if verbose:
            enable_verbose_logging(logger)

    def __call__(
        self,
        rng_key: Key,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        Key,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        model = resources[self.model_resource]
        assert isinstance(model, NFModel), "Target resource must be a NFModel"
        data_resource = resources[self.data_resource]
        assert isinstance(data_resource, Buffer), "Data resource must be a buffer"
        optimizer = resources[self.optimizer_resource]
        assert isinstance(optimizer, Optimizer), (
            "Optimizer resource must be an optimizer"
        )
        n_chains = data_resource.data.shape[0]
        n_dims = data_resource.data.shape[-1]
        training_data = data_resource.data[
            jnp.isfinite(data_resource.data).all(axis=-1)
        ].reshape(n_chains, -1, n_dims)
        training_data = training_data[:, -self.history_window :].reshape(-1, n_dims)
        rng_key, subkey = jax.random.split(rng_key)
        training_data = training_data[
            jax.random.choice(
                subkey,
                jnp.arange(training_data.shape[0]),
                shape=(self.n_max_examples,),
                replace=True,
            )
        ]
        rng_key, subkey = jax.random.split(rng_key)

        logger.info(f"Training {self.model_resource} model")
        logger.debug(f"  Training data shape: {training_data.shape}")
        logger.debug(f"  Number of epochs: {self.n_epochs}")
        logger.debug(f"  Batch size: {self.batch_size}")

        (rng_key, model, optim_state, loss_values) = model.train(
            rng=subkey,
            data=training_data,
            optim=optimizer.optim,
            state=optimizer.optim_state,
            num_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        if self.loss_buffer_name != "":
            loss_buffer = resources[self.loss_buffer_name]
            assert isinstance(loss_buffer, Buffer), (
                "Loss buffer resource must be a buffer"
            )
            loss_buffer.update_buffer(loss_values, start=loss_buffer.cursor)
            loss_buffer.cursor += len(loss_values)
            resources[self.loss_buffer_name] = loss_buffer

        optimizer.optim_state = optim_state
        resources[self.model_resource] = model
        resources[self.optimizer_resource] = optimizer

        logger.info(
            f"Completed training {self.model_resource} - Final loss: {loss_values[-1]:.6f}"
        )
        logger.debug(
            f"  Loss values: min={jnp.min(loss_values):.6f}, max={jnp.max(loss_values):.6f}"
        )

        return rng_key, resources, initial_position

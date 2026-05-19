from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key
import equinox as eqx

from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.resource.logPDF import LogPDF, TemperedPDF
from flowMC.resource.kernel.HMC import HMC
from flowMC.resource.kernel.NF_proposal import NFProposal
from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.strategy.lambda_function import Lambda
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.update_state import UpdateState
from flowMC.strategy.parallel_tempering import ParallelTempering
from flowMC.strategy.adapt_step_size import AdaptStepSize, AdaptStepSizePerDim
from flowMC.strategy.check_early_stop import CheckEarlyStop

from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle
import logging

logger = logging.getLogger(__name__)


class RQSpline_HMC_PT_Bundle(ResourceStrategyBundle):
    """A bundle that uses a Rational Quadratic Spline as a normalizing flow model and
    Hamiltonian Monte Carlo as a local sampler.

    The main difference between this and the RQSpline_HMC_Bundle is that this bundle
    uses an additional parallel tempering step to sample from the target distribution.

    In this bundle, the sampler requires an additional logpdf function that is the prior
    distribution. If the log prior is not provided, it will be set to 0.
    """

    def __repr__(self):
        return "RQSpline HMC PT Bundle"

    def __init__(
        self,
        # --- Required ---
        rng_key: Key,
        n_chains: int,
        n_dims: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_local_steps: int,
        n_global_steps: int,
        n_training_loops: int,
        n_production_loops: int,
        n_epochs: int,
        # --- Local sampler ---
        hmc_step_size: float = 0.1,
        hmc_n_leapfrog: int = 10,
        condition_matrix: Float | Float[Array, " n_dim"] = 1,
        adapt_step_size: bool = True,
        adapt_step_size_per_dim: bool = True,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
        # --- Normalizing flow model ---
        rq_spline_hidden_units: list[int] = [32, 32],
        rq_spline_n_bins: int = 8,
        rq_spline_n_layers: int = 4,
        n_NFproposal_batch_size: int = 10000,
        # --- Training ---
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        history_window: int = 100,
        # --- Sampling execution ---
        chain_batch_size: int = 0,
        local_thinning: int = 1,
        global_thinning: int = 1,
        # --- Parallel tempering ---
        n_temperatures: int = 5,
        max_temperature: float = 5.0,
        n_tempered_steps: int = -1,
        logprior: Callable[[Float[Array, " n_dim"], dict], Float] = lambda x, _: 0.0,
        # --- Early stopping ---
        early_stopping: bool = False,
        early_stopping_tolerance: float = 0.05,
        early_stopping_patience: int = 3,
        early_stopping_min_acceptance: float = 0.1,
        # --- Misc ---
        verbose: bool = False,
    ) -> None:
        """Build all resources and strategies for an RQSpline + HMC + PT sampling run.

        Args:
            rng_key (Key): JAX PRNGKey used to initialise the normalizing flow.
            n_chains (int): Number of parallel MCMC chains.
            n_dims (int): Dimensionality of the target distribution.
            logpdf (Callable): Log-likelihood ``f(x, data) -> Float``.
            n_local_steps (int): HMC steps per training/production loop iteration.
            n_global_steps (int): NF-proposal steps per training/production loop iteration.
            n_training_loops (int): Number of train-then-sample iterations (warmup).
            n_production_loops (int): Number of production sampling iterations.
            n_epochs (int): NF training epochs per training loop.

            hmc_step_size (float): Initial leapfrog step size. Defaults to 0.1.
            hmc_n_leapfrog (int): Number of leapfrog steps per HMC proposal.
                Defaults to 10.
            condition_matrix (float | Float[Array, "n_dim"]): Diagonal mass-matrix
                elements; scalar (broadcast) or per-dimension array. Defaults to 1.
            adapt_step_size (bool): Adapt the HMC step size during training.
                Defaults to True.
            adapt_step_size_per_dim (bool): Also tune per-dimension effective
                step sizes via the mass matrix using empirical posterior variance.
                Runs after adapt_step_size. Defaults to True.
            periodic (dict[int, tuple[float, float]] | None): Periodic boundary
                conditions as ``{dim_index: (lower, upper)}``. Defaults to None.

            rq_spline_hidden_units (list[int]): Hidden units per conditioner MLP layer.
                Defaults to ``[32, 32]``.
            rq_spline_n_bins (int): Number of RQ-spline bins. Defaults to 8.
            rq_spline_n_layers (int): Number of masked coupling layers. Defaults to 4.
            n_NFproposal_batch_size (int): NF log-prob evaluation batch size.
                Defaults to 10000.

            learning_rate (float): Adam learning rate for NF training. Defaults to 1e-3.
            batch_size (int): Mini-batch size for NF training. Defaults to 10000.
            n_max_examples (int): Maximum training examples per call. Defaults to 10000.
            history_window (int): Use only the last ``history_window`` stored steps as
                training data. Defaults to 100.

            chain_batch_size (int): Process chains in sub-batches of this size to
                reduce peak memory. 0 disables batching. Defaults to 0.
            local_thinning (int): Store every ``local_thinning``-th local step.
                Defaults to 1.
            global_thinning (int): Store every ``global_thinning``-th global step.
                Defaults to 1.

            n_temperatures (int): Number of parallel tempering temperature levels.
                Defaults to 5.
            max_temperature (float): Maximum temperature (lowest inverse temperature).
                Defaults to 5.0.
            n_tempered_steps (int): Local kernel steps per temperature per PT exchange.
                Pass -1 to use ``n_local_steps``. Defaults to -1.
            logprior (Callable): Log-prior ``f(x, data) -> Float`` used to construct
                the tempered density. Defaults to a constant-zero prior.

            early_stopping (bool): Enable early stopping based on global acceptance
                rate stability. Defaults to False.
            early_stopping_tolerance (float): Relative change threshold for early
                stopping. Defaults to 0.05.
            early_stopping_patience (int): Consecutive stable loops required before
                stopping. Defaults to 3.
            early_stopping_min_acceptance (float): Minimum global acceptance rate
                that also triggers early stopping. Defaults to 0.1.

            verbose (bool): Enable progress bars and debug logging. Defaults to False.
        """
        if local_thinning > n_local_steps:
            raise ValueError(
                f"local_thinning ({local_thinning}) must not exceed n_local_steps "
                f"({n_local_steps}). This would result in zero samples being stored. "
                f"Either increase n_local_steps or decrease local_thinning."
            )
        if global_thinning > n_global_steps:
            raise ValueError(
                f"global_thinning ({global_thinning}) must not exceed n_global_steps "
                f"({n_global_steps}). This would result in zero samples being stored. "
                f"Either increase n_global_steps or decrease global_thinning."
            )

        n_training_steps = (
            n_local_steps // local_thinning * n_training_loops
            + n_global_steps // global_thinning * n_training_loops
        )
        n_production_steps = (
            n_local_steps // local_thinning * n_production_loops
            + n_global_steps // global_thinning * n_production_loops
        )
        n_total_epochs = n_training_loops * n_epochs

        positions_training = Buffer(
            "positions_training", (n_chains, n_training_steps, n_dims), 1
        )
        log_prob_training = Buffer("log_prob_training", (n_chains, n_training_steps), 1)
        local_accs_training = Buffer(
            "local_accs_training", (n_chains, n_training_steps), 1
        )
        global_accs_training = Buffer(
            "global_accs_training", (n_chains, n_training_steps), 1
        )
        loss_buffer = Buffer("loss_buffer", (n_total_epochs,), 0)

        position_production = Buffer(
            "positions_production", (n_chains, n_production_steps, n_dims), 1
        )
        log_prob_production = Buffer(
            "log_prob_production", (n_chains, n_production_steps), 1
        )
        local_accs_production = Buffer(
            "local_accs_production", (n_chains, n_production_steps), 1
        )
        global_accs_production = Buffer(
            "global_accs_production", (n_chains, n_production_steps), 1
        )

        # Convert scalar condition matrix to 1D array if needed
        if isinstance(condition_matrix, (int, float)):
            condition_matrix = jnp.full(n_dims, condition_matrix)
        # Create periodic mask and bounds arrays for HMC
        if periodic is None:
            periodic = {}
        periodic_mask = jnp.zeros(n_dims, dtype=bool)
        periodic_bounds = jnp.zeros((n_dims, 2))
        for dim_idx, (lower, upper) in periodic.items():
            if not (0 <= dim_idx < n_dims):
                raise ValueError(
                    f"periodic dim_idx={dim_idx} is out of range [0, {n_dims})"
                )
            if lower >= upper:
                raise ValueError(
                    f"periodic bounds for dim {dim_idx} must satisfy lower < upper,"
                    f" got ({lower}, {upper})"
                )
            periodic_mask = periodic_mask.at[dim_idx].set(True)
            periodic_bounds = periodic_bounds.at[dim_idx].set(jnp.array([lower, upper]))
        local_sampler = HMC(
            step_size=hmc_step_size,
            n_leapfrog=hmc_n_leapfrog,
            condition_matrix=condition_matrix,
            periodic_mask=periodic_mask,
            periodic_bounds=periodic_bounds,
        )
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            n_dims, rq_spline_n_layers, rq_spline_hidden_units, rq_spline_n_bins, subkey
        )
        global_sampler = NFProposal(
            model, n_NFproposal_batch_size=n_NFproposal_batch_size
        )
        optimizer = Optimizer(model=model, learning_rate=learning_rate)
        logpdf = LogPDF(logpdf, n_dims=n_dims)

        # Here are the resources for the parallel tempering
        tempered_logpdf = TemperedPDF(
            logpdf,
            logprior,
            n_dims=n_dims,
            n_temps=n_temperatures,
        )
        tempered_positions = Buffer(
            "tempered_positions", (n_chains, n_temperatures - 1, n_dims), 2
        )

        temperatures = Buffer("temperature", (n_temperatures,), 0)
        temperatures.update_buffer(
            jax.numpy.linspace(1.0, max_temperature, n_temperatures)
        )

        sampler_state = State(
            {
                "target_positions": "positions_training",
                "target_log_prob": "log_prob_training",
                "target_local_accs": "local_accs_training",
                "target_global_accs": "global_accs_training",
                "training": True,
                "early_stopped": False,
            },
            name="sampler_state",
        )

        self.resources = {
            "logpdf": logpdf,
            "positions_training": positions_training,
            "log_prob_training": log_prob_training,
            "local_accs_training": local_accs_training,
            "global_accs_training": global_accs_training,
            "loss_buffer": loss_buffer,
            "positions_production": position_production,
            "log_prob_production": log_prob_production,
            "local_accs_production": local_accs_production,
            "global_accs_production": global_accs_production,
            "local_sampler": local_sampler,
            "global_sampler": global_sampler,
            "model": model,
            "optimizer": optimizer,
            "sampler_state": sampler_state,
            "tempered_logpdf": tempered_logpdf,
            "tempered_positions": tempered_positions,
            "temperatures": temperatures,
        }

        local_stepper = TakeSerialSteps(
            "logpdf",
            "local_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_local_accs"],
            n_local_steps,
            thinning=local_thinning,
            chain_batch_size=chain_batch_size,
            verbose=verbose,
        )

        global_stepper = TakeGroupSteps(
            "logpdf",
            "global_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_global_accs"],
            n_global_steps,
            thinning=global_thinning,
            chain_batch_size=chain_batch_size,
            verbose=verbose,
        )

        model_trainer = TrainModel(
            "model",
            "positions_training",
            "optimizer",
            loss_buffer_name="loss_buffer",
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            history_window=history_window,
            verbose=verbose,
        )

        update_state = UpdateState(
            "sampler_state",
            [
                "target_positions",
                "target_log_prob",
                "target_local_accs",
                "target_global_accs",
                "training",
            ],
            [
                "positions_production",
                "log_prob_production",
                "local_accs_production",
                "global_accs_production",
                False,
            ],
        )

        def reset_steppers(
            rng_key: Key,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            Key,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Reset the steppers to the initial position."""
            local_stepper.set_current_position(0)
            global_stepper.set_current_position(0)
            return rng_key, resources, initial_position

        reset_steppers_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: reset_steppers(
                rng_key, resources, initial_position, data
            )
        )

        update_global_step = Lambda(
            lambda rng_key, resources, initial_position, data: (
                global_stepper.set_current_position(local_stepper.current_position)
            )
        )
        update_local_step = Lambda(
            lambda rng_key, resources, initial_position, data: (
                local_stepper.set_current_position(global_stepper.current_position)
            )
        )

        def update_model(
            rng_key: Key,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            Key,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Update the model."""
            model = resources["model"]
            resources["global_sampler"] = eqx.tree_at(
                lambda x: x.model,
                resources["global_sampler"],
                model,
            )
            return rng_key, resources, initial_position

        update_model_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: update_model(
                rng_key, resources, initial_position, data
            )
        )

        if n_tempered_steps <= 0:
            logger.warning(
                "n_tempered_steps value is not valid. Setting to n_local_steps"
            )
            n_tempered_steps = n_local_steps

        parallel_tempering_strat = ParallelTempering(
            n_steps=n_tempered_steps,
            tempered_logpdf_name="tempered_logpdf",
            kernel_name="local_sampler",
            tempered_buffer_names=["tempered_positions", "temperatures"],
            state_name="sampler_state",
            verbose=verbose,
        )

        def initialize_tempered_positions(
            rng_key: Key,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            Key,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Initialize the tempered positions."""
            tempered_positions.update_buffer(
                jnp.repeat(initial_position[:, None], n_temperatures - 1, axis=1)
            )
            return rng_key, resources, initial_position

        initialize_tempered_positions_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: (
                initialize_tempered_positions(
                    rng_key, resources, initial_position, data
                )
            )
        )

        # Adapt local sampler step size during training
        adapt_local_sampler = AdaptStepSize(
            kernel_name="local_sampler",
            state_name="sampler_state",
            acceptance_buffer_key="target_local_accs",
            target_acceptance_rate=0.65,
            acceptance_window=n_local_steps,
            n_loops_skip=3,
            verbose=verbose,
        )

        adapt_local_sampler_per_dim = AdaptStepSizePerDim(
            kernel_name="local_sampler",
            state_name="sampler_state",
            positions_buffer_key="target_positions",
            verbose=verbose,
        )

        check_early_stop = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=early_stopping_tolerance,
            acceptance_window=n_global_steps * 3 // global_thinning,
            n_loops_skip=3,
            patience=early_stopping_patience,
            min_acceptance_rate=early_stopping_min_acceptance,
            verbose=verbose,
        )

        self.strategies = {
            "local_stepper": local_stepper,
            "global_stepper": global_stepper,
            "model_trainer": model_trainer,
            "update_state": update_state,
            "update_global_step": update_global_step,
            "update_local_step": update_local_step,
            "reset_steppers": reset_steppers_lambda,
            "update_model": update_model_lambda,
            "parallel_tempering": parallel_tempering_strat,
            "initialize_tempered_positions": initialize_tempered_positions_lambda,
            "adapt_local_sampler": adapt_local_sampler,
            "adapt_local_sampler_per_dim": adapt_local_sampler_per_dim,
            "check_early_stop": check_early_stop,
        }

        training_phase = [
            "parallel_tempering",
            "local_stepper",
            *(("adapt_local_sampler",) if adapt_step_size else []),
            *(("adapt_local_sampler_per_dim",) if adapt_step_size_per_dim else []),
            "update_global_step",
            "model_trainer",
            "update_model",
            "global_stepper",
            "update_local_step",
            *(("check_early_stop",) if early_stopping else []),
        ]
        production_phase = [
            "parallel_tempering",
            "local_stepper",
            "update_global_step",
            "global_stepper",
            "update_local_step",
        ]
        strategy_order = ["initialize_tempered_positions"]
        for _ in range(n_training_loops):
            strategy_order.extend(training_phase)

        strategy_order.append("reset_steppers")
        strategy_order.append("update_state")
        for _ in range(n_production_loops):
            strategy_order.extend(production_phase)

        self.strategy_order = strategy_order

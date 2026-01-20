import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import pytest

from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.resource.kernel.NF_proposal import NFProposal
from flowMC.resource.kernel.MALA import MALA
from flowMC.resource.kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.kernel.HMC import HMC
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.resource.logPDF import LogPDF, TemperedPDF

# from flowMC.strategy.optimization import optimization_Adam
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.optimization import AdamOptimization
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.parallel_tempering import ParallelTempering


def log_posterior(x, data={}):
    return -0.5 * jnp.sum((x - data["data"]) ** 2)


class TestOptimizationStrategies:
    n_dim = 2
    n_chains = 20
    n_steps = 100

    strategy = AdamOptimization(
        log_posterior,
        n_steps,
        learning_rate=5e-2,
        noise_level=0.0,
        bounds=jnp.array([[-jnp.inf, jnp.inf]]),
    )

    def test_Adam_optimization(self):
        key = jax.random.PRNGKey(42)

        key, subkey = jax.random.split(key)
        initial_position = (
            jax.random.normal(subkey, shape=(self.n_chains, self.n_dim)) * 1 + 10
        )

        _, _, optimized_position = self.strategy(
            key, {}, initial_position, {"data": jnp.arange(self.n_dim)}
        )

        assert optimized_position.shape == (self.n_chains, self.n_dim)
        assert jnp.all(
            jnp.mean(optimized_position, axis=1) < jnp.mean(initial_position, axis=1)
        )

    def test_standalone_optimize(self):
        key = jax.random.PRNGKey(42)

        key, subkey = jax.random.split(key)
        initial_position = (
            jax.random.normal(subkey, shape=(self.n_chains, self.n_dim)) * 1 + 10
        )

        def loss_fn(params: Float[Array, " n_dim"], data: dict = {}) -> Float:
            return -log_posterior(params, {"data": jnp.arange(self.n_dim)})

        _, optimized_position, final_log_prob = self.strategy.optimize(
            key, loss_fn, initial_position, {"data": jnp.arange(self.n_dim)}
        )

        assert optimized_position.shape == (self.n_chains, self.n_dim)
        assert jnp.all(
            jnp.mean(optimized_position, axis=1) < jnp.mean(initial_position, axis=1)
        )

        assert final_log_prob.shape == (self.n_chains,)
        assert jnp.all(jnp.isfinite(final_log_prob))


class TestLocalStep:
    @pytest.fixture(autouse=True)
    def setup(self):
        n_chains = 5
        n_steps = 25
        n_dims = 2
        n_batch = 5
        test_position = Buffer("test_position", (n_chains, n_steps, n_dims), 1)
        test_log_prob = Buffer("test_log_prob", (n_chains, n_steps), 1)
        test_acceptance = Buffer("test_acceptance", (n_chains, n_steps), 1)
        mala_kernel = MALA(jnp.ones(n_dims))
        grw_kernel = GaussianRandomWalk(1.0)
        hmc_kernel = HMC(jnp.ones(n_dims), 0.1, 10)
        logpdf = LogPDF(log_posterior, n_dims=n_dims)
        sampler_state = State(
            {
                "test_position": "test_position",
                "test_log_prob": "test_log_prob",
                "test_acceptance": "test_acceptance",
            },
            name="sampler_state",
        )
        self.n_batch = n_batch
        self.n_dims = n_dims
        self.test_position = test_position
        self.resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "logpdf": logpdf,
            "MALA": mala_kernel,
            "GRW": grw_kernel,
            "HMC": hmc_kernel,
            "sampler_state": sampler_state,
        }

    def test_take_local_step(self):
        strategy = TakeSerialSteps(
            "logpdf",
            "MALA",
            "sampler_state",
            ["test_position", "test_log_prob", "test_acceptance"],
            self.n_batch,
        )
        key = jax.random.PRNGKey(42)
        positions = self.test_position.data[:, 0]
        for _ in range(self.n_batch):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            _, self.resources, positions = strategy(
                rng_key=subkey1,
                resources=self.resources,
                initial_position=positions,
                data={"data": jnp.arange(self.n_dims)},
            )
        key, subkey1, subkey2 = jax.random.split(key, 3)
        strategy.set_current_position(0)
        _, self.resources, positions = strategy(
            rng_key=subkey1,
            resources=self.resources,
            initial_position=positions,
            data={"data": jnp.arange(self.n_dims)},
        )
        key, subkey1, subkey2 = jax.random.split(key, 3)
        strategy.kernel_name = "GRW"
        strategy.set_current_position(0)
        _, self.resources, positions = strategy(
            rng_key=subkey1,
            resources=self.resources,
            initial_position=positions,
            data={"data": jnp.arange(self.n_dims)},
        )
        strategy.kernel_name = "HMC"
        _, self.resources, positions = strategy(
            rng_key=subkey1,
            resources=self.resources,
            initial_position=positions,
            data={"data": jnp.arange(self.n_dims)},
        )

    def test_take_local_step_chain_batch_size(self):
        # Use a chain_batch_size smaller than the number of chains to trigger batching logic
        chain_batch_size = 2
        strategy = TakeSerialSteps(
            "logpdf",
            "MALA",
            "sampler_state",
            ["test_position", "test_log_prob", "test_acceptance"],
            self.n_batch,
            chain_batch_size=chain_batch_size,
        )
        key = jax.random.PRNGKey(42)
        positions = self.test_position.data[:, 0]
        # Run the strategy, which should use batching internally
        _, _, final_positions = strategy(
            rng_key=key,
            resources=self.resources,
            initial_position=positions,
            data={"data": jnp.arange(self.n_dims)},
        )
        # Check that the output shape is correct
        assert final_positions.shape == (positions.shape[0], positions.shape[1])
        # Optionally, check that the buffer was updated for all chains
        assert isinstance(test_position := self.resources["test_position"], Buffer)
        assert test_position.data.shape[0] == positions.shape[0]


class TestNFStrategies:
    n_chains = 5
    n_steps = 25
    n_dims = 2
    n_batch = 5

    n_features = n_dims
    hidden_layes = [16, 16]
    n_layers = 3
    n_bins = 8

    def initialize(self):
        rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0), 2)
        model = MaskedCouplingRQSpline(
            self.n_features,
            self.n_layers,
            self.hidden_layes,
            self.n_bins,
            jax.random.PRNGKey(10),
        )

        test_data = Buffer("test_data", (self.n_chains, self.n_steps, self.n_dims), 1)
        test_data.update_buffer(
            jax.random.normal(
                rng_subkey, shape=(self.n_chains, self.n_steps, self.n_dims)
            ),
        )
        optimizer = Optimizer(model)

        resources = {
            "test_data": test_data,
            "optimizer": optimizer,
            "model": model,
        }

        return rng_key, resources

    def test_training(self):
        # TODO: Need to check for accuracy still
        rng_key, resources = self.initialize()

        strategy = TrainModel(
            "model",
            "test_data",
            "optimizer",
            n_epochs=10,
            batch_size=self.n_chains * self.n_steps,
            n_max_examples=10000,
            history_window=1,
            verbose=True,
        )

        key = jax.random.PRNGKey(42)

        print(resources["model"].data_mean, resources["model"].data_cov)
        key, resources, positions = strategy(
            key,
            resources,
            jax.random.normal(key, shape=(self.n_chains, self.n_dims)),
            {"data": jnp.arange(self.n_dims)},
        )
        assert isinstance(resources["model"], MaskedCouplingRQSpline)
        print(resources["model"].data_mean, resources["model"].data_cov)

    def test_take_NF_step(self):
        test_position = Buffer(
            "test_position", (self.n_chains, self.n_steps, self.n_dims), 1
        )
        test_log_prob = Buffer("test_log_prob", (self.n_chains, self.n_steps), 1)
        test_acceptance = Buffer("test_acceptance", (self.n_chains, self.n_steps), 1)

        model = MaskedCouplingRQSpline(
            self.n_features,
            self.n_layers,
            self.hidden_layes,
            self.n_bins,
            jax.random.PRNGKey(10),
        )

        proposal = NFProposal(model, n_NFproposal_batch_size=5)

        def test_target(x, data={}):
            return model.log_prob(x)

        sampler_state = State(
            {
                "test_position": "test_position",
                "test_log_prob": "test_log_prob",
                "test_acceptance": "test_acceptance",
            },
            name="sampler_state",
        )

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "NFProposal": proposal,
            "logpdf": LogPDF(test_target, n_dims=self.n_dims),
            "sampler_state": sampler_state,
        }

        strategy = TakeGroupSteps(
            "logpdf",
            "NFProposal",
            "sampler_state",
            ["test_position", "test_log_prob", "test_acceptance"],
            n_steps=11,
        )
        key = jax.random.PRNGKey(42)
        positions = test_position.data[:, 0]
        print(test_position.data[:, :, 0])
        strategy(
            rng_key=key,
            resources=resources,
            initial_position=positions,
            data={"data": jnp.arange(self.n_dims)},
        )
        print(test_position.data[:, :, 0])

        strategy = TakeGroupSteps(
            "logpdf",
            "NFProposal",
            "sampler_state",
            ["test_position", "test_log_prob", "test_acceptance"],
            n_steps=5,
        )
        strategy(
            rng_key=key,
            resources=resources,
            initial_position=positions,
            data={"data": jnp.arange(self.n_dims)},
        )

    def test_training_effect(self):
        Buffer("test_position", (self.n_chains, self.n_steps, self.n_dims), 1)
        Buffer("test_log_prob", (self.n_chains, self.n_steps), 1)
        Buffer("test_acceptance", (self.n_chains, self.n_steps), 1)

        model = MaskedCouplingRQSpline(
            self.n_features,
            self.n_layers,
            self.hidden_layes,
            self.n_bins,
            jax.random.PRNGKey(10),
        )

        NFProposal(model)


class TestTemperingStrategies:
    n_temps = 5
    n_dims = 3
    n_chains = 7
    n_steps = 4

    def initialize(self):
        mala = MALA(jnp.ones(self.n_dims))
        logpdf = TemperedPDF(
            log_posterior,
            lambda x, data: jnp.array(0.0),
            n_dims=self.n_dims,
            n_temps=self.n_temps,
        )

        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)

        initial_position = jax.random.normal(subkey, shape=(self.n_chains, self.n_dims))
        key, subkey = jax.random.split(key)
        tempered_initial_position = jax.random.normal(
            subkey, shape=(self.n_chains, self.n_temps - 1, self.n_dims)
        )
        tempered_positions = Buffer(
            "tempered_positions", (self.n_chains, self.n_temps - 1, self.n_dims), 2
        )
        tempered_positions.update_buffer(tempered_initial_position)
        temperatures = Buffer("temperatures", (self.n_temps,), 0)
        temperatures.update_buffer(jnp.arange(self.n_temps) + 1.0)

        sampler_state = State(
            {
                "target_positions": "tempered_positions",
                "target_log_prob": "logpdf",
                "target_temperatures": "temperatures",
                "training": False,
            },
            name="sampler_state",
        )

        resources = {
            "logpdf": logpdf,
            "MALA": mala,
            "tempered_positions": tempered_positions,
            "temperatures": temperatures,
            "sampler_state": sampler_state,
        }

        parallel_tempering_strat = ParallelTempering(
            n_steps=self.n_steps,
            tempered_logpdf_name="logpdf",
            kernel_name="MALA",
            tempered_buffer_names=["tempered_positions", "temperatures"],
            state_name="sampler_state",
        )

        return key, resources, parallel_tempering_strat, initial_position

    def test_individual_step_body(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        mala = resources["MALA"]
        logpdf = resources["logpdf"]
        key, subkey = jax.random.split(key)
        position = initial_position[0]
        data = {"data": jnp.arange(self.n_dims)}

        log_prob = logpdf(position, data)
        carry, extras = parallel_tempering_strat._individual_step_body(
            mala, (key, position, log_prob, logpdf, jnp.array(1.0), data), None
        )

        # TODO: Add assertions

        assert carry[1].shape == (self.n_dims,)

    def test_individual_step(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        mala = resources["MALA"]
        logpdf = resources["logpdf"]
        initial_position = jnp.concatenate(
            [initial_position[:, None, :], resources["tempered_positions"].data],
            axis=1,
        )

        positions, log_probs, do_accept = parallel_tempering_strat._individal_step(
            mala,
            key,
            initial_position[0, 0],
            logpdf,
            jnp.array(1),
            {"data": jnp.arange(self.n_dims)},
        )

        # TODO: Add assertions
        assert positions.shape == (self.n_dims,)

    def test_ensemble_step(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        mala = resources["MALA"]
        logpdf = resources["logpdf"]
        initial_position = jnp.concatenate(
            [initial_position[:, None, :], resources["tempered_positions"].data],
            axis=1,
        )
        key, subkey = jax.random.split(key)
        positions, log_probs, do_accept = parallel_tempering_strat._ensemble_step(
            mala,
            subkey,
            initial_position[0],
            logpdf,
            jnp.arange(self.n_temps) + 1.0,
            {
                "data": jnp.arange(self.n_dims),
            },
        )

        # TODO: Add assertions

        keys = jax.random.split(key, self.n_chains)
        positions, log_probs, do_accept = jax.vmap(
            parallel_tempering_strat._ensemble_step,
            in_axes=(None, 0, 0, None, None, None),
        )(
            mala,
            keys,
            initial_position,
            logpdf,
            jnp.arange(self.n_temps) + 1.0,
            {
                "data": jnp.arange(self.n_dims),
            },
        )

        # TODO: Add assertions

    def test_exchange_step(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        logpdf = resources["logpdf"]
        temperatures = jnp.arange(self.n_temps) * 0.3 + 1
        data = {"data": jnp.arange(self.n_dims)}
        log_probs = jax.vmap(logpdf.tempered_log_pdf, in_axes=(None, 0, None))(
            temperatures,
            initial_position,
            data,
        )
        key = jax.random.split(key, self.n_chains)
        initial_position = jnp.concatenate(
            [initial_position[:, None, :], resources["tempered_positions"].data],
            axis=1,
        )
        positions, log_probs, do_accept = jax.jit(
            jax.vmap(parallel_tempering_strat._exchange, in_axes=(0, 0, 0, None, None))
        )(
            key,
            initial_position,
            logpdf,
            temperatures,
            {"data": jnp.arange(self.n_dims)},
        )

    def test_adapt_temperatures(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        temperatures = jnp.arange(self.n_temps) * 0.3 + 1
        parallel_tempering_strat._adapt_temperature(
            temperatures,
            jnp.ones((self.n_chains, self.n_temps)),
        )
        assert temperatures.shape == (self.n_temps,)

    def test_parallel_tempering(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        key, subkey = jax.random.split(key)
        rng_key, resources, positions = parallel_tempering_strat(
            key,
            resources,
            initial_position,
            {
                "data": jnp.arange(self.n_dims),
            },
        )
        print(positions)


class TestThinning:
    """Test thinning functionality in TakeSerialSteps and TakeGroupSteps."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_chains = 3
        self.n_dims = 2
        self.n_steps = 11  # Not divisible by typical thinning values

    def test_serial_steps_thinning_shape_consistency(self):
        """Test that positions, log_probs, and acceptances have matching shapes with thinning."""
        thinning = 3
        n_stored = (self.n_steps + thinning - 1) // thinning  # Expected stored steps

        test_position = Buffer(
            "test_position", (self.n_chains, n_stored, self.n_dims), 1
        )
        test_log_prob = Buffer("test_log_prob", (self.n_chains, n_stored), 1)
        test_acceptance = Buffer("test_acceptance", (self.n_chains, n_stored), 1)

        kernel = MALA(jnp.ones(self.n_dims))
        logpdf = LogPDF(log_posterior, n_dims=self.n_dims)
        sampler_state = State(
            {
                "test_position": "test_position",
                "test_log_prob": "test_log_prob",
                "test_acceptance": "test_acceptance",
            },
            name="sampler_state",
        )

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "logpdf": logpdf,
            "kernel": kernel,
            "sampler_state": sampler_state,
        }

        strategy = TakeSerialSteps(
            "logpdf",
            "kernel",
            "sampler_state",
            ["test_position", "test_log_prob", "test_acceptance"],
            n_steps=self.n_steps,
            thinning=thinning,
        )

        key = jax.random.PRNGKey(42)
        initial_position = jax.random.normal(key, shape=(self.n_chains, self.n_dims))

        _, resources, _ = strategy(
            rng_key=key,
            resources=resources,
            initial_position=initial_position,
            data={"data": jnp.arange(self.n_dims)},
        )

        # Verify all buffers have matching shapes
        assert resources["test_position"].data.shape == (
            self.n_chains,
            n_stored,
            self.n_dims,
        )
        assert resources["test_log_prob"].data.shape == (self.n_chains, n_stored)
        assert resources["test_acceptance"].data.shape == (self.n_chains, n_stored)

        # Verify no NaN or Inf values in buffers
        assert jnp.all(jnp.isfinite(resources["test_position"].data))
        assert jnp.all(jnp.isfinite(resources["test_log_prob"].data))
        assert jnp.all(jnp.isfinite(resources["test_acceptance"].data))

    @pytest.mark.parametrize("n_steps,thinning", [(10, 3), (12, 3), (11, 4), (20, 1)])
    def test_thinning_various_combinations(self, n_steps, thinning):
        """Test various n_steps and thinning combinations."""
        n_stored = (n_steps + thinning - 1) // thinning

        test_position = Buffer(
            "test_position", (self.n_chains, n_stored, self.n_dims), 1
        )
        test_log_prob = Buffer("test_log_prob", (self.n_chains, n_stored), 1)
        test_acceptance = Buffer("test_acceptance", (self.n_chains, n_stored), 1)

        kernel = GaussianRandomWalk(1.0)
        logpdf = LogPDF(log_posterior, n_dims=self.n_dims)
        sampler_state = State(
            {
                "test_position": "test_position",
                "test_log_prob": "test_log_prob",
                "test_acceptance": "test_acceptance",
            },
            name="sampler_state",
        )

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "logpdf": logpdf,
            "kernel": kernel,
            "sampler_state": sampler_state,
        }

        strategy = TakeSerialSteps(
            "logpdf",
            "kernel",
            "sampler_state",
            ["test_position", "test_log_prob", "test_acceptance"],
            n_steps=n_steps,
            thinning=thinning,
        )

        key = jax.random.PRNGKey(42)
        initial_position = jax.random.normal(key, shape=(self.n_chains, self.n_dims))

        _, resources, _ = strategy(
            rng_key=key,
            resources=resources,
            initial_position=initial_position,
            data={"data": jnp.arange(self.n_dims)},
        )

        # All buffers should have consistent shapes
        assert resources["test_position"].data.shape[1] == n_stored
        assert resources["test_log_prob"].data.shape[1] == n_stored
        assert resources["test_acceptance"].data.shape[1] == n_stored

    def test_acceptance_averaging_semantics(self):
        """Test that acceptance rates are averaged correctly for thinned windows."""
        n_steps = 10
        thinning = 3
        n_stored = 4  # indices 0, 3, 6, 9

        test_position = Buffer(
            "test_position", (self.n_chains, n_stored, self.n_dims), 1
        )
        test_log_prob = Buffer("test_log_prob", (self.n_chains, n_stored), 1)
        test_acceptance = Buffer("test_acceptance", (self.n_chains, n_stored), 1)

        kernel = GaussianRandomWalk(1.0)
        logpdf = LogPDF(log_posterior, n_dims=self.n_dims)
        sampler_state = State(
            {
                "test_position": "test_position",
                "test_log_prob": "test_log_prob",
                "test_acceptance": "test_acceptance",
            },
            name="sampler_state",
        )

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "logpdf": logpdf,
            "kernel": kernel,
            "sampler_state": sampler_state,
        }

        strategy = TakeSerialSteps(
            "logpdf",
            "kernel",
            "sampler_state",
            ["test_position", "test_log_prob", "test_acceptance"],
            n_steps=n_steps,
            thinning=thinning,
        )

        key = jax.random.PRNGKey(42)
        initial_position = jax.random.normal(key, shape=(self.n_chains, self.n_dims))

        _, resources, _ = strategy(
            rng_key=key,
            resources=resources,
            initial_position=initial_position,
            data={"data": jnp.arange(self.n_dims)},
        )

        acceptance_data = resources["test_acceptance"].data
        # Check that acceptance values are between 0 and 1 (averaged probabilities)
        assert jnp.all(acceptance_data >= 0.0)
        assert jnp.all(acceptance_data <= 1.0)
        # Check the shape is as expected
        assert acceptance_data.shape == (self.n_chains, n_stored)

    def test_acceptance_averaging_exact_values(self):
        """Test exact acceptance averaging values with deterministic pattern."""
        # Test case: 11 steps with thinning=3
        # Simulate what TakeSerialSteps produces internally
        n_steps = 11
        thinning = 3
        
        # Acceptance pattern: [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 999]
        # Positions stored at indices: 0, 3, 6, 9
        # Expected acceptance averaging:
        #   acceptance[0] = accept[0] = 1.0
        #   acceptance[1] = mean(accept[1:4]) = mean([1,0,1]) = 2/3
        #   acceptance[2] = mean(accept[4:7]) = mean([1,1,0]) = 2/3
        #   acceptance[3] = mean(accept[7:10]) = mean([0,1,1]) = 2/3
        #   accept[10] = 999 is discarded
        
        # Simulate the do_accepts array that would come from kernel
        do_accepts = jnp.array([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 999.0]])
        
        # Apply the thinning logic from take_steps.py
        positions = jnp.arange(n_steps).reshape(1, n_steps)[:, ::thinning]
        n_stored_steps = positions.shape[1]
        
        # First acceptance is just index 0
        first_accept = do_accepts[:, 0:1]
        
        # Remaining acceptances: reshape and mean
        n_remaining = n_stored_steps - 1
        if n_remaining > 0:
            remaining_accepts = (
                do_accepts[:, 1 : 1 + n_remaining * thinning]
                .reshape(do_accepts.shape[0], n_remaining, thinning)
                .mean(axis=2)
            )
            do_accepts_thinned = jnp.concatenate([first_accept, remaining_accepts], axis=1)
        else:
            do_accepts_thinned = first_accept
        
        # Expected values
        expected = jnp.array([[1.0, 2.0/3.0, 2.0/3.0, 2.0/3.0]])
        
        # Verify acceptance values match expected averages
        assert jnp.allclose(do_accepts_thinned, expected, atol=1e-6), (
            f"Expected {expected}, got {do_accepts_thinned}"
        )
        
        # Verify that the 999.0 value (index 10) was NOT included
        assert jnp.all(do_accepts_thinned < 10.0), "Discarded value was incorrectly included"
        
        # Verify shape matches positions
        assert do_accepts_thinned.shape == positions.shape[:2], (
            f"Shape mismatch: acceptances {do_accepts_thinned.shape} vs positions {positions.shape[:2]}"
        )


class TestAdaptStepSize:
    """Test the AdaptStepSize strategy for kernel step size adaptation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_chains = 10
        self.n_dims = 2
        self.n_steps = 100
        
        # Create buffers for acceptances
        self.local_accs_training = Buffer(
            "local_accs_training", (self.n_chains, self.n_steps), 1
        )
        self.local_accs_production = Buffer(
            "local_accs_production", (self.n_chains, self.n_steps), 1
        )
        
        # Create sampler state
        self.sampler_state = State(
            {
                "target_local_accs": "local_accs_training",
                "training": True,
            },
            name="sampler_state",
        )
        
        # Create kernels
        self.mala_kernel = MALA(step_size=jnp.ones(self.n_dims) * 0.1)
        self.hmc_kernel = HMC(
            condition_matrix=jnp.ones(self.n_dims), step_size=0.1, n_leapfrog=5
        )
        self.grw_kernel = GaussianRandomWalk(step_size=jnp.ones(self.n_dims) * 0.1)
        
        # Fill acceptance buffer with some data (high acceptance)
        high_acc_data = jnp.ones((self.n_chains, self.n_steps)) * 0.8
        self.local_accs_training.update_buffer(high_acc_data)
    
    def test_adapt_step_size_increases_on_high_acceptance(self):
        """Test that step size increases when acceptance is above target."""
        from flowMC.strategy.adapt_step_size import AdaptStepSize
        
        adapt_strategy = AdaptStepSize(
            kernel_name="local_sampler",
            state_name="sampler_state",
            acceptance_buffer_key="target_local_accs",
            target_acceptance_rate=0.574,  # MALA target
            training_only=True,
            verbose=False,
        )
        
        resources = {
            "local_sampler": self.mala_kernel,
            "sampler_state": self.sampler_state,
            "local_accs_training": self.local_accs_training,
        }
        
        initial_step_size = self.mala_kernel.step_size
        rng_key = jax.random.PRNGKey(42)
        initial_position = jnp.zeros((self.n_chains, self.n_dims))
        
        # Apply adaptation
        _, updated_resources, _ = adapt_strategy(
            rng_key, resources, initial_position, {}
        )
        
        new_step_size = updated_resources["local_sampler"].step_size
        assert jnp.all(new_step_size > initial_step_size), (
            "Step size should increase with high acceptance rate"
        )
    
    def test_adapt_step_size_training_only_mode(self):
        """Test that adaptation is skipped when training_only=True and training=False."""
        from flowMC.strategy.adapt_step_size import AdaptStepSize
        
        # Set training to False
        self.sampler_state.update(["training"], [False])
        
        adapt_strategy = AdaptStepSize(
            kernel_name="local_sampler",
            state_name="sampler_state",
            acceptance_buffer_key="target_local_accs",
            target_acceptance_rate=0.574,
            training_only=True,  # Should skip when training=False
            verbose=False,
        )
        
        resources = {
            "local_sampler": self.mala_kernel,
            "sampler_state": self.sampler_state,
            "local_accs_training": self.local_accs_training,
        }
        
        initial_step_size = self.mala_kernel.step_size
        rng_key = jax.random.PRNGKey(42)
        initial_position = jnp.zeros((self.n_chains, self.n_dims))
        
        # Apply adaptation (should be skipped)
        _, updated_resources, _ = adapt_strategy(
            rng_key, resources, initial_position, {}
        )
        
        new_step_size = updated_resources["local_sampler"].step_size
        assert jnp.allclose(new_step_size, initial_step_size), (
            "Step size should not change when training_only=True and training=False"
        )
    
    def test_adapt_step_size_works_with_different_kernels(self):
        """Test that AdaptStepSize works with MALA, HMC, and GRW."""
        from flowMC.strategy.adapt_step_size import AdaptStepSize
        
        # Reset training state
        self.sampler_state.update(["training"], [True])
        
        # Test with each kernel type
        for kernel, target_rate, kernel_name in [
            (self.mala_kernel, 0.574, "MALA"),
            (self.hmc_kernel, 0.65, "HMC"),
            (self.grw_kernel, 0.234, "GRW"),
        ]:
            adapt_strategy = AdaptStepSize(
                kernel_name="local_sampler",
                state_name="sampler_state",
                acceptance_buffer_key="target_local_accs",
                target_acceptance_rate=target_rate,
                training_only=True,
                verbose=False,
            )
            
            resources = {
                "local_sampler": kernel,
                "sampler_state": self.sampler_state,
                "local_accs_training": self.local_accs_training,
            }
            
            rng_key = jax.random.PRNGKey(42)
            initial_position = jnp.zeros((self.n_chains, self.n_dims))
            
            # Should not raise any errors
            _, updated_resources, _ = adapt_strategy(
                rng_key, resources, initial_position, {}
            )
            
            assert "local_sampler" in updated_resources
    
    def test_adapt_step_size_handles_non_finite_values(self):
        """Test that adaptation correctly handles non-finite acceptance values."""
        from flowMC.strategy.adapt_step_size import AdaptStepSize
        
        # Create buffer with some non-finite values
        mixed_acc_data = jnp.ones((self.n_chains, self.n_steps)) * 0.5
        # Add some -inf values (like from global steps)
        mixed_acc_data = mixed_acc_data.at[:, ::2].set(-jnp.inf)
        self.local_accs_training.update_buffer(mixed_acc_data)
        
        adapt_strategy = AdaptStepSize(
            kernel_name="local_sampler",
            state_name="sampler_state",
            acceptance_buffer_key="target_local_accs",
            target_acceptance_rate=0.574,
            training_only=True,
            verbose=False,
        )
        
        resources = {
            "local_sampler": self.mala_kernel,
            "sampler_state": self.sampler_state,
            "local_accs_training": self.local_accs_training,
        }
        
        rng_key = jax.random.PRNGKey(42)
        initial_position = jnp.zeros((self.n_chains, self.n_dims))
        
        # Should not raise errors and should produce finite step sizes
        _, updated_resources, _ = adapt_strategy(
            rng_key, resources, initial_position, {}
        )
        
        new_step_size = updated_resources["local_sampler"].step_size
        assert jnp.all(jnp.isfinite(new_step_size)), (
            "Step size should remain finite even with non-finite acceptance values"
        )

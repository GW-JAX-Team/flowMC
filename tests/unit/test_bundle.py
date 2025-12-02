import jax
import pytest
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle
from flowMC.resource_strategy_bundle.RQSpline_HMC import RQSpline_HMC_Bundle
from flowMC.resource_strategy_bundle.RQSpline_HMC_PT import RQSpline_HMC_PT_Bundle


def logpdf(x, _):
    return -0.5 * (x**2).sum()


class TestRQSplineMALABundle:
    """Tests for RQSpline_MALA_Bundle."""

    def test_initialization(self):
        bundle = RQSpline_MALA_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
        )
        assert repr(bundle) == "RQSpline MALA Bundle"

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="local_thinning.*must not exceed n_local_steps"):
            RQSpline_MALA_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=5,
                n_global_steps=10,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                local_thinning=10,
            )

    def test_global_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="global_thinning.*must not exceed n_global_steps"):
            RQSpline_MALA_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=10,
                n_global_steps=5,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                global_thinning=10,
            )


class TestRQSplineMALAPTBundle:
    """Tests for RQSpline_MALA_PT_Bundle."""

    def test_initialization(self):
        bundle = RQSpline_MALA_PT_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
        )
        assert repr(bundle) == "RQSpline MALA PT Bundle"

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="local_thinning.*must not exceed n_local_steps"):
            RQSpline_MALA_PT_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=5,
                n_global_steps=10,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                local_thinning=10,
            )

    def test_global_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="global_thinning.*must not exceed n_global_steps"):
            RQSpline_MALA_PT_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=10,
                n_global_steps=5,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                global_thinning=10,
            )


class TestRQSplineHMCBundle:
    """Tests for RQSpline_HMC_Bundle."""

    def test_initialization(self):
        bundle = RQSpline_HMC_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
        )
        assert repr(bundle) == "RQSpline HMC Bundle"

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="local_thinning.*must not exceed n_local_steps"):
            RQSpline_HMC_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=5,
                n_global_steps=10,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                local_thinning=10,
            )

    def test_global_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="global_thinning.*must not exceed n_global_steps"):
            RQSpline_HMC_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=10,
                n_global_steps=5,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                global_thinning=10,
            )


class TestRQSplineHMCPTBundle:
    """Tests for RQSpline_HMC_PT_Bundle."""

    def test_initialization(self):
        bundle = RQSpline_HMC_PT_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
        )
        assert repr(bundle) == "RQSpline HMC PT Bundle"

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="local_thinning.*must not exceed n_local_steps"):
            RQSpline_HMC_PT_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=5,
                n_global_steps=10,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                local_thinning=10,
            )

    def test_global_thinning_exceeds_steps(self):
        with pytest.raises(ValueError, match="global_thinning.*must not exceed n_global_steps"):
            RQSpline_HMC_PT_Bundle(
                rng_key=jax.random.PRNGKey(0),
                n_chains=2,
                n_dims=3,
                logpdf=logpdf,
                n_local_steps=10,
                n_global_steps=5,
                n_training_loops=1,
                n_production_loops=1,
                n_epochs=1,
                global_thinning=10,
            )

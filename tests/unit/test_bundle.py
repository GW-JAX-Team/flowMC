import jax
import jax.numpy as jnp
import pytest

from flowMC.resource.kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.kernel.HMC import HMC
from flowMC.resource.kernel.MALA import MALA
from flowMC.resource_strategy_bundle.RQSpline_GRW import RQSpline_GRW_Bundle
from flowMC.resource_strategy_bundle.RQSpline_GRW_PT import RQSpline_GRW_PT_Bundle
from flowMC.resource_strategy_bundle.RQSpline_HMC import RQSpline_HMC_Bundle
from flowMC.resource_strategy_bundle.RQSpline_HMC_PT import RQSpline_HMC_PT_Bundle
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle


def logpdf(x, _):
    return -0.5 * (x**2).sum()


class TestRQSplineMALABundle:
    """Tests for RQSpline_MALA_Bundle."""

    def test_initialization(self):
        bundle = RQSpline_MALA_Bundle(
            rng_key=jax.random.key(0),
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

        # Without periodic, mask should be all-False
        kernel = bundle.resources["local_sampler"]
        assert isinstance(kernel, MALA)
        assert not jnp.any(kernel.periodic_mask)

        # With periodic, mask and bounds should be set correctly
        periodic = {0: (0.0, 6.28), 2: (-3.14, 3.14)}
        bundle_p = RQSpline_MALA_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
            periodic=periodic,
        )
        kernel_p = bundle_p.resources["local_sampler"]
        assert isinstance(kernel_p, MALA)
        assert jnp.array_equal(kernel_p.periodic_mask, jnp.array([True, False, True]))
        assert jnp.isclose(kernel_p.periodic_bounds[0, 1], 6.28)
        assert jnp.isclose(kernel_p.periodic_bounds[2, 0], -3.14)

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(
            ValueError, match="local_thinning.*must not exceed n_local_steps"
        ):
            RQSpline_MALA_Bundle(
                rng_key=jax.random.key(0),
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
        with pytest.raises(
            ValueError, match="global_thinning.*must not exceed n_global_steps"
        ):
            RQSpline_MALA_Bundle(
                rng_key=jax.random.key(0),
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
            rng_key=jax.random.key(0),
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

        # Without periodic, mask should be all-False
        kernel = bundle.resources["local_sampler"]
        assert isinstance(kernel, MALA)
        assert not jnp.any(kernel.periodic_mask)

        # With periodic, mask should be set correctly
        periodic = {1: (0.0, 3.14)}
        bundle_p = RQSpline_MALA_PT_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
            periodic=periodic,
        )
        kernel_p = bundle_p.resources["local_sampler"]
        assert isinstance(kernel_p, MALA)
        assert jnp.array_equal(kernel_p.periodic_mask, jnp.array([False, True, False]))

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(
            ValueError, match="local_thinning.*must not exceed n_local_steps"
        ):
            RQSpline_MALA_PT_Bundle(
                rng_key=jax.random.key(0),
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
        with pytest.raises(
            ValueError, match="global_thinning.*must not exceed n_global_steps"
        ):
            RQSpline_MALA_PT_Bundle(
                rng_key=jax.random.key(0),
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
            rng_key=jax.random.key(0),
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

        # Without periodic, mask should be all-False
        kernel = bundle.resources["local_sampler"]
        assert isinstance(kernel, HMC)
        assert not jnp.any(kernel.periodic_mask)

        # With periodic, mask should be set correctly
        periodic = {0: (0.0, 6.28), 2: (-3.14, 3.14)}
        bundle_p = RQSpline_HMC_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
            periodic=periodic,
        )
        kernel_p = bundle_p.resources["local_sampler"]
        assert isinstance(kernel_p, HMC)
        assert jnp.array_equal(kernel_p.periodic_mask, jnp.array([True, False, True]))

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(
            ValueError, match="local_thinning.*must not exceed n_local_steps"
        ):
            RQSpline_HMC_Bundle(
                rng_key=jax.random.key(0),
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
        with pytest.raises(
            ValueError, match="global_thinning.*must not exceed n_global_steps"
        ):
            RQSpline_HMC_Bundle(
                rng_key=jax.random.key(0),
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
            rng_key=jax.random.key(0),
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

        # Without periodic, mask should be all-False
        kernel = bundle.resources["local_sampler"]
        assert isinstance(kernel, HMC)
        assert not jnp.any(kernel.periodic_mask)

        # With periodic, mask should be set correctly
        periodic = {1: (0.0, 3.14)}
        bundle_p = RQSpline_HMC_PT_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
            periodic=periodic,
        )
        kernel_p = bundle_p.resources["local_sampler"]
        assert isinstance(kernel_p, HMC)
        assert jnp.array_equal(kernel_p.periodic_mask, jnp.array([False, True, False]))

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(
            ValueError, match="local_thinning.*must not exceed n_local_steps"
        ):
            RQSpline_HMC_PT_Bundle(
                rng_key=jax.random.key(0),
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
        with pytest.raises(
            ValueError, match="global_thinning.*must not exceed n_global_steps"
        ):
            RQSpline_HMC_PT_Bundle(
                rng_key=jax.random.key(0),
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


class TestRQSplineGRWBundle:
    """Tests for RQSpline_GRW_Bundle."""

    def test_initialization(self):
        bundle = RQSpline_GRW_Bundle(
            rng_key=jax.random.key(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
        )
        assert repr(bundle) == "RQSpline GRW Bundle"

        # Without periodic, mask should be all-False
        kernel = bundle.resources["local_sampler"]
        assert isinstance(kernel, GaussianRandomWalk)
        assert not jnp.any(kernel.periodic_mask)

        # With periodic, mask should be set correctly
        periodic = {0: (0.0, 6.28), 2: (-3.14, 3.14)}
        bundle_p = RQSpline_GRW_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
            periodic=periodic,
        )
        kernel_p = bundle_p.resources["local_sampler"]
        assert isinstance(kernel_p, GaussianRandomWalk)
        assert jnp.array_equal(kernel_p.periodic_mask, jnp.array([True, False, True]))

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(
            ValueError, match="local_thinning.*must not exceed n_local_steps"
        ):
            RQSpline_GRW_Bundle(
                rng_key=jax.random.key(0),
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
        with pytest.raises(
            ValueError, match="global_thinning.*must not exceed n_global_steps"
        ):
            RQSpline_GRW_Bundle(
                rng_key=jax.random.key(0),
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


class TestRQSplineGRWPTBundle:
    """Tests for RQSpline_GRW_PT_Bundle."""

    def test_initialization(self):
        bundle = RQSpline_GRW_PT_Bundle(
            rng_key=jax.random.key(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
        )
        assert repr(bundle) == "RQSpline GRW PT Bundle"

        # Without periodic, mask should be all-False
        kernel = bundle.resources["local_sampler"]
        assert isinstance(kernel, GaussianRandomWalk)
        assert not jnp.any(kernel.periodic_mask)

        # With periodic, mask should be set correctly
        periodic = {1: (0.0, 3.14)}
        bundle_p = RQSpline_GRW_PT_Bundle(
            rng_key=jax.random.PRNGKey(0),
            n_chains=2,
            n_dims=3,
            logpdf=logpdf,
            n_local_steps=10,
            n_global_steps=5,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=3,
            periodic=periodic,
        )
        kernel_p = bundle_p.resources["local_sampler"]
        assert isinstance(kernel_p, GaussianRandomWalk)
        assert jnp.array_equal(kernel_p.periodic_mask, jnp.array([False, True, False]))

    def test_local_thinning_exceeds_steps(self):
        with pytest.raises(
            ValueError, match="local_thinning.*must not exceed n_local_steps"
        ):
            RQSpline_GRW_PT_Bundle(
                rng_key=jax.random.key(0),
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
        with pytest.raises(
            ValueError, match="global_thinning.*must not exceed n_global_steps"
        ):
            RQSpline_GRW_PT_Bundle(
                rng_key=jax.random.key(0),
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


@pytest.mark.parametrize(
    "bundle_cls",
    [RQSpline_MALA_PT_Bundle, RQSpline_HMC_PT_Bundle, RQSpline_GRW_PT_Bundle],
)
def test_pt_bundle_tempers_only_the_likelihood(bundle_cls):
    def logprior(x, _):
        return -0.5 * ((x - 3.0) ** 2).sum()

    bundle = bundle_cls(
        rng_key=jax.random.key(0),
        n_chains=2,
        n_dims=3,
        logpdf=logpdf,
        n_local_steps=10,
        n_global_steps=5,
        n_training_loops=2,
        n_production_loops=1,
        n_epochs=3,
        logprior=logprior,
    )
    x = jnp.array([0.5, -1.0, 2.0])
    log_likelihood, log_prior = logpdf(x, {}), logprior(x, {})
    assert jnp.isclose(bundle.resources["logpdf"](x, {}), log_likelihood + log_prior)
    assert jnp.isclose(
        bundle.resources["tempered_logpdf"].tempered_log_pdf(4.0, x, {}),
        log_likelihood / 4.0 + log_prior,
    )

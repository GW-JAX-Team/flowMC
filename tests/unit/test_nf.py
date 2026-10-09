import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowMC.resource.model.common import Gaussian
from flowMC.resource.model.nf_model.realNVP import AffineCoupling, RealNVP
from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline


def test_affine_coupling_forward_and_inverse():
    n_features = 2
    n_hidden = 4
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.where(jnp.arange(n_features) % 2 == 0, 1.0, 0.0)
    key = jax.random.key(0)
    dt = 0.5
    layer = AffineCoupling(n_features, n_hidden, mask, key, dt)

    y_forward, log_det_forward = jax.vmap(layer.forward)(x)
    x_recon, log_det_inverse = jax.vmap(layer.inverse)(y_forward)

    assert jnp.allclose(x, jnp.round(x_recon, decimals=5))
    assert jnp.allclose(log_det_forward, -log_det_inverse)


def test_realnvp():
    n_features = 3
    n_hidden = 4
    n_layers = 2
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    rng_key, _rng_subkey = jax.random.split(jax.random.key(0), 2)
    model = RealNVP(n_features, n_layers, n_hidden, rng_key)

    assert model.n_features == n_features

    y, log_det = jax.vmap(model)(x)

    assert y.shape == x.shape
    assert log_det.shape == (2,)

    y_inv, log_det_inv = jax.vmap(model.inverse)(y)

    assert y_inv.shape == x.shape
    assert log_det_inv.shape == (2,)
    assert jnp.allclose(x, y_inv)
    assert jnp.allclose(log_det, -log_det_inv)

    rng_key = jax.random.key(0)
    samples = model.sample(rng_key, 2)

    assert samples.shape == (2, 3)

    log_prob = jax.vmap(model.log_prob)(samples)

    assert log_prob.shape == (2,)


def test_realnvp_log_prob_custom_base_dist():
    n_features = 3
    n_hidden = 4
    n_layers = 2
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    rng_key, _rng_subkey = jax.random.split(jax.random.key(0), 2)
    base_dist = Gaussian(
        jnp.array([1.0, -2.0, 0.5]),
        jnp.diag(jnp.array([2.0, 0.5, 1.5])),
        learnable=False,
    )
    model = RealNVP(n_features, n_layers, n_hidden, rng_key, base_dist=base_dist)

    y, log_det = jax.vmap(model)(x)
    expected_log_prob = jax.vmap(base_dist.log_prob)(y) + log_det
    actual_log_prob = jax.vmap(model.log_prob)(x)

    assert jnp.allclose(actual_log_prob, expected_log_prob)
    # Sanity check: a non-standard base_dist should disagree with a standard normal.
    standard_normal_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        y, jnp.zeros(n_features), jnp.eye(n_features)
    )
    assert not jnp.allclose(actual_log_prob, standard_normal_log_prob + log_det)


def test_rqspline():
    n_features = 3
    hidden_layes = [16, 16]
    n_layers = 2
    n_bins = 8

    rng_key, _rng_subkey = jax.random.split(jax.random.key(0), 2)
    model = MaskedCouplingRQSpline(
        n_features, n_layers, hidden_layes, n_bins, jax.random.key(10)
    )

    assert model.n_features == n_features

    rng_key = jax.random.key(0)
    samples = model.sample(rng_key, 2)

    assert samples.shape == (2, 3)

    log_prob = jax.vmap(model.log_prob)(samples)

    assert log_prob.shape == (2,)


def _rqspline_with_nonidentity_affines(n_features, n_layers, **kwargs):
    model = MaskedCouplingRQSpline(
        n_features, n_layers, [8, 8], 4, jax.random.key(10), **kwargs
    )
    # Identity affine initialization hides an incorrect inverse order inside a block.
    return eqx.tree_at(
        lambda m: (m.layers[0].bijector.scale, m.layers[0].bijector.shift),
        model,
        (
            jnp.linspace(0.2, -0.15, n_layers),
            jnp.linspace(0.3, -0.2, n_layers),
        ),
    )


@pytest.mark.parametrize("n_features", [2, 3])
@pytest.mark.parametrize("n_layers", [1, 3])
def test_rqspline_forward_inverse_roundtrips(n_features, n_layers):
    model = _rqspline_with_nonidentity_affines(n_features, n_layers)
    # Include interior points and both linear tails of the default [-10, 10] spline.
    points = jnp.array(
        [
            [-2.0, 0.7, 1.5],
            [0.2, -1.4, 3.0],
            [0.0, 0.0, 0.0],
            [-12.0, 11.0, -13.0],
            [12.0, -11.0, 13.0],
        ]
    )[:, :n_features]

    forward = jax.vmap(model.forward)
    inverse = jax.vmap(model.inverse)
    for compiled in (False, True):
        if compiled:
            forward = jax.jit(forward)
            inverse = jax.jit(inverse)

        latent, forward_logdet = forward(points)
        recovered, inverse_logdet = inverse(latent)
        np.testing.assert_allclose(recovered, points, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            forward_logdet, -inverse_logdet, rtol=1e-5, atol=1e-5
        )

        data, inverse_logdet = inverse(points)
        recovered, forward_logdet = forward(data)
        np.testing.assert_allclose(recovered, points, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            forward_logdet, -inverse_logdet, rtol=1e-5, atol=1e-5
        )


def test_rqspline_sample_log_prob_consistency():
    model = _rqspline_with_nonidentity_affines(
        3,
        3,
        data_mean=jnp.array([1.0, -2.0, 0.5]),
        data_cov=jnp.diag(jnp.array([0.04, 9.0, 0.25])),
    )
    key = jax.random.key(20)
    latents = model.base_dist.sample(key, 32)
    _, inverse_logdet = jax.vmap(model.inverse)(latents)
    scale = jnp.sqrt(jnp.diag(model.data_cov))
    expected_log_prob = (
        jax.vmap(model.base_dist.log_prob)(latents)
        - inverse_logdet
        - jnp.sum(jnp.log(scale))
    )

    # Check the public sampling/density API against the density of its known draws.
    for sample, log_prob in (
        (model.sample, model.log_prob),
        (eqx.filter_jit(model.sample), eqx.filter_jit(model.log_prob)),
    ):
        samples = sample(key, 32)
        recovered_latents, _ = jax.vmap(model.forward)(
            (samples - model.data_mean) / scale
        )
        np.testing.assert_allclose(recovered_latents, latents, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            log_prob(samples), expected_log_prob, rtol=1e-5, atol=1e-5
        )

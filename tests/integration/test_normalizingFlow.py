import equinox as eqx  # Equinox utilities
import jax
import jax.numpy as jnp  # JAX NumPy
import optax  # Optimizers

from flowMC.resource.model.nf_model.realNVP import RealNVP
from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline


def test_realNVP():
    key1, rng, init_rng = jax.random.split(jax.random.key(0), 3)
    data = jax.random.normal(key1, (100, 2))

    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    model = RealNVP(2, 4, 32, rng)
    optim = optax.adam(learning_rate, momentum)
    state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    rng, _best_model, state, _loss_values = model.train(
        init_rng, data, optim, state, num_epochs, batch_size, verbose=True
    )
    rng_key_nf = jax.random.key(124098)
    model.sample(rng_key_nf, 10000)


def test_rqSpline():
    n_dim = 2
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    key1, rng, init_rng = jax.random.split(jax.random.key(0), 3)
    data = jax.random.normal(key1, (batch_size, n_dim))

    n_layers = 4
    hidden_dim = 32
    num_bins = 4

    model = MaskedCouplingRQSpline(
        n_dim,
        n_layers,
        [hidden_dim, hidden_dim],
        num_bins,
        rng,
        data_mean=jnp.mean(data, axis=0),
        data_cov=jnp.cov(data.T),
    )
    optim = optax.adam(learning_rate, momentum)
    state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    rng, model, state, _loss_values = model.train(
        init_rng, data, optim, state, num_epochs, batch_size, verbose=True
    )
    rng_key_nf = jax.random.key(124098)
    model.sample(rng_key_nf, 10000)

    # Training moves the affine layers away from identity, exposing composition bugs.
    assert jnp.any(model.layers[0].bijector.scale != 0) or jnp.any(
        model.layers[0].bijector.shift != 0
    )
    latent = model.base_dist.sample(rng_key_nf, 64)
    standardized, log_det_inverse = jax.vmap(model.inverse)(latent)
    reconstructed, log_det_forward = jax.vmap(model.forward)(standardized)
    assert jnp.allclose(reconstructed, latent, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(log_det_inverse + log_det_forward, 0, atol=2e-5)

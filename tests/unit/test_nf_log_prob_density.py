"""Tests for normalized normalizing-flow log densities in data space."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.model.nf_model.realNVP import RealNVP

N_DIM = 4
# deliberately far from the identity: prod(sqrt(diag)) = 0.2*3*0.5*1 = 0.3
DATA_MEAN = jnp.array([1.0, -2.0, 0.5, 3.0])
DATA_COV = jnp.diag(jnp.array([0.04, 9.0, 0.25, 1.0]))


def _models():
    key = jax.random.PRNGKey(0)
    yield "rqSpline", MaskedCouplingRQSpline(
        N_DIM, 3, [16, 16], 4, key, data_mean=DATA_MEAN, data_cov=DATA_COV
    )
    yield "realNVP", RealNVP(
        N_DIM, 4, 16, key, data_mean=DATA_MEAN, data_cov=DATA_COV
    )


def _log_gauss(x, mu, cov):
    d = x - mu[None, :]
    sol = np.linalg.solve(cov, d.T).T
    return (-0.5 * np.einsum("ij,ij->i", d, sol)
            - 0.5 * x.shape[1] * np.log(2 * np.pi)
            - 0.5 * np.linalg.slogdet(cov)[1])


@pytest.mark.parametrize("name", ["rqSpline", "realNVP"])
def test_log_prob_integrates_to_one(name):
    """Z = int exp(log_prob(x)) dx == 1, by importance sampling against a broad
    Gaussian covering the flow.  This is the test the missing Jacobian fails."""
    model = dict(_models())[name]
    rng = np.random.default_rng(0)
    s = np.asarray(model.sample(jax.random.PRNGKey(2), 20000))
    mu = s.mean(0)
    cov = np.cov(s.T) * 4.0 + 1e-9 * np.eye(N_DIM)   # inflate: cover the tails
    Lc = np.linalg.cholesky(cov)
    n = 200000
    z = mu[None, :] + rng.standard_normal((n, N_DIM)) @ Lc.T
    lw = np.asarray(model.log_prob(jnp.asarray(z))) - _log_gauss(z, mu, cov)
    m = lw.max()
    w = np.exp(lw - m)
    Z = float(np.exp(m) * w.mean())
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    assert ess > 2000, "proposal does not cover the flow (ESS=%.0f); test is void" % ess
    # MC error on Z is ~ 1/sqrt(ESS); 5% is many sigma yet far tighter than the
    # 0.3x error the missing Jacobian produces on this data_cov.
    assert abs(Z - 1.0) < 0.05, (
        "%s: exp(log_prob) integrates to %.5f, not 1 "
        "(prod(sqrt(diag(data_cov))) = %.5f -- missing standardization Jacobian?)"
        % (name, Z, float(jnp.prod(jnp.sqrt(jnp.diag(DATA_COV))))))


@pytest.mark.parametrize("name", ["rqSpline", "realNVP"])
def test_log_prob_batched_matches_looped(name):
    """log_prob must accept an (n_sample, n_dim) batch and agree elementwise
    with looping over single (n_dim,) samples."""
    model = dict(_models())[name]
    x = jnp.asarray(model.sample(jax.random.PRNGKey(3), 64))
    batched = np.asarray(model.log_prob(x))
    assert batched.shape == (64,), "batched log_prob returned %r" % (batched.shape,)
    looped = np.array([float(model.log_prob(xi)) for xi in x])
    assert np.allclose(batched, looped, rtol=1e-6, atol=1e-5), (
        "%s: batched and looped log_prob differ by up to %.3g"
        % (name, np.max(np.abs(batched - looped))))


@pytest.mark.parametrize("name", ["rqSpline", "realNVP"])
def test_log_prob_single_sample_returns_scalar(name):
    """The single-sample contract every internal caller relies on (loss_fn and
    NF_proposal both vmap log_prob over rows) must keep working."""
    model = dict(_models())[name]
    x = jnp.asarray(model.sample(jax.random.PRNGKey(4), 1))[0]
    assert np.asarray(model.log_prob(x)).shape == ()
    v = jax.vmap(model.log_prob)(jnp.asarray(model.sample(jax.random.PRNGKey(5), 8)))
    assert np.asarray(v).shape == (8,)


def test_identity_data_cov_is_unaffected():
    """Control: with data_cov = I the Jacobian is 0, so the fix must be a no-op
    there.  Guards against 'fixing' normalization with an unrelated constant."""
    key = jax.random.PRNGKey(7)
    m = MaskedCouplingRQSpline(N_DIM, 3, [16, 16], 4, key,
                               data_mean=jnp.zeros(N_DIM), data_cov=jnp.eye(N_DIM))
    x = jnp.asarray(m.sample(jax.random.PRNGKey(8), 16))
    scale = jnp.sqrt(jnp.diag(m.data_cov))
    y, log_det = m.__call__((x[0] - m.data_mean) / scale)
    raw = float(log_det + m.base_dist.log_prob(y))
    assert abs(float(m.log_prob(x[0])) - raw) < 1e-12

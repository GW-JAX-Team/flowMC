# Resource-Strategy Bundles

A **resource-strategy bundle** packages a local MCMC sampler, a normalizing flow global proposal, and a training schedule into a single object.
You pass it to `Sampler` and it configures everything for you.

```python
from flowMC.Sampler import Sampler

sampler = Sampler(
    n_dim=n_dims,
    n_chains=n_chains,
    rng_key=rng_key,
    resource_strategy_bundles=bundle,
)
sampler.sample(initial_positions, data)
```

---

## Available bundles

| Bundle | Local sampler | Parallel tempering |
| --- | --- | --- |
| `RQSpline_MALA_Bundle` | MALA | No |
| `RQSpline_MALA_PT_Bundle` | MALA | Yes |
| `RQSpline_HMC_Bundle` | HMC | No |
| `RQSpline_HMC_PT_Bundle` | HMC | Yes |
| `RQSpline_GRW_Bundle` | Gaussian random walk | No |
| `RQSpline_GRW_PT_Bundle` | Gaussian random walk | Yes |

All bundles are importable from `flowMC.resource_strategy_bundle`.

---

## RQSpline_MALA_Bundle

Uses the **Metropolis-Adjusted Langevin Algorithm (MALA)** as the local sampler.
MALA uses the gradient of the log-density to bias proposals toward regions of higher probability, giving better performance than a simple random walk at the cost of requiring a differentiable `logpdf`.

```python
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

bundle = RQSpline_MALA_Bundle(
    rng_key=rng_key,
    n_chains=500,
    n_dims=n_dims,
    logpdf=logpdf,
    n_local_steps=25,
    n_global_steps=25,
    n_training_loops=20,
    n_production_loops=10,
    n_epochs=30,
)
```

The local step size targets an acceptance rate of ~57% and is adapted automatically during the training phase when `adapt_step_size=True` (the default).

---

## RQSpline_MALA_PT_Bundle

Extends `RQSpline_MALA_Bundle` with **parallel tempering**. Additional replicas of the chains are run at elevated temperatures, exploring the prior more freely.
Periodic swap proposals between adjacent temperature levels allow the target chains to escape local modes.

```python
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle

bundle = RQSpline_MALA_PT_Bundle(
    rng_key=rng_key,
    n_chains=500,
    n_dims=n_dims,
    logpdf=logpdf,
    n_local_steps=25,
    n_global_steps=25,
    n_training_loops=20,
    n_production_loops=10,
    n_epochs=30,
    # Parallel tempering
    logprior=logprior,
    n_temperatures=5,
    max_temperature=5.0,
)
```

The tempered log-density at temperature $T$ is

$$\log \tilde{p}_T(x) = \frac{1}{T} \log \mathcal{L}(x) + \log \pi(x)$$

so the prior is preserved at all temperatures.

---

## RQSpline_HMC_Bundle

Uses **Hamiltonian Monte Carlo (HMC)** as the local sampler.
HMC integrates the Hamiltonian equations of motion using a leapfrog integrator, producing proposals that travel far in parameter space with high acceptance.
It requires a differentiable `logpdf` and benefits from a good `condition_matrix` (diagonal inverse mass matrix).

```python
from flowMC.resource_strategy_bundle.RQSpline_HMC import RQSpline_HMC_Bundle

bundle = RQSpline_HMC_Bundle(
    rng_key=rng_key,
    n_chains=500,
    n_dims=n_dims,
    logpdf=logpdf,
    n_local_steps=25,
    n_global_steps=25,
    n_training_loops=20,
    n_production_loops=10,
    n_epochs=30,
    hmc_step_size=0.1,
    hmc_n_leapfrog=10,
    condition_matrix=1,   # or a 1-D array of per-dimension scales
)
```

The local step size targets an acceptance rate of ~65%.

---

## RQSpline_HMC_PT_Bundle

Extends `RQSpline_HMC_Bundle` with parallel tempering.
The constructor accepts all `RQSpline_HMC_Bundle` parameters plus the parallel tempering parameters (`logprior`, `n_temperatures`, `max_temperature`, `n_tempered_steps`).

```python
from flowMC.resource_strategy_bundle.RQSpline_HMC_PT import RQSpline_HMC_PT_Bundle
```

---

## RQSpline_GRW_Bundle

Uses a **Gaussian random walk** (Metropolis–Hastings) as the local sampler.
No gradient is required, making this the right choice when `logpdf` is not differentiable.
The optimal acceptance rate for a Gaussian random walk is ~23%.

```python
from flowMC.resource_strategy_bundle.RQSpline_GRW import RQSpline_GRW_Bundle

bundle = RQSpline_GRW_Bundle(
    rng_key=rng_key,
    n_chains=500,
    n_dims=n_dims,
    logpdf=logpdf,
    n_local_steps=25,
    n_global_steps=25,
    n_training_loops=20,
    n_production_loops=10,
    n_epochs=30,
    grw_step_size=0.1,
)
```

---

## RQSpline_GRW_PT_Bundle

Extends `RQSpline_GRW_Bundle` with parallel tempering.

```python
from flowMC.resource_strategy_bundle.RQSpline_GRW_PT import RQSpline_GRW_PT_Bundle
```

---

## Sampling loop structure

Every bundle organises sampling into a **training phase** followed by a **production phase**.

**Training phase** (`n_training_loops` iterations):

1. Run `n_local_steps` local MCMC steps per chain
2. (Optional) Adapt the local step size
3. Run `n_global_steps` NF proposal steps per chain
4. Train the normalizing flow for `n_epochs` on accumulated samples
5. (PT bundles) Attempt parallel-tempering swaps at the start of each loop

**Production phase** (`n_production_loops` iterations):

1. Run `n_local_steps` local MCMC steps per chain
2. Run `n_global_steps` NF proposal steps per chain
3. (PT bundles) Attempt parallel-tempering swaps at the start of each loop

The flow is not updated during the production phase, so detailed balance is restored and standard MCMC convergence diagnostics can be applied safely.

Early stopping can terminate the training phase early once the global acceptance rate has stabilised; see [`early_stopping`](hyperparameters.md#early_stopping).

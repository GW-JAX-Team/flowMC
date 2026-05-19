# Hyperparameter Reference

Quick-reference index by category. Click any parameter name to jump to its description.

## Required

| Parameter | Description |
| --- | --- |
| [`rng_key`](#rng_key) | JAX PRNG key |
| [`n_chains`](#n_chains) | Number of parallel chains |
| [`n_dims`](#n_dims) | Dimensionality of the parameter space |
| [`logpdf`](#logpdf) | Log-density function |
| [`n_local_steps`](#n_local_steps) | Local MCMC steps per loop |
| [`n_global_steps`](#n_global_steps) | NF proposal steps per loop |
| [`n_training_loops`](#n_training_loops) | Number of training phase loops |
| [`n_production_loops`](#n_production_loops) | Number of production phase loops |
| [`n_epochs`](#n_epochs) | Training epochs per loop |

## Local sampler — MALA

| Parameter | Default | Description |
| --- | --- | --- |
| [`mala_step_size`](#mala_step_size) | `1e-1` | Initial MALA step size |
| [`adapt_step_size`](#adapt_step_size) | `True` | Auto-adapt global step size |
| [`adapt_step_size_per_dim`](#adapt_step_size_per_dim) | `True` | Auto-adapt per-dimension step sizes |
| [`periodic`](#periodic) | `None` | Periodic dimensions |

## Local sampler — HMC

| Parameter | Default | Description |
| --- | --- | --- |
| [`hmc_step_size`](#hmc_step_size) | `0.1` | Initial leapfrog step size |
| [`hmc_n_leapfrog`](#hmc_n_leapfrog) | `10` | Leapfrog steps per proposal |
| [`condition_matrix`](#condition_matrix) | `1` | Diagonal inverse mass matrix |
| [`adapt_step_size`](#adapt_step_size) | `True` | Auto-adapt global step size |
| [`adapt_step_size_per_dim`](#adapt_step_size_per_dim) | `True` | Auto-adapt per-dimension step sizes |
| [`periodic`](#periodic) | `None` | Periodic dimensions |

## Local sampler — Gaussian random walk

| Parameter | Default | Description |
| --- | --- | --- |
| [`grw_step_size`](#grw_step_size) | `1e-1` | Initial random walk step size |
| [`adapt_step_size`](#adapt_step_size) | `True` | Auto-adapt global step size |
| [`adapt_step_size_per_dim`](#adapt_step_size_per_dim) | `True` | Auto-adapt per-dimension step sizes |
| [`periodic`](#periodic) | `None` | Periodic dimensions |

## Normalizing flow

| Parameter | Default | Description |
| --- | --- | --- |
| [`rq_spline_n_layers`](#rq_spline_n_layers) | `4` | Number of coupling layers |
| [`rq_spline_hidden_units`](#rq_spline_hidden_units) | `[32, 32]` | Hidden layer widths |
| [`rq_spline_n_bins`](#rq_spline_n_bins) | `8` | Spline bins per layer |
| [`n_NFproposal_batch_size`](#n_nfproposal_batch_size) | `10000` | NF sampling batch size |

## Training

| Parameter | Default | Description |
| --- | --- | --- |
| [`learning_rate`](#learning_rate) | `1e-3` | AdamW learning rate |
| [`batch_size`](#batch_size) | `10000` | Training mini-batch size |
| [`n_max_examples`](#n_max_examples) | `10000` | Max samples in training buffer |
| [`history_window`](#history_window) | `100` | Most-recent steps per chain used as NF training data |

## Execution

| Parameter | Default | Description |
| --- | --- | --- |
| [`chain_batch_size`](#chain_batch_size) | `0` | Split vmap over chains |
| [`local_thinning`](#local_thinning) | `1` | Thinning for local samples |
| [`global_thinning`](#global_thinning) | `1` | Thinning for global samples |
| [`verbose`](#verbose) | `False` | Print progress |

## Early stopping

| Parameter | Default | Description |
| --- | --- | --- |
| [`early_stopping`](#early_stopping) | `False` | Enable early stopping |
| [`early_stopping_tolerance`](#early_stopping_tolerance) | `0.05` | Relative acceptance-rate change threshold |
| [`early_stopping_patience`](#early_stopping_patience) | `3` | Consecutive loops to confirm stability |
| [`early_stopping_min_acceptance`](#early_stopping_min_acceptance) | `0.1` | Acceptance rate that triggers early stopping |

## Parallel tempering (PT bundles only)

| Parameter | Default | Description |
| --- | --- | --- |
| [`logprior`](#logprior) | flat | Log-prior function |
| [`n_temperatures`](#n_temperatures) | `5` | Number of temperature levels |
| [`max_temperature`](#max_temperature) | `5.0` | Highest temperature |
| [`n_tempered_steps`](#n_tempered_steps) | `-1` | Swap attempts per loop |

---

## Required parameters

### rng_key

A JAX PRNG key used to initialise all random state inside the bundle (chain positions, model weights, optimizer state).

### n_chains

Number of parallel chains.
The algorithm parallelises all chains simultaneously via `vmap`, so the performance benefit saturates once the chains fill the available hardware threads.
More chains give a better picture of the global landscape, which helps train the normalizing flow faster and reduces the chance of mode collapse.

### n_dims

Dimensionality of the parameter space. Must match the input dimension of `logpdf`.

### logpdf

A callable with signature `logpdf(x, data) -> float` where `x` is a 1-D array of length `n_dims` and `data` is the auxiliary data passed to `Sampler.sample`.
The function must be JAX-differentiable if you use MALA or HMC.

### n_local_steps

Number of local MCMC steps executed per chain in each training and production loop.
Together with `n_chains` this controls how many samples are added to the training buffer per loop.

### n_global_steps

Number of normalizing-flow proposal steps executed per chain in each loop.
These are independent-MH moves drawn from the current flow, so they are most effective once the flow has been trained for a few loops.

### n_training_loops

Number of local–global–train cycles to run in the training phase.
During each cycle the local sampler runs, the flow is trained on the accumulated samples, and the global proposer runs.
Increasing this gives the flow more opportunities to improve, at the cost of a longer warmup.

### n_production_loops

Number of local–global cycles to run in the production phase.
The flow is not updated during production, so detailed balance is restored and standard MCMC convergence diagnostics apply.
All production samples are stored.

### n_epochs

Number of gradient-descent epochs per training step.
Higher values improve the flow fit but increase training time per loop.

---

## Local sampler parameters

### mala_step_size

*MALA bundles only.* Initial step size for MALA.
Default `1e-1`.
Accepts either a scalar (broadcast to all dimensions) or a 1-D array of length `n_dims` to set a different initial step size per dimension.
The optimal MALA acceptance rate is ~57%, and `adapt_step_size` will drive the step size toward that target automatically.

### hmc_step_size

*HMC bundles only.* Initial leapfrog step size.
Default `0.1`.
The optimal HMC acceptance rate is ~65%.

### hmc_n_leapfrog

*HMC bundles only.* Number of leapfrog steps per HMC proposal. Default `10`.
More steps produce proposals further along the trajectory but increase cost per step.

### condition_matrix

*HMC bundles only.* Diagonal of the inverse mass matrix (preconditioning matrix).
Default `1` (scalar, broadcast to all dimensions).
Pass a 1-D array of length `n_dims` to set per-dimension scales.
Useful when parameters have very different characteristic scales.

### grw_step_size

*GRW bundles only.* Initial step size for the Gaussian random walk proposal.
Default `1e-1`.
Accepts either a scalar (broadcast to all dimensions) or a 1-D array of length `n_dims` to set a different initial step size per dimension.
The optimal random-walk acceptance rate is ~23%.

### adapt_step_size

Whether to automatically adapt the local step size during the training phase.
Default `True`.
The step size is updated after each training loop using a simple multiplicative rule targeting the algorithm-specific optimal acceptance rate.
Adaptation is disabled during the production phase.

### adapt_step_size_per_dim

Whether to additionally tune the per-dimension step size profile using the empirical posterior scale.
Default `True`.
After the global step size is adjusted by `adapt_step_size`, this strategy rescales each dimension's step size so that parameters with a larger posterior scale get proportionally larger steps.
The geometric mean of the step sizes is preserved, so `adapt_step_size` retains full control of the overall magnitude, while this strategy only reshapes the per-dimension profile.
For MALA and GRW, the per-dimension `step_size` array is scaled directly; for HMC the `condition_matrix` (diagonal mass matrix) is adjusted instead.
Adaptation is disabled during the production phase.

### periodic

A `dict[int, tuple[float, float]]` mapping dimension indices to `(lower, upper)` bounds for periodic dimensions.
Default `None` (no periodic dimensions).
Proposals in periodic dimensions are automatically wrapped into `[lower, upper]`.

```python
# e.g. dim 0 is periodic on [0, 2π] and dim 2 on [-1, 1]
periodic = {0: (0.0, 6.283), 2: (-1.0, 1.0)}
```

---

## Normalizing flow parameters

### rq_spline_n_layers

Number of masked coupling layers in the RQ-spline normalizing flow.
Default `4`.
More layers increase expressive power at the cost of memory and training time.

### rq_spline_hidden_units

List of hidden layer widths for the conditioner MLPs inside each coupling layer.
Default `[32, 32]`.
Increasing the width or depth improves the flow's capacity to represent complex distributions.

### rq_spline_n_bins

Number of bins for the rational-quadratic spline bijection.
Default `8`.
More bins give a finer piecewise approximation.

### n_NFproposal_batch_size

Batch size used when running the NF proposal over many steps.
Default `10000`.
When `n_global_steps` exceeds this value, the flow samples are computed in chunks of this size using `jax.lax.map` rather than a single `vmap`, reducing peak memory.

---

## Training parameters

### learning_rate

Learning rate for the AdamW optimizer used to train the normalizing flow. Default `1e-3`.

### batch_size

Mini-batch size for each gradient step during flow training.
Default `10000`.
The training loss is computed on random mini-batches of the accumulated samples.
Within the memory budget, larger batches tend to be better because the training dataset is continuously evolving and overfitting is not a concern.

### n_max_examples

Maximum number of samples retained in the training buffer.
Default `10000`.
When the buffer exceeds this size, only the most recent `n_max_examples` samples are kept for training.
Choose this based on your device's memory capacity.
Setting it too low risks mode collapse, as the flow may forget regions of the posterior not recently visited by the chains.

### history_window

Number of most-recent position steps per chain used when selecting training data for the normalizing flow.
Default `100`.
At each training step, only the last `history_window` positions from each chain's buffer are considered; `n_max_examples` samples are then drawn from that pool.
Reducing this value focuses the flow on the current mode but risks forgetting regions not recently visited.

---

## Execution parameters

### chain_batch_size

If non-zero, splits the `vmap` over chains into sequential batches of this size.
Default `0` (no splitting; all chains are vmapped at once).
Use this when vmapping over all chains simultaneously exceeds device memory.

### local_thinning

Store every `local_thinning`-th position from the local sampler.
Default `1` (store all).
All `n_local_steps` are always executed — thinning only controls which steps are written to the buffer.
Increasing it reduces memory and the number of training samples available to the normalizing flow, but also reduces autocorrelation in the stored samples.
The acceptance rate stored per slot is the mean over the corresponding thinning window.
Note that `local_thinning` must not exceed `n_local_steps`.

### global_thinning

Store every `global_thinning`-th position from the NF global proposal.
Default `1`.
Same semantics as `local_thinning`: all `n_global_steps` are always executed, but only every `global_thinning`-th result is written to the buffer.
This affects memory, the acceptance-rate values seen by early stopping, and the number of samples available in the training buffer.
`global_thinning` must not exceed `n_global_steps`.

A key practical motivation for increasing `global_thinning` is that the normalizing flow can never perfectly represent the posterior, so the global acceptance rate is typically low.
When many consecutive proposals are rejected, the chain stays at the same position, and storing every step fills the buffer with duplicate points that contribute nothing to the posterior estimate or NF training.
Thinning skips over these repeated positions while still executing all `n_global_steps`, preserving the full opportunity to make large jumps whenever a proposal is accepted.

### verbose

Print training loss and acceptance rates during sampling. Default `False`.

---

## Early stopping parameters

### early_stopping

Whether to enable early stopping of the training phase.
Default `False`.
When enabled, the training phase ends early and the sampler transitions to the production phase once either of the two stopping conditions is met (see `early_stopping_tolerance` and `early_stopping_min_acceptance` below).
The first 3 training loops are always completed regardless.

### early_stopping_tolerance

Relative change threshold for the global acceptance rate stability check.
Early stopping triggers when both the mean global acceptance rate **and** its coefficient of variation across chains change by less than this fraction for `early_stopping_patience` consecutive loops: `|current - prev| / prev < tolerance`.
Default `0.05`.

### early_stopping_patience

Number of consecutive loops either stopping criterion must be satisfied before the training phase ends.
Default `3`.

### early_stopping_min_acceptance

Global acceptance rate threshold that, once reached, triggers early stopping regardless of stability (after `early_stopping_patience` consecutive loops).
Default `0.1`.
Set to `0` to disable this trigger and rely solely on the stability check.

---

## Parallel tempering parameters

*These parameters are only available in the PT bundles: `RQSpline_MALA_PT_Bundle`, `RQSpline_HMC_PT_Bundle`, `RQSpline_GRW_PT_Bundle`.*

### logprior

Log-prior function with signature `logprior(x, data) -> float`. Default is a flat prior that always returns `0`.
The tempered log-density is

$$\log \tilde{p}_T(x) = \frac{1}{T} \log \mathcal{L}(x) + \log \pi(x)$$

so the prior is preserved at all temperatures.

### n_temperatures

Number of temperature levels, including the target (temperature = 1). Default `5`.
Chains at higher temperatures explore the prior more freely and occasionally swap positions with the target chains, helping escape local modes.

### max_temperature

Highest temperature in the ladder.
Default `5.0`.
The temperatures are spaced linearly between 1 and `max_temperature`.
A larger value allows more prior-like exploration but reduces the swap acceptance rate between adjacent levels.

### n_tempered_steps

Number of parallel-tempering swap attempts per loop.
Default `-1`, which is interpreted as equal to `n_local_steps`.
Reduce this to save computation if swap acceptance is low.

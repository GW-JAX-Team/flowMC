# Architecture

This guide describes the internal architecture of flowMC — how its components are designed, how they interact, and how to extend them for your own use case.

## Anatomy of flowMC

Prior to version 0.4.0, `flowMC` was a package that was designed to execute the algorithm detailed in [this paper](https://arxiv.org/pdf/2105.12603). Since then the community has tried applying `flowMC` to different problems. While there were some successes, there are also limiting factors in terms of performance. One of the biggest issues `flowMC` faced is the fact that the global-local sampling algorithm was baked into the top level `Sampler` API, which means `flowMC` can only use the exact algorithm described in the paper. What if the users want to use a different model? Or run some optimisation steps during the sampling stage? Or apply annealing? These are either impossible or not very intuitive in `flowMC` prior to version 0.4.0.

Seeing this limitation, we redesigned the middle level API of `flowMC` while keeping the top level API as similar as possible.

### Target distribution

The target distribution should be defined as a log-probability density function, which follows the following function signature:

```python
def target_log_prob_fn(x: Array, data: dict) -> Float:
    ...
    return log_prob
```

The `target_log_prob_fn` should take in a 1-D array `x` of length `n_dim` and a dictionary `data` that contains any additional data that the target distribution depends on. The function should return a scalar that is the log-probability density of the target distribution at `x`.

To ensure the target distribution is well-defined and performant, you should also check whether the function is behaving as expected when `jax.jit` and `jax.grad` are applied to it.

### Sampler

On the top level, the `Sampler` class is a thin wrapper on top of the resource-strategy pair (defined below) that provides a couple of extra functionality. The `Sampler` class manages the resources and strategies, as well as run-related parameters such as where would the resources be stored if the user decides to serialise the resources.

```python
nf_sampler = Sampler(
    n_dim,
    n_chains,
    rng_key,
    # you can either supply the resources and strategies directly,
    # which is prioritized over the resource-strategy bundles
    resources=resources,
    strategies=strategies,
    strategy_order=strategy_order,
    # or you can supply the resource-strategy bundles
    resource_strategy_bundles=bundle,
)
```

The main loop of `Sampler` is straightforward after initialisation: given the available resources, it iterates through the list of strategies, each of which takes the resources, performs some actions (such as taking local steps or training a normalising flow), and returns the updated resources. The `Sampler` supports early stopping: if a `State` resource sets an `early_stopped` flag during training, the main loop skips the remaining training strategies and jumps directly to the production phase.

### Resource and Strategy

At the core of the flowMC API are the resource and strategy interfaces. Broadly speaking, resources are similar to data classes and strategies are similar to functions.

**Resources** store some attribute and can be manipulated, but should not have too many methods associated with them. For example, a buffer that stores the sampling results is a resource, a MALA kernel is a resource, and a normalising flow model is a resource. **Strategies** are functions that take in resources and return updated resources. For example, taking a local step requires two kinds of resources: a proposal distribution and the buffer where the samples are stored.

If you are initialising the resources and strategies directly, you can do something like:

```python
resources = {
    "buffer": Buffer(name, n_chains, n_steps, n_dims),
    "proposal": MALA(step_size),
    "model": NormalizingFlow(model_parameters),
}

strategies = {
    "Strategy 1": Strategy1(),
    "Strategy 2": Strategy2(),
}

strategy_order = ["Strategy 1", "Strategy 2", "Strategy 1", ...]
```

This separation allows users to compose different strategies together. For example, a strategy can take in both a MALA kernel and a normalising flow model, and update the MALA step size using information from the flow — without that logic being hard-coded into either component.

A good rule of thumb for deciding whether something should be a resource or a strategy: ask whether the new data structure/functionality is something that should be updated by other strategies. If yes, it should be a resource; if no, it should be a strategy.

One extra criterion is JAX compatibility. Resources should be compatible with `jit`; strategies are not required to be. For example, a training loop that iterates over epochs and logs metadata does not need to be jitted — it should be a strategy. A neural network that runs on GPU during both sampling and training should be a resource.

You can find the hyperparameters of a resource, a strategy, or a resource-strategy bundle in the [API docs](../api/flowMC/Sampler.md).

## Guiding principles

### Write the likelihood function in JAX

If your likelihood is fully defined in [JAX](https://github.com/google/jax), there are a couple of benefits that compound with each other:

1. JAX allows you to access the gradient of the likelihood function through automatic differentiation. This enables gradient-based local samplers such as MALA and HMC, which handle high-dimensional problems more efficiently than gradient-free alternatives like Metropolis-Hastings.
2. JAX uses [XLA](https://www.tensorflow.org/xla) to compile and optimise code for GPUs and TPUs. Multiple MCMC chains help speed up normalising flow training, and accelerators scale this further.

Since version 0.4.0, flowMC requires likelihood functions to be compatible with JAX transformations. flowMC is designed to leverage GPU acceleration and machine learning methods, and rewriting a likelihood in JAX is often worthwhile on its own for the speedup alone.

### Parallelise whenever you can

Resource and strategy design should centre on leveraging parallelisation. This is reflected by `n_chains` being a required parameter for the `Sampler` class — flowMC is designed to solve problems with complex geometry using adaptive sampling methods that benefit tremendously from multiple chains running in parallel.

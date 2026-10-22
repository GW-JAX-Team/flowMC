# flowMC

## A JAX-based normalizing-flow-enhanced MCMC sampler for probabilistic inference

![flowMC_logo](logo.png)

flowMC is a JAX-based package for normalizing-flow-enhanced Markov chain Monte Carlo (MCMC) sampling.
By using normalizing flows as a global proposal, flowMC accelerates convergence for multi-modal and high-dimensional posteriors while running natively on GPU with minimal hyperparameter tuning.

!!! warning
    flowMC has not yet reached v1.0.0 and the API may change.
    Use at your own risk.
    Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install flowMC
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Hands-on worked examples
- **[Guides](guides/index.md)** — One part of flowMC at a time, shown in use
- **[How-to](how-to/index.md)** — Short recipes for specific tasks
- **[Technical Notes](notes/index.md)** — Architecture, the built-in bundles, and every hyperparameter
- **[Development](dev/index.md)** — Internals for contributors
- **[FAQ](FAQ.md)** — Answers to common questions
- **[Citing flowMC](citing.md)** — How to cite flowMC
- **[Contributing to flowMC](contributing.md)** — How to contribute to flowMC

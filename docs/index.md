# flowMC

## A JAX-based normalizing-flow-enhanced MCMC sampler for probabilistic inference

[![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/GW-JAX-Team/flowMC/blob/main/LICENSE) [![coverage](https://img.shields.io/coveralls/github/GW-JAX-Team/flowMC/main)](https://coveralls.io/github/GW-JAX-Team/flowMC?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/flowMC/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/flowMC/main)

![flowMC_logo](logo.png)

flowMC is a JAX-based package for normalizing-flow-enhanced Markov chain Monte Carlo (MCMC) sampling. By using normalizing flows as a global proposal, flowMC accelerates convergence for multi-modal and high-dimensional posteriors while running natively on GPU with minimal hyperparameter tuning.

!!! warning
    flowMC has not yet reached v1.0.0 and the API may change. Use at your own risk. Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install flowMC
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Hands-on worked examples
- **[Guides](guides/index.md)** — In-depth coverage of each module
- **[FAQ](FAQ.md)** — Answers to common questions

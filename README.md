# flowMC

### A JAX-based normalizing-flow-enhanced MCMC sampler for probabilistic inference

[![docs](https://img.shields.io/badge/docs-online-blue)](https://gw-jax-team.github.io/flowMC/) [![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/GW-JAX-Team/flowMC/blob/main/LICENSE) [![coverage](https://img.shields.io/coveralls/github/GW-JAX-Team/flowMC/main)](https://coveralls.io/github/GW-JAX-Team/flowMC?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/flowMC/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/flowMC/main)

![flowMC_logo](https://raw.githubusercontent.com/GW-JAX-Team/flowMC/main/docs/logo.png)

flowMC is a JAX-based package for normalizing-flow-enhanced Markov chain Monte Carlo (MCMC) sampling. By using normalizing flows as a global proposal, flowMC accelerates convergence for multi-modal and high-dimensional posteriors while running natively on GPU with minimal hyperparameter tuning.

For a quick introduction, see the [Quick Start guide](https://gw-jax-team.github.io/flowMC/stable/quickstart/).

> [!WARNING]
> flowMC has not yet reached v1.0.0 and the API may change. Use at your own risk. Consider pinning to a specific version if you need API stability.

## Installation

The simplest way to install flowMC is through pip:

```bash
pip install flowMC
```

This will install the latest stable release and its dependencies.
flowMC is built on [JAX](https://github.com/google/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install flowMC[cuda]
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/flowMC.git
cd flowMC
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## Attribution

If you use flowMC in your research, please cite the following papers:

```bibtex
@article{Wong:2022xvh,
    author = "Wong, Kaze W. k. and Gabri\'e, Marylou and Foreman-Mackey, Daniel",
    title = "{flowMC: Normalizing flow enhanced sampling package for probabilistic inference in JAX}",
    eprint = "2211.06397",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.21105/joss.05021",
    journal = "J. Open Source Softw.",
    volume = "8",
    number = "83",
    pages = "5021",
    year = "2023"
}

@article{Gabrie:2021tlu,
    author = "Gabri\'e, Marylou and Rotskoff, Grant M. and Vanden-Eijnden, Eric",
    title = "{Adaptive Monte Carlo augmented with normalizing flows}",
    eprint = "2105.12603",
    archivePrefix = "arXiv",
    primaryClass = "physics.data-an",
    doi = "10.1073/pnas.2109420119",
    journal = "Proc. Nat. Acad. Sci.",
    volume = "119",
    number = "10",
    pages = "e2109420119",
    year = "2022"
}
```

# flowMC

**Normalizing-flow enhanced sampling package for probabilistic inference**

<a href="https://flowmc.readthedocs.io/en/main/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/GW-JAX-Team/flowMC/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="license"/>
</a>
<a href='https://coveralls.io/github/GW-JAX-Team/flowMC?branch=main'>
<img src='https://badgen.net/coveralls/c/github/GW-JAX-Team/flowMC/main' alt='coverage' />
</a>

> [!WARNING]
> flowMC has not yet reached v1.0.0, and the API may change. Higher-level APIs are more stable, while intermediate-level APIs (such as the resource strategy interface) may undergo major revisions for performance improvements.

![flowMC_logo](./docs/logo_0810.png)

flowMC is a JAX-based Python package for normalizing-flow enhanced Markov chain Monte Carlo (MCMC) sampling. The code is open source under the MIT license and is under active development.

flowMC implements methods described in [Gabrié et al. (2021)](https://openreview.net/pdf?id=mvtooHbjOwx) and [Gabrié et al. (2022)](https://www.pnas.org/doi/10.1073/pnas.2109420119). See the accompanying paper, [Wong, Gabrié, Foreman-Mackey (2023)](https://joss.theoj.org/papers/10.21105/joss.05021), for more details.

- Just-in-time compilation support
- Native GPU acceleration
- Effective for multi-modal problems
- Minimal hyperparameter tuning required

# Installation

The simplest way to install flowMC is through pip:

```
pip install flowMC
```

This will install the latest stable release and its dependencies.
flowMC is built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).
By default, this installs the CPU version of JAX from [PyPI](https://pypi.org).
If you have a GPU and want to leverage hardware acceleration, install the CUDA-enabled version:

```
pip install flowMC[cuda]
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```
git clone https://github.com/GW-JAX-Team/flowMC.git
cd flowMC
pip install -e .
```

Additional optional dependencies are available:
- `flowMC[docs]`: Documentation dependencies
- `flowMC[visualize]`: Visualization dependencies

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.
# Attribution

If you use flowMC in your research, please cite the following papers:

```
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

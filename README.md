# flowMC

**Normalizing-flow enhanced sampling package for probabilistic inference**

<a href="https://flowmc.readthedocs.io/en/main/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/GW-JAX-Team/flowMC/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="license"/>
</a>
<a href='https://coveralls.io/github/GW-JAX-Team/flowMC?branch=main'>
<img src='https://badgen.net/coveralls/c/github/GW-JAX-Team/flowMC/main' alt='Coverage Status' />
</a>

> [!WARNING]
> Note that `flowMC` has not reached v1.0.0, meaning the API could subject to changes. In general, the higher level the API, the less likely it is going to change. However, intermediate level API such as the resource strategy interface could subject to major revision for performance concerns.

![flowMC_logo](./docs/logo_0810.png)

flowMC is a Jax-based python package for normalizing-flow enhanced Markov chain Monte Carlo (MCMC) sampling.
The code is open source under MIT license, and it is under active development.

- Just-in-time compilation is supported.
- Native support for GPU acceleration.
- Suit for problems with multi-modality.
- Minimal tuning.

# Installation 

The simplest way to install the package is to do it through pip

```
pip install flowMC
```

This will install the latest stable release and its dependencies.
flowMC is based on [Jax](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).
By default, installing flowMC will automatically install Jax and Equinox available on [PyPI](https://pypi.org).
By default this install the CPU version of Jax. If you have a GPU and want to use it, you can install the GPU version of Jax by running:

```
pip install flowMC[cuda]
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```
git clone https://github.com/GW-JAX-Team/flowMC.git
cd flowMC
pip install -e .
```

There are a couple more extras that you can install with flowMC, including:
- `flowMC[docs]`: Install the documentation dependencies.
- `flowMC[codeqa]`: Install the code quality dependencies.
- `flowMC[visualize]`: Install the visualization dependencies.

On top of `pip` installation, we highly encourage you to use [uv](https://docs.astral.sh/uv/) to manage your python environment. Once you clone the repo, you can run `uv sync` to create a virtual environment with all the dependencies installed.
# Attribution

If you used `flowMC` in your research, we would really appreciate it if you could at least cite the following papers:

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

This will help `flowMC` getting more recognition, and the main benefit *for you* is this means the `flowMC` community will grow and it will be continuously improved. If you believe in the magic of open-source software, please support us by attributing our software in your work.


`flowMC` is a Jax implementation of methods described in: 
> *Efficient Bayesian Sampling Using Normalizing Flows to Assist Markov Chain Monte Carlo Methods* Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - ICML INNF+ workshop 2021 - [pdf](https://openreview.net/pdf?id=mvtooHbjOwx)

> *Adaptive Monte Carlo augmented with normalizing flows.*
Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - PNAS 2022 - [doi](https://www.pnas.org/doi/10.1073/pnas.2109420119), [arxiv](https://arxiv.org/abs/2105.12603)

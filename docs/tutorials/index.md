# Tutorials

These tutorials walk you through flowMC from first principles to advanced customization. They are ordered from simplest to most involved — if you are new to flowMC, start with the dual moon example.

---

## [A Step-by-Step Example](dualmoon)

The canonical flowMC example. Sample a 5-dimensional multi-modal "dual moon" distribution using the high-level `RQSpline_MALA_Bundle`, which pairs a MALA local kernel with a Masked Coupling RQSpline normalising flow as a learned global proposal. Covers the full workflow: defining a log-probability function, configuring the sampler, running it, and diagnosing convergence with loss curves, acceptance rates, and R-hat.

---

## [Parallel Tempering](parallel_tempering)

Demonstrates why standard local kernels struggle with well-separated modes and how parallel tempering (replica exchange MCMC) overcomes this. Builds a parallel tempering sampler from scratch using the low-level resource-strategy API, then contrasts it with the convenience `RQSpline_MALA_PT_Bundle`.

---

## [Custom Strategy Composition](custom_strategy)

Shows the flexibility of the resource-strategy interface by composing a custom pipeline that prepends an Adam optimization phase to a random-walk MCMC kernel. A useful pattern when chains are initialized far from the high-probability region.

---

## [Training Normalising Flows](train_normalizing_flow)

A standalone introduction to the two normalising flow architectures available in flowMC — **RealNVP** and **Masked Coupling RQSpline** — trained on a 2D toy dataset. Compares their expressiveness and computational cost, and shows how a trained flow plugs into the resource-strategy system via the `TrainModel` strategy.

---

## [Training Flow Matching Models](train_flow_match)

Introduces **flow matching** as an alternative to traditional normalising flows. Trains a Conditional Optimal Transport model on a 2D distribution using an MLP-parameterised velocity field, then generates samples by integrating the learned ODE and evaluates the inferred density on a grid.

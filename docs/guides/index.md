# Guides

Each guide takes one part of flowMC and shows it in use, with examples of its options.

- **[Training Normalising Flows](train_normalizing_flow)** — The two normalising flow architectures in flowMC, RealNVP and Masked Coupling RQSpline, trained on a 2D toy dataset: their expressiveness and cost, and how a trained flow plugs into the resource-strategy system via the `TrainModel` strategy.
- **[Training Flow Matching Models](train_flow_match)** — Flow matching as an alternative to normalising flows: training a Conditional Optimal Transport model on a 2D distribution, sampling by integrating the learned ODE, and evaluating the density on a grid.

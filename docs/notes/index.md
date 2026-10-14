# Technical Notes

These notes describe how flowMC is designed and what each of its built-in options does.

- **[Architecture](architecture.md)** — How flowMC is designed internally: the resource-strategy API, the `Sampler` control loop, and the guiding principles behind the design.
- **[Bundles](bundles.md)** — Reference for the built-in resource-strategy bundles (`RQSpline_MALA`, `RQSpline_HMC`, `RQSpline_GRW`, and their parallel-tempering variants).
  Explains what each bundle does, when to choose it, and the structure of the training and production sampling loops.
- **[Hyperparameter Reference](hyperparameters.md)** — Complete reference for every constructor parameter across all bundles, organised into categories (required, local sampler, normalising flow, training, execution, early stopping, parallel tempering).

# Reduce GPU Memory Use

If a run fails because the GPU is out of memory, shrink how much work runs simultaneously.
flowMC has two independent memory bottlenecks, so there are two settings to lower.
Both are bundle arguments (e.g. to `RQSpline_MALA_Bundle`); see the [Hyperparameter Reference](../notes/hyperparameters.md) for details.

## Lower the NF proposal batch size

This is usually the bottleneck, so try it first.
During each global step the target log-PDF (`logpdf`) is evaluated at all `n_chains * n_global_steps` flow proposals at once.
Reduce `n_NFproposal_batch_size` (default `10000`): when `n_global_steps` exceeds it, the proposals are evaluated in smaller `jax.lax.map` chunks rather than a single `vmap`, lowering peak memory.

## Batch the chains

This matters when your `logpdf` is expensive to evaluate.
Sometimes the problem is not the `n_chains * n_global_steps` proposals, but that even a single `logpdf` evaluation per chain across all `n_chains` chains will not fit.
In that case also set `chain_batch_size` so the local sampler processes chains in sequential sub-batches of that size; smaller values use less memory.
Sub-batching only happens when `chain_batch_size` is greater than 1 and less than `n_chains`; otherwise (including the default `0`) all chains are processed in one `vmap`.

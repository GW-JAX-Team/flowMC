from math import ceil

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from typing import Callable
import equinox as eqx

from flowMC.resource.model.nf_model.base import NFModel
from flowMC.resource.kernel.base import ProposalBase
from flowMC.resource.logPDF import LogPDF


class NFProposal(ProposalBase):
    model: NFModel
    n_batch_size: int

    def __repr__(self):
        return "NF proposal with " + self.model.__repr__()

    def __init__(self, model: NFModel, n_NFproposal_batch_size: int = 100):
        super().__init__()
        self.model = model
        self.n_batch_size = n_NFproposal_batch_size

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[
        Float[Array, "n_step n_dim"], Float[Array, "n_step 1"], Int[Array, "n_step 1"]
    ]:

        print("Compiling NF proposal kernel")
        n_steps = data["n_steps"]

        rng_key, subkey = random.split(rng_key)

        # nf_current is size (1, n_dim)
        log_prob_nf_current = eqx.filter_jit(self.model.log_prob)(position)

        # All these are size (n_steps, n_dim)
        proposed_position, log_prob_nf_proposed = eqx.filter_jit(self.sample_flow)(
            subkey, n_steps
        )
        if n_steps > self.n_batch_size:

            def logpdf_single(pos):
                return logpdf(pos, data)

            log_prob_proposed = jax.lax.map(
                logpdf_single, proposed_position, batch_size=self.n_batch_size
            )
        else:
            log_prob_proposed = jax.vmap(logpdf, in_axes=(0, None))(
                proposed_position, data
            )

        def body(carry, data):
            (
                rng_key,
                position_initial,
                log_prob_initial,
                log_prob_nf_initial,
            ) = carry
            (position_proposal, log_prob_proposal, log_prob_nf_proposal) = data
            rng_key, subkey = random.split(rng_key)
            ratio = (log_prob_proposal - log_prob_initial) - (
                log_prob_nf_proposal - log_prob_nf_initial
            )
            uniform_random = jnp.log(jax.random.uniform(subkey))
            do_accept = uniform_random < ratio
            position_current = jnp.where(do_accept, position_proposal, position_initial)
            log_prob_current = jnp.where(do_accept, log_prob_proposal, log_prob_initial)
            log_prob_nf_current = jnp.where(
                do_accept, log_prob_nf_proposal, log_prob_nf_initial
            )

            return (rng_key, position_current, log_prob_current, log_prob_nf_current), (
                position_current,
                log_prob_current,
                do_accept,
            )

        _, (positions, log_prob, do_accept) = jax.lax.scan(
            body,
            (
                rng_key,
                position,
                log_prob,
                log_prob_nf_current,
            ),
            (proposed_position, log_prob_proposed, log_prob_nf_proposed),
        )

        return positions, log_prob, do_accept

    def sample_flow(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
    ):
        proposal_position = self.model.sample(rng_key, n_steps)

        if n_steps > self.n_batch_size:
            proposed_log_prob = jax.lax.map(
                self.model.log_prob, proposal_position, batch_size=self.n_batch_size
            )
        else:
            proposed_log_prob = eqx.filter_vmap(self.model.log_prob)(proposal_position)

        return proposal_position, proposed_log_prob

    def print_parameters(self):
        # TODO: Implement this
        raise NotImplementedError

    def save_resource(self, path):
        # TODO: Implement this
        raise NotImplementedError

    def load_resource(self, path):
        # TODO: Implement this
        raise NotImplementedError

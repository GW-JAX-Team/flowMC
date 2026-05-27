"""Unit tests for flowMC.Sampler checkpoint/resume behaviour."""

import pickle

import jax
import jax.numpy as jnp
import pytest

from flowMC.Sampler import Sampler
from flowMC.resource.states import State
from flowMC.strategy.base import Strategy


# ── helpers ───────────────────────────────────────────────────────────────────


class _PassthroughStrategy(Strategy):
    """Records call count; returns inputs unchanged."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.n_calls = 0

    def __call__(self, rng_key, resources, initial_position, data):  # noqa: ARG002
        self.n_calls += 1
        return rng_key, resources, initial_position


class _ResetSteppers(Strategy):
    def __init__(self) -> None:
        pass

    def __call__(self, rng_key, resources, initial_position, data):  # noqa: ARG002
        return rng_key, resources, initial_position


def _make_sampler(
    strategy_order, strategies, resources, tmp_path=None, checkpoint_interval=0.0
):
    """Build a Sampler via __new__ with checkpoint attrs explicitly set."""
    s = Sampler.__new__(Sampler)
    s.rng_key = jax.random.key(0)
    s.n_dim = 2
    s.n_chains = 3
    s.resources = resources
    s.strategies = strategies
    s.strategy_order = strategy_order
    s.checkpoint_path = (tmp_path / "ckpt.pkl") if tmp_path is not None else None
    s.checkpoint_interval = checkpoint_interval
    return s


def _write_ckpt(path, meta, strategy_idx=0):
    """Write a minimal synthetic checkpoint file."""
    data = {
        "_meta": meta,
        "strategy_idx": strategy_idx,
        "rng_key": jax.random.key(0),
        "last_step": jnp.zeros((3, 2)),
        "resources": {},
        "stepper_cursors": {},
        "strategy_states": {},
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ── _training_loop_end_indices ───────────────────────────────────────────────


class TestTrainingPhaseEndIndices:
    def _s(self, order):
        s = Sampler.__new__(Sampler)
        s.strategy_order = order
        return s

    def test_no_reset_sentinel_returns_empty(self):
        assert self._s(["a", "b", "c"])._training_loop_end_indices() == set()

    def test_empty_training_section_returns_empty(self):
        assert self._s(["reset_steppers", "a"])._training_loop_end_indices() == set()

    def test_single_loop_no_repeating_strategy(self):
        # [a, b] → no repeat found; fallback treats whole section as one block → {1}
        result = self._s(["a", "b", "reset_steppers", "c"])._training_loop_end_indices()
        assert result == {1}

    def test_two_loops_repeating_block(self):
        # [a, b, a, b] → period=2, preamble=0 → indices {1, 3}
        result = self._s(
            ["a", "b", "a", "b", "reset_steppers", "c"]
        )._training_loop_end_indices()
        assert result == {1, 3}

    def test_three_loops_repeating_block(self):
        # [a, b, a, b, a, b] → period=2, preamble=0 → indices {1, 3, 5}
        result = self._s(
            ["a", "b", "a", "b", "a", "b", "reset_steppers"]
        )._training_loop_end_indices()
        assert result == {1, 3, 5}

    def test_pt_preamble_one_time_strategy(self):
        # [init_pt, a, b, a, b] → preamble=1 (init_pt never repeats), period=2 → indices {2, 4}
        result = self._s(
            ["init_pt", "a", "b", "a", "b", "reset_steppers", "c"]
        )._training_loop_end_indices()
        assert result == {2, 4}


# ── checkpoint write ──────────────────────────────────────────────────────────


class TestCheckpointWrite:
    def test_checkpoint_file_created(self, tmp_path):
        """Checkpoint file must be written after the training loop completes."""
        strategies = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
        }
        s = _make_sampler(
            ["step_a", "step_b", "reset_steppers"],
            strategies,
            {},
            tmp_path=tmp_path,
        )
        s.sample(jnp.zeros((3, 2)), {})
        assert (tmp_path / "ckpt.pkl").exists()

    def test_checkpoint_content_is_valid(self, tmp_path):
        """The saved checkpoint must include _meta and all required keys."""
        state = State({"value": 42}, name="some_state")
        strategies = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
        }
        s = _make_sampler(
            ["step_a", "step_b", "reset_steppers"],
            strategies,
            {"some_state": state},
            tmp_path=tmp_path,
        )
        s.sample(jnp.zeros((3, 2)), {})

        with open(tmp_path / "ckpt.pkl", "rb") as f:
            ckpt = pickle.load(f)

        assert ckpt["_meta"]["n_dim"] == 2
        assert ckpt["_meta"]["n_chains"] == 3
        assert ckpt["_meta"]["strategy_order"] == ["step_a", "step_b", "reset_steppers"]
        # step_b is at index 1, the last strategy before reset_steppers
        assert ckpt["strategy_idx"] == 1
        assert "rng_key" in ckpt
        assert "last_step" in ckpt
        assert "resources" in ckpt
        assert "stepper_cursors" in ckpt
        assert "strategy_states" in ckpt
        assert "skip_to_production" not in ckpt  # derived from State on restore

    def test_no_checkpoint_without_path(self, tmp_path):
        """No file is written when checkpoint_path is None."""
        strategies = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
        }
        s = _make_sampler(
            ["step_a", "step_b", "reset_steppers"],
            strategies,
            {},
            tmp_path=None,  # no checkpoint
        )
        s.sample(jnp.zeros((3, 2)), {})
        # Nothing should have been written anywhere in tmp_path
        assert not list(tmp_path.iterdir())


# ── checkpoint resume ─────────────────────────────────────────────────────────


class TestCheckpointResume:
    """Verify that a resumed run skips already-completed strategies."""

    _order = ["step_a", "step_b", "reset_steppers", "step_prod"]

    def _first_run(self, tmp_path):
        strategies = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
            "step_prod": _PassthroughStrategy("step_prod"),
        }
        s = _make_sampler(self._order, strategies, {}, tmp_path=tmp_path)
        s.sample(jnp.zeros((3, 2)), {})
        return strategies

    def test_completed_strategies_not_re_run(self, tmp_path):
        """step_a and step_b must not run again after they were checkpointed."""
        self._first_run(tmp_path)

        resume = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
            "step_prod": _PassthroughStrategy("step_prod"),
        }
        s2 = _make_sampler(self._order, resume, {}, tmp_path=tmp_path)
        s2.sample(jnp.zeros((3, 2)), {})

        # Checkpoint was at idx=1 (step_b) → start_idx=2 → step_a and step_b skipped.
        assert resume["step_a"].n_calls == 0
        assert resume["step_b"].n_calls == 0

    def test_post_checkpoint_strategies_still_run(self, tmp_path):
        """reset_steppers and step_prod must still execute on resume."""
        self._first_run(tmp_path)

        resume = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
            "step_prod": _PassthroughStrategy("step_prod"),
        }
        s2 = _make_sampler(self._order, resume, {}, tmp_path=tmp_path)
        s2.sample(jnp.zeros((3, 2)), {})

        assert resume["step_prod"].n_calls == 1


# ── checkpoint validation ─────────────────────────────────────────────────────


_VALID_META = {
    "n_dim": 2,
    "n_chains": 3,
    "strategy_order": ["a", "reset_steppers"],
    "logpdf_fingerprint": None,
}
_VALIDATION_ORDER = ["a", "reset_steppers"]


def _validation_strategies():
    return {"a": _PassthroughStrategy("a"), "reset_steppers": _ResetSteppers()}


class TestCheckpointValidation:
    def test_n_dim_mismatch_raises(self, tmp_path):
        _write_ckpt(tmp_path / "ckpt.pkl", {**_VALID_META, "n_dim": 99})
        s = _make_sampler(
            _VALIDATION_ORDER, _validation_strategies(), {}, tmp_path=tmp_path
        )
        with pytest.raises(ValueError, match="n_dim"):
            s.sample(jnp.zeros((3, 2)), {})

    def test_n_chains_mismatch_raises(self, tmp_path):
        _write_ckpt(tmp_path / "ckpt.pkl", {**_VALID_META, "n_chains": 99})
        s = _make_sampler(
            _VALIDATION_ORDER, _validation_strategies(), {}, tmp_path=tmp_path
        )
        with pytest.raises(ValueError, match="n_chains"):
            s.sample(jnp.zeros((3, 2)), {})

    def test_strategy_order_mismatch_raises(self, tmp_path):
        _write_ckpt(
            tmp_path / "ckpt.pkl",
            {**_VALID_META, "strategy_order": ["x", "reset_steppers"]},
        )
        s = _make_sampler(
            _VALIDATION_ORDER, _validation_strategies(), {}, tmp_path=tmp_path
        )
        with pytest.raises(ValueError, match="strategy_order"):
            s.sample(jnp.zeros((3, 2)), {})

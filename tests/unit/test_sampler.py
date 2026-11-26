"""Unit tests for flowMC.Sampler checkpoint/resume behaviour."""

import pickle
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.compilation_cache import compilation_cache

from flowMC.resource.states import State
from flowMC.Sampler import Sampler
from flowMC.strategy.base import Strategy

# ── helpers ───────────────────────────────────────────────────────────────────


class _PassthroughStrategy(Strategy):
    """Records call count; returns inputs unchanged."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.n_calls = 0

    def __call__(self, rng_key, resources, initial_position, data):
        self.n_calls += 1
        return rng_key, resources, initial_position


class _ResetSteppers(Strategy):
    def __init__(self) -> None:
        pass

    def __call__(self, rng_key, resources, initial_position, data):
        return rng_key, resources, initial_position


def _make_sampler(
    strategy_order, strategies, resources, tmp_path=None, checkpoint_interval=1e-9
):
    """Build a Sampler via __new__ with checkpoint attrs explicitly set."""
    s = Sampler.__new__(Sampler)
    s.rng_key = jax.random.key(0)
    s.n_dim = 2
    s.n_chains = 3
    s.resources = resources
    s.strategies = strategies
    s.strategy_order = strategy_order
    s.outdir = str(tmp_path) if tmp_path is not None else "./outdir/"
    # Use interval=0 to disable checkpointing when no tmp_path is provided.
    s.checkpoint_interval = checkpoint_interval if tmp_path is not None else 0.0
    s._prev_elapsed = 0.0
    return s


def _write_ckpt(path, meta, strategy_idx=0):
    """Write a minimal synthetic checkpoint file."""
    data = {
        "_meta": meta,
        "strategy_idx": strategy_idx,
        "rng_key": jax.random.key(0),
        "last_step": jnp.zeros((3, 2)),
        "elapsed_time": 0.0,
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


# ── JAX compilation cache ─────────────────────────────────────────────────────


class TestJaxCache:
    @pytest.fixture(autouse=True)
    def _restore_cache_setting(self):
        original = jax.config.jax_compilation_cache_dir
        yield
        jax.config.update("jax_compilation_cache_dir", original)
        compilation_cache.reset_cache()

    def _sampler(self, tmp_path):
        strategies = {"a": _PassthroughStrategy("a")}
        return Sampler(
            n_dim=2,
            n_chains=3,
            rng_key=jax.random.key(0),
            resources={},
            strategies=strategies,
            strategy_order=["a"],
            outdir=str(tmp_path),
        )

    def test_cache_is_released_after_sample(self, tmp_path, recwarn):
        jax.config.update("jax_compilation_cache_dir", None)
        sampler = self._sampler(tmp_path)
        jax.jit(lambda x: jnp.sin(x) * 2)(jnp.ones(3)).block_until_ready()
        sampler.sample(jnp.zeros((3, 2)), {})

        assert jax.config.jax_compilation_cache_dir is None
        assert not (tmp_path / "jax_cache").exists()
        jax.jit(lambda x: jnp.cos(x) + 3)(jnp.ones(3)).block_until_ready()
        assert not [w for w in recwarn if "compilation cache" in str(w.message)]

    def test_other_cache_setting_is_kept(self, tmp_path):
        other = str(tmp_path / "other_cache")
        jax.config.update("jax_compilation_cache_dir", other)
        sampler = self._sampler(tmp_path / "run")
        sampler.sample(jnp.zeros((3, 2)), {})
        assert jax.config.jax_compilation_cache_dir == other


# ── checkpoint write ──────────────────────────────────────────────────────────


class TestCheckpointWrite:
    def test_checkpoint_file_created(self, tmp_path, monkeypatch):
        """Checkpoint is written during sampling and deleted on clean completion."""
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
        # Suppress deletion of only the checkpoint file so we can inspect it.
        ckpt_path = tmp_path / "checkpoint.pkl"
        _orig = Path.unlink
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=False: (
                None if self == ckpt_path else _orig(self, missing_ok=missing_ok)
            ),
        )
        s.sample(jnp.zeros((3, 2)), {})
        monkeypatch.setattr(Path, "unlink", _orig)
        assert ckpt_path.exists()

    def test_checkpoint_content_is_valid(self, tmp_path, monkeypatch):
        """The saved checkpoint must include _meta, elapsed_time, and all required keys."""
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
        ckpt_path = tmp_path / "checkpoint.pkl"
        _orig = Path.unlink
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=False: (
                None if self == ckpt_path else _orig(self, missing_ok=missing_ok)
            ),
        )
        s.sample(jnp.zeros((3, 2)), {})
        monkeypatch.setattr(Path, "unlink", _orig)

        with open(tmp_path / "checkpoint.pkl", "rb") as f:
            ckpt = pickle.load(f)

        assert ckpt["_meta"]["n_dim"] == 2
        assert ckpt["_meta"]["n_chains"] == 3
        assert ckpt["_meta"]["strategy_order"] == ["step_a", "step_b", "reset_steppers"]
        # step_b is at index 1, the last strategy before reset_steppers
        assert ckpt["strategy_idx"] == 1
        assert "elapsed_time" in ckpt
        assert ckpt["elapsed_time"] >= 0.0
        assert "rng_key" in ckpt
        assert "last_step" in ckpt
        assert "resources" in ckpt
        assert "stepper_cursors" in ckpt
        assert "strategy_states" in ckpt
        assert "skip_to_production" not in ckpt  # derived from State on restore

    def test_no_checkpoint_when_interval_is_zero(self, tmp_path):
        """No file is written when checkpoint_interval=0 (disabled)."""
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
            checkpoint_interval=0.0,
        )
        s.sample(jnp.zeros((3, 2)), {})
        assert not (tmp_path / "checkpoint.pkl").exists()


# ── checkpoint resume ─────────────────────────────────────────────────────────


class TestCheckpointResume:
    """Verify that a resumed run skips already-completed strategies."""

    _order: ClassVar[list[str]] = ["step_a", "step_b", "reset_steppers", "step_prod"]

    def _first_run(self, tmp_path, monkeypatch):
        """Run to completion while preserving the checkpoint (simulates a crash)."""
        strategies = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
            "step_prod": _PassthroughStrategy("step_prod"),
        }
        s = _make_sampler(self._order, strategies, {}, tmp_path=tmp_path)
        ckpt_path = tmp_path / "checkpoint.pkl"
        _orig = Path.unlink
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=False: (
                None if self == ckpt_path else _orig(self, missing_ok=missing_ok)
            ),
        )
        s.sample(jnp.zeros((3, 2)), {})
        monkeypatch.setattr(Path, "unlink", _orig)
        return strategies

    def test_completed_strategies_not_re_run(self, tmp_path, monkeypatch):
        """step_a and step_b must not run again after they were checkpointed."""
        self._first_run(tmp_path, monkeypatch)

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

    def test_post_checkpoint_strategies_still_run(self, tmp_path, monkeypatch):
        """reset_steppers and step_prod must still execute on resume."""
        self._first_run(tmp_path, monkeypatch)

        resume = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
            "step_prod": _PassthroughStrategy("step_prod"),
        }
        s2 = _make_sampler(self._order, resume, {}, tmp_path=tmp_path)
        s2.sample(jnp.zeros((3, 2)), {})

        assert resume["step_prod"].n_calls == 1

    def test_resume_restores_rng_key(self, tmp_path, monkeypatch):
        """RNG key restored from checkpoint must equal the key at that point in a fresh run."""
        # Run 1: record rng_key after step_b (last training strategy before reset)
        rng_key_after_training = {}

        class _RecordingStrategy(Strategy):
            def __init__(self):
                pass

            def __call__(self, rng_key, resources, position, data):
                rng_key_after_training["val"] = rng_key
                return rng_key, resources, position

        strategies1 = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _RecordingStrategy(),
            "reset_steppers": _ResetSteppers(),
            "step_prod": _PassthroughStrategy("step_prod"),
        }
        s1 = _make_sampler(self._order, strategies1, {}, tmp_path=tmp_path)
        ckpt_path = tmp_path / "checkpoint.pkl"
        _orig = Path.unlink
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=False: (
                None if self == ckpt_path else _orig(self, missing_ok=missing_ok)
            ),
        )
        s1.sample(jnp.zeros((3, 2)), {})
        monkeypatch.setattr(Path, "unlink", _orig)

        # Run 2: resume from checkpoint — step_a and step_b are skipped, so key
        # must be restored from the checkpoint, not re-computed
        rng_key_at_prod = {}

        class _ProdRecorder(Strategy):
            def __init__(self):
                pass

            def __call__(self, rng_key, resources, position, data):
                rng_key_at_prod["val"] = rng_key
                return rng_key, resources, position

        strategies2 = {
            "step_a": _PassthroughStrategy("step_a"),
            "step_b": _PassthroughStrategy("step_b"),
            "reset_steppers": _ResetSteppers(),
            "step_prod": _ProdRecorder(),
        }
        s2 = _make_sampler(self._order, strategies2, {}, tmp_path=tmp_path)
        s2.sample(jnp.zeros((3, 2)), {})

        # The RNG key passed to step_prod in run 2 must equal what reset_steppers
        # would receive if run1 continued from the checkpoint key
        assert "val" in rng_key_at_prod  # step_prod ran
        assert jnp.array_equal(s2.rng_key, s1.rng_key)  # final rng_key must match


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
        _write_ckpt(tmp_path / "checkpoint.pkl", {**_VALID_META, "n_dim": 99})
        s = _make_sampler(
            _VALIDATION_ORDER, _validation_strategies(), {}, tmp_path=tmp_path
        )
        with pytest.raises(ValueError, match="n_dim"):
            s.sample(jnp.zeros((3, 2)), {})

    def test_n_chains_mismatch_raises(self, tmp_path):
        _write_ckpt(tmp_path / "checkpoint.pkl", {**_VALID_META, "n_chains": 99})
        s = _make_sampler(
            _VALIDATION_ORDER, _validation_strategies(), {}, tmp_path=tmp_path
        )
        with pytest.raises(ValueError, match="n_chains"):
            s.sample(jnp.zeros((3, 2)), {})

    def test_strategy_order_mismatch_raises(self, tmp_path):
        _write_ckpt(
            tmp_path / "checkpoint.pkl",
            {**_VALID_META, "strategy_order": ["x", "reset_steppers"]},
        )
        s = _make_sampler(
            _VALIDATION_ORDER, _validation_strategies(), {}, tmp_path=tmp_path
        )
        with pytest.raises(ValueError, match="strategy_order"):
            s.sample(jnp.zeros((3, 2)), {})

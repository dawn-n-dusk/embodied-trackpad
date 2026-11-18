"""Focused tests for raw eval traces, exact episode seeds, and the
nonfinite-gradient guard (v002 behavior)."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

try:
    import pytest
except ModuleNotFoundError:
    class _PytestStub:
        @staticmethod
        def importorskip(mod):
            try:
                return __import__(mod)
            except ModuleNotFoundError:
                raise SystemExit(f"skipped: {mod} not installed")

        class fixture:
            def __init__(self, *a, **k):
                pass

            def __call__(self, fn):
                return fn

        class raises:
            def __init__(self, exc):
                self.exc = exc

            def __enter__(self):
                return self

            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(f"expected {self.exc}")
                return issubclass(et, self.exc)

    pytest = _PytestStub()

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_sharpa_supermimic import make_dataset  # noqa: E402

import embodied_trackpad.sharpa_learning.supermimic as sm  # noqa: E402
from embodied_trackpad.sharpa_learning.supermimic import (  # noqa: E402
    train_supermimic,
)

ACTION_DIM = 29
PROPRIO_DIM = 58
TOUCH_DIM = 5
T_STEPS = 5


class StubEnv:
    """Minimal env with the real observation contract and recorded seeds."""

    def __init__(self, config=None):
        self.seeds = []
        self.t = 0

    def _obs(self):
        return {
            "screen_rgb": np.full((128, 128, 3), self.t % 251, dtype=np.uint8),
            "wrist_rgb": np.full((128, 128, 3), (255 - self.t) % 251,
                                 dtype=np.uint8),
            "proprio": np.tile(np.array([self.t, 0.25], dtype=np.float32),
                               PROPRIO_DIM // 2),
            "touch": np.zeros(TOUCH_DIM, dtype=np.float32) + self.t * 0.1,
            "touch_valid": np.ones(TOUCH_DIM, dtype=np.uint8),
        }

    def _info(self):
        return {
            "success": self.t >= T_STEPS,
            "executed_q_ref": np.full(ACTION_DIM, 0.01, dtype=np.float32),
            "cursor": np.array([self.t * 1.0, 2.0], dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        self.seeds.append(seed)
        self.t = 0
        return self._obs(), self._info()

    def step(self, action):
        self.t += 1
        terminated = self.t >= T_STEPS
        return self._obs(), 1.0, terminated, False, self._info()

    def close(self):
        pass


def _train_tiny_ckpt(dataset_path, tmp_path):
    out = str(tmp_path / "trace_ckpt")
    train_supermimic(_ns_train(dataset_path, out))
    return os.path.join(out, "checkpoint_final.pt")


def _ns_train(dataset_path, out):
    from argparse import Namespace
    return Namespace(dataset=dataset_path, output=out, wm_warmup_steps=2,
                     actor_updates=2, imagine_horizon=4, burn_in_steps=1,
                     seed=0, device="cpu", time_budget=0.0, resume_from=None,
                     finetune_from=None, small_config=True)


def test_eval_raw_trace_and_exact_seeds(tmp_path):
    root = tempfile.mkdtemp(prefix="sharpa_sm_trace_")
    try:
        ds = make_dataset(root, seed=7)
        ckpt = _train_tiny_ckpt(ds, tmp_path)
        out = str(tmp_path / "trace_out")
        res = sm.evaluate_supermimic(_ns_eval(
            ckpt, out, episodes=3, save_trace=2, seed_base=90000000))
        assert res["episodes"] == 3
        # Exact native protocol seeds: 90000000, 90000001, 90000002.
        assert [e["seed"] for e in res["episode_records"]] == [
            90000000, 90000001, 90000002]
        traces = sorted(glob.glob(os.path.join(out, "traces",
                                               "episode_*_raw.npz")))
        assert len(traces) == 2
        with np.load(traces[0], allow_pickle=False) as z:
            screen = z["screen_rgb"]
            assert screen.shape[1:] == (128, 128, 3)
            assert int(z["_seed"][0]) == 90000000
            assert not np.array_equal(screen[0], screen[1])
        jl = traces[0].replace("_raw.npz", ".jsonl")
        recs = [json.loads(l) for l in open(jl)]
        assert recs[0]["terminated"] is False
        assert "action" in recs[0] and "obs_small" in recs[0]
        assert "cursor" in recs[0]["info_small"]
        assert "executed_q_ref" in recs[0]["info_small"]
        assert all("max_step_reached" in r for r in recs)
        # Both view videos encoded for the first traced episode.
        for view in ("screen_rgb", "wrist_rgb"):
            mp4 = traces[0].replace("_raw.npz", f"_{view}.mp4")
            assert os.path.getsize(mp4) > 0, mp4
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_eval_max_steps_marks_truncated(tmp_path):
    root = tempfile.mkdtemp(prefix="sharpa_sm_trunc_")
    try:
        ds = make_dataset(root, seed=7)
        ckpt = _train_tiny_ckpt(ds, tmp_path)
        out = str(tmp_path / "trunc_out")
        res = sm.evaluate_supermimic(_ns_eval(
            ckpt, out, episodes=1, save_trace=0, seed_base=None,
            max_steps=2))
        rec = res["episode_records"][0]
        assert rec["steps"] == 2 and rec["truncated"] is True
        assert rec["terminated"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _ns_eval(ckpt, out, episodes, save_trace, seed_base, max_steps=10):
    from argparse import Namespace
    return Namespace(checkpoint=ckpt,
                     env_factory="test_supermimic_eval_trace:StubEnv",
                     dataset=None, openloop_split="test", episodes=episodes,
                     max_steps=max_steps, seed=0, task=None, env_config=None,
                     success_key="success", output=out, device="cpu",
                     save_trace=save_trace, episode_seed_base=seed_base)


def _write_minimal_trace(trace_dir, ep_i):
    import numpy as _np
    raw = {"screen_rgb": [_np.zeros((8, 8, 3), dtype=_np.uint8)],
           "wrist_rgb": [_np.zeros((8, 8, 3), dtype=_np.uint8)]}
    sm._write_trace(trace_dir, ep_i, [], raw, True, 123)


def test_write_trace_npz_is_compressed(tmp_path):
    trace_dir = str(tmp_path / "traces")
    _write_minimal_trace(trace_dir, 0)
    npz = os.path.join(trace_dir, "episode_000_raw.npz")
    import zipfile
    with zipfile.ZipFile(npz) as zf:
        assert zf.namelist(), "npz must contain entries"
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED, (
                f"{info.filename} not deflate-compressed")


def test_write_trace_refuses_existing_outputs(tmp_path):
    trace_dir = str(tmp_path / "traces")
    _write_minimal_trace(trace_dir, 0)
    with pytest.raises(FileExistsError):
        _write_minimal_trace(trace_dir, 0)


def test_mp4_encoded_ok_boundary(tmp_path):
    empty = str(tmp_path / "empty.mp4")
    open(empty, "wb").close()
    assert sm._mp4_encoded_ok(empty) is False
    garbage = str(tmp_path / "garbage.mp4")
    with open(garbage, "wb") as fh:
        fh.write(b"not a real mp4 payload" * 64)
    assert sm._mp4_encoded_ok(garbage) is False
    try:
        import imageio  # noqa: F401
    except ModuleNotFoundError:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            return  # no encoder available; negative cases above still hold
    good_dir = str(tmp_path / "good")
    _write_minimal_trace(good_dir, 0)
    mp4 = os.path.join(good_dir, "episode_000_screen_rgb.mp4")
    if os.path.isfile(mp4):
        assert sm._mp4_encoded_ok(mp4) is True
        meta = json.load(open(os.path.join(good_dir,
                                           "episode_000_video_meta.json")))
        assert meta.get("screen_rgb") == "mp4"
    else:
        meta = json.load(open(os.path.join(good_dir,
                                           "episode_000_video_meta.json")))
        assert meta["screen_rgb"].startswith("unavailable")


def test_nonfinite_grad_guard_preserves_checkpoint(tmp_path):
    """Nonfinite grads before step abort with the last good checkpoint kept."""
    root = tempfile.mkdtemp(prefix="sharpa_sm_nfg_")
    try:
        ds = make_dataset(root, seed=7)
        orig = sm._grads_finite
        sm._grads_finite = lambda model: False
        try:
            metrics = train_supermimic(_ns_nfg(ds, tmp_path))
        finally:
            sm._grads_finite = orig
        assert metrics["stopped_reason"] == "nonfinite_grad"
        assert metrics["wm_updates"] == 0
        assert metrics["nonfinite_grad_events"] == 1
        assert os.path.isfile(os.path.join(str(tmp_path / "nfg"),
                                           "checkpoint_latest.pt"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _ns_nfg(ds, tmp_path):
    from argparse import Namespace
    return Namespace(dataset=ds, output=str(tmp_path / "nfg"),
                     wm_warmup_steps=3, actor_updates=3, imagine_horizon=4,
                     burn_in_steps=1, seed=0, device="cpu", time_budget=0.0,
                     resume_from=None, finetune_from=None, small_config=True)


def _run_all() -> int:
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    import pathlib
    for name, fn in tests:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="sharpa_sm_v2_tmp_"))
        try:
            fn(tmp)
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())

"""Focused tests for explicit new-dataset finetune (--finetune-from).

Finetune restores model+optimizer states and cumulative counters from a
parent checkpoint but rebuilds batchers on the new dataset. It must never
weaken strict same-dataset --resume-from behavior.
"""

from __future__ import annotations

import hashlib
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

from embodied_trackpad.sharpa_learning.supermimic import (  # noqa: E402
    train_supermimic,
    load_supermimic_checkpoint,
)


def _ns(**kw):
    from argparse import Namespace
    base = dict(dataset="", output="", wm_warmup_steps=2, actor_updates=2,
                imagine_horizon=4, burn_in_steps=1, seed=0, device="cpu",
                time_budget=0.0, resume_from=None, finetune_from=None,
                small_config=True)
    base.update(kw)
    return Namespace(**base)


def _parent_run(tmp_path, dataset_path, wm_steps=2, actor_steps=2):
    out = str(tmp_path / "parent")
    metrics = train_supermimic(_ns(
        dataset=dataset_path, output=out, wm_warmup_steps=wm_steps,
        actor_updates=actor_steps, small_config=True))
    return out, metrics


def _adamw_step_counts(opt_state):
    return {int(v["step"]) for v in opt_state["state"].values()}


def test_finetune_restores_weights_optimizers_counters(tmp_path):
    """Finetune keeps parent weights/optimizers/counters and continues them."""
    root_a = tempfile.mkdtemp(prefix="sharpa_sm_ft_a_")
    root_b = tempfile.mkdtemp(prefix="sharpa_sm_ft_b_")
    try:
        ds_a = make_dataset(root_a, seed=7)
        ds_b = make_dataset(root_b, seed=12345)
        out_p, m_p = _parent_run(tmp_path, ds_a)
        assert m_p["wm_updates"] == 2 and m_p["actor_updates"] == 2
        ckpt = os.path.join(out_p, "checkpoint_final.pt")

        out_f = str(tmp_path / "finetune")
        m_f = train_supermimic(_ns(
            dataset=ds_b, output=out_f, wm_warmup_steps=3, actor_updates=3,
            small_config=False, finetune_from=ckpt))
        # Counters continue from parent (2) instead of resetting.
        assert m_f["wm_updates"] == 3
        assert m_f["actor_updates"] == 3
        assert m_f["critic_updates"] == 3

        wm_p, actor_p, critic_p, _cfg_p, _ = load_supermimic_checkpoint(ckpt)
        wm_f, actor_f, critic_f, _cfg_f, payload_f = load_supermimic_checkpoint(
            os.path.join(out_f, "checkpoint_final.pt"))
        # Training actually continued on the new dataset: weights moved.
        wm_sd_p, wm_sd_f = wm_p.state_dict(), wm_f.state_dict()
        assert any(not torch.allclose(wm_sd_p[k].float(), wm_sd_f[k].float())
                   for k in wm_sd_p)
        # AdamW steps prove the optimizer state was restored, not rebuilt:
        # parent took 2 WM / 2 actor / 2 critic steps, finetune adds 1 each.
        assert _adamw_step_counts(payload_f["optimizer_state"]["wm"]) == {3}
        assert _adamw_step_counts(payload_f["optimizer_state"]["actor"]) == {3}
        assert _adamw_step_counts(payload_f["optimizer_state"]["critic"]) == {3}
        for name, mp, mf in (("actor", actor_p, actor_f), ("critic", critic_p, critic_f)):
            assert any(not torch.allclose(a.float(), b.float())
                       for a, b in zip(mp.parameters(), mf.parameters())), name
    finally:
        shutil.rmtree(root_a, ignore_errors=True)
        shutil.rmtree(root_b, ignore_errors=True)


def test_finetune_records_lineage_and_parent_sha(tmp_path):
    root_a = tempfile.mkdtemp(prefix="sharpa_sm_ft2_a_")
    root_b = tempfile.mkdtemp(prefix="sharpa_sm_ft2_b_")
    try:
        ds_a = make_dataset(root_a, seed=7)
        ds_b = make_dataset(root_b, seed=12345)
        out_p, _ = _parent_run(tmp_path, ds_a)
        ckpt = os.path.join(out_p, "checkpoint_final.pt")
        parent_payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        parent_run_id = parent_payload["provenance"]["run_id"]
        parent_ds_sha = parent_payload["provenance"]["dataset_sha256"]
        parent_ckpt_sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()

        out_f = str(tmp_path / "finetune2")
        m_f = train_supermimic(_ns(
            dataset=ds_b, output=out_f, wm_warmup_steps=3, actor_updates=3,
            small_config=False, finetune_from=ckpt))
        payload_f = torch.load(os.path.join(out_f, "checkpoint_final.pt"),
                               map_location="cpu", weights_only=False)
        prov = payload_f["provenance"]
        assert prov["finetune"] is True
        assert prov["resume_kind"] == "finetune_new_dataset"
        assert prov["parent_checkpoint_sha256"] == parent_ckpt_sha
        assert prov["parent_run_id"] == parent_run_id
        assert prov["parent_dataset_sha256"] == parent_ds_sha
        assert prov["dataset_sha256"] != parent_ds_sha
        assert prov["run_id"] == m_f["run_id"]
        assert prov["run_id"] != parent_run_id
        assert prov["lineage"]["parent_run_id"] == parent_run_id
        assert prov["lineage"]["child_run_id"] == prov["run_id"]
        assert prov["finetune_from_counters"] == {
            "wm_updates": 2, "actor_updates": 2, "critic_updates": 2}
        assert m_f["finetune_parent_checkpoint_sha256"] == parent_ckpt_sha
    finally:
        shutil.rmtree(root_a, ignore_errors=True)
        shutil.rmtree(root_b, ignore_errors=True)


def test_finetune_rejects_resume_combo_and_missing(tmp_path):
    root_a = tempfile.mkdtemp(prefix="sharpa_sm_ft3_a_")
    root_b = tempfile.mkdtemp(prefix="sharpa_sm_ft3_b_")
    try:
        ds_a = make_dataset(root_a, seed=7)
        ds_b = make_dataset(root_b, seed=12345)
        out_p, _ = _parent_run(tmp_path, ds_a)
        ckpt = os.path.join(out_p, "checkpoint_final.pt")
        with pytest.raises(ValueError):
            train_supermimic(_ns(dataset=ds_b, output=str(tmp_path / "x"),
                                 finetune_from=ckpt, resume_from=ckpt,
                                 small_config=False))
        with pytest.raises(FileNotFoundError):
            train_supermimic(_ns(dataset=ds_b, output=str(tmp_path / "y"),
                                 finetune_from=str(tmp_path / "nope.pt"),
                                 small_config=False))
    finally:
        shutil.rmtree(root_a, ignore_errors=True)
        shutil.rmtree(root_b, ignore_errors=True)


def test_finetune_strict_resume_still_rejects_new_dataset(tmp_path):
    """--resume-from must keep rejecting a different-dataset checkpoint."""
    root_a = tempfile.mkdtemp(prefix="sharpa_sm_ft4_a_")
    root_b = tempfile.mkdtemp(prefix="sharpa_sm_ft4_b_")
    try:
        ds_a = make_dataset(root_a, seed=7)
        ds_b = make_dataset(root_b, seed=12345)
        out_p, _ = _parent_run(tmp_path, ds_a)
        ckpt = os.path.join(out_p, "checkpoint_final.pt")
        with pytest.raises(ValueError):
            train_supermimic(_ns(dataset=ds_b, output=str(tmp_path / "bad"),
                                 resume_from=ckpt, small_config=True))
    finally:
        shutil.rmtree(root_a, ignore_errors=True)
        shutil.rmtree(root_b, ignore_errors=True)


def test_finetune_batchers_fresh_and_deterministic(tmp_path):
    """Two identical finetune runs match exactly: fresh deterministic batchers."""
    root_a = tempfile.mkdtemp(prefix="sharpa_sm_ft5_a_")
    root_b = tempfile.mkdtemp(prefix="sharpa_sm_ft5_b_")
    try:
        ds_a = make_dataset(root_a, seed=7)
        ds_b = make_dataset(root_b, seed=12345)
        out_p, _ = _parent_run(tmp_path, ds_a)
        ckpt = os.path.join(out_p, "checkpoint_final.pt")

        outs = []
        for name in ("run_a", "run_b"):
            out_f = str(tmp_path / name)
            train_supermimic(_ns(dataset=ds_b, output=out_f, wm_warmup_steps=3,
                                 actor_updates=3, small_config=False,
                                 finetune_from=ckpt))
            outs.append(out_f)
        models_a = load_supermimic_checkpoint(os.path.join(outs[0], "checkpoint_final.pt"))
        models_b = load_supermimic_checkpoint(os.path.join(outs[1], "checkpoint_final.pt"))
        for ma, mb in zip(models_a[:3], models_b[:3]):
            for pa, pb in zip(ma.parameters(), mb.parameters()):
                assert torch.allclose(pa.float(), pb.float())
        # Parent small-config dimensions are retained in the new run config.
        assert models_a[3].wm_config.gru_dim == 32
    finally:
        shutil.rmtree(root_a, ignore_errors=True)
        shutil.rmtree(root_b, ignore_errors=True)


def _run_all() -> int:
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    import pathlib
    for name, fn in tests:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="sharpa_sm_ft_tmp_"))
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

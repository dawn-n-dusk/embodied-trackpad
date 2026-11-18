"""Focused regression test: evaluate_supermimic writes closed_loop_eval.json before env.close().

Isaac SimulationApp.close() exits Python, so the report must be assembled and
persisted inside the finally block before close() is called. A fake env whose
close() raises SystemExit exercises the fix.
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied_trackpad.sharpa_learning.bc_ppo import (
    ACTION_DIM,
    IMAGE_SIZE,
    PROPRIO_DIM,
    TOUCH_DIM,
    TOUCH_VALID_DIM,
)
from embodied_trackpad.sharpa_learning.supermimic import (
    LatentActor,
    LatentCritic,
    SuperMimicConfig,
    build_argparser,
    evaluate_supermimic,
    save_supermimic_checkpoint,
)
from embodied_trackpad.sharpa_learning.world_model import SharpaWorldModel, WorldModelConfig


class FakeBatcher:
    def get_state(self):
        return {"window": 4, "batch_size": 2, "bit_generator_state": "test"}


class FakeEnv:
    """Minimal SharpaEnv stand-in; close() aborts the process."""

    def __init__(self):
        self.closed = False
        self.episodes = 0

    def reset(self, seed=None, options=None):
        self.episodes += 1
        return self._obs(), {"sim_time": 0.0}

    def step(self, action):
        return self._obs(), 0.1, self.episodes >= 2, False, {"success": self.episodes >= 2}

    def close(self):
        self.closed = True
        raise SystemExit("Isaac-style fast shutdown")

    def _obs(self):
        return {
            "screen_rgb": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "proprio": np.zeros((PROPRIO_DIM,), dtype=np.float32),
            "touch": np.zeros((TOUCH_DIM,), dtype=np.float32),
            "touch_valid": np.zeros((TOUCH_VALID_DIM,), dtype=np.uint8),
        }


def fake_env_factory(_config):
    return FakeEnv()


def _make_tiny_checkpoint(path: str) -> None:
    cfg = SuperMimicConfig(
        seed=0,
        device="cpu",
        wm_config=WorldModelConfig(
            image_size=IMAGE_SIZE,
            proprio_dim=PROPRIO_DIM,
            touch_dim=TOUCH_DIM,
            touch_valid_dim=TOUCH_VALID_DIM,
            action_dim=ACTION_DIM,
            cnn_channels=(8, 16),
            view_embed_dim=8,
            proprio_embed_dim=8,
            fused_dim=24,
            latent_dim=16,
            gru_dim=16,
            ensemble_size=2,
            batch_size=2,
            window=4,
            device="cpu",
        ),
    )
    wm = SharpaWorldModel(cfg.wm_config)
    actor = LatentActor(cfg.wm_config.gru_dim, ACTION_DIM, cfg.actor_hidden_dim)
    critic = LatentCritic(cfg.wm_config.gru_dim, cfg.critic_hidden_dim)
    save_supermimic_checkpoint(
        path, wm, actor, critic,
        torch.optim.AdamW(wm.parameters()),
        torch.optim.AdamW(actor.parameters()),
        torch.optim.AdamW(critic.parameters()),
        cfg, {"updates": 1, "wm_updates": 1, "actor_updates": 0, "critic_updates": 0},
        FakeBatcher(), FakeBatcher(), "test-run", {"dataset_sha256": "abc"},
    )


def test_evaluate_writes_report_before_close():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "ckpt.pt")
        out = os.path.join(tmp, "out")
        _make_tiny_checkpoint(ckpt)

        # Make the fake factory importable by the CLI's module:attr resolver.
        mod = types.ModuleType("_test_fake_sharpa_env")
        mod.fake_env_factory = fake_env_factory
        sys.modules[mod.__name__] = mod

        parser = build_argparser()
        args = parser.parse_args([
            "evaluate",
            "--checkpoint", ckpt,
            "--env-factory", f"{mod.__name__}:fake_env_factory",
            "--episodes", "2",
            "--max-steps", "10",
            "--seed", "0",
            "--output", out,
            "--device", "cpu",
        ])
        try:
            evaluate_supermimic(args)
        except SystemExit:
            pass  # expected: Isaac-style close() aborts Python

        report_path = os.path.join(out, "closed_loop_eval.json")
        assert os.path.isfile(report_path), "closed_loop_eval.json was not written before env.close()"
        with open(report_path) as fh:
            disk = json.load(fh)
        assert disk["episodes"] == 2
        assert disk["mean_return"] > 0


if __name__ == "__main__":
    test_evaluate_writes_report_before_close()
    print("test_supermimic_eval_close PASSED")

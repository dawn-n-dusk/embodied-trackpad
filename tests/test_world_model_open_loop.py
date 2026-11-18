"""Regression tests for world_model.open_loop_eval protocol.

The two historical bugs being guarded:
  (a) reward persistence baseline used reward[start] (the first future target)
      and therefore reported zero error at horizon 1.  The fix samples
      start>=1 and uses reward[start-1] as the persistence prediction.
  (b) the latent metric compared a fixed 1-step predicted embedding to the
      true embedding at start+k.  The fix uses the k-step predicted embedding
      produced by rolling the latent state forward for k-1 steps and then
      predicting from action[k-1].
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied_trackpad.sharpa_learning.bc_ppo import (
    ACTION_DIM,
    IMAGE_SIZE,
    PROPRIO_DIM,
    TOUCH_DIM,
    TOUCH_VALID_DIM,
)
from embodied_trackpad.sharpa_learning.world_model import (
    SharpaWorldModel,
    WorldModelConfig,
    open_loop_eval,
)


def _tiny_cfg() -> WorldModelConfig:
    return WorldModelConfig(
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
    )


class _FakeEpisode:
    """Minimal Episode stand-in that satisfies open_loop_eval data access."""

    def __init__(self, T: int = 10):
        self.T = T
        self.arrays = {
            "action": np.zeros((T, ACTION_DIM), dtype=np.float32),
            "proprio": np.zeros((T + 1, PROPRIO_DIM), dtype=np.float32),
            "touch": np.zeros((T + 1, TOUCH_DIM), dtype=np.float32),
            "touch_valid": np.ones((T + 1, TOUCH_VALID_DIM), dtype=np.uint8),
            "reward": np.zeros((T,), dtype=np.float32),
            "screen_rgb": np.zeros((T + 1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((T + 1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        }

    def actor_obs(self, t: int) -> dict:
        return {k: self.arrays[k][t] for k in ("screen_rgb", "wrist_rgb",
                                                "proprio", "touch", "touch_valid")}


def _model_with_patched_predict(cfg: WorldModelConfig):
    """Return a tiny model whose predict/reconstruct/encode are deterministic."""
    model = SharpaWorldModel(cfg)

    def _encode(obs):
        # Encode the true observation so its embedding is determined by the
        # first proprio scalar, which the test sets to a known frame marker.
        marker = float(obs["proprio"].flat[0])
        return torch.full((1, cfg.latent_dim), marker, dtype=torch.float32)

    def _predict(h, action):
        # The predicted embedding is determined by the action marker.  The test
        # sets action[t] to have first scalar == t, so the embedding produced
        # after consuming action[k-1] is k-1.
        marker = float(action[0, 0].item())
        return {
            "reward": torch.zeros(cfg.ensemble_size, action.shape[0]),
            "continuation_logit": torch.zeros(cfg.ensemble_size, action.shape[0]),
            "continuation": torch.ones(cfg.ensemble_size, action.shape[0]),
            "next_emb": torch.full((cfg.ensemble_size, action.shape[0], cfg.latent_dim),
                                   marker, dtype=torch.float32),
            "reward_std": torch.zeros(action.shape[0]),
            "latent_std": torch.zeros(action.shape[0]),
        }

    def _reconstruct(h):
        return {
            "screen_small": torch.zeros(1, 3, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
            "wrist_small": torch.zeros(1, 3, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
            "proprio_touch": torch.zeros(1, PROPRIO_DIM + TOUCH_DIM),
        }

    def _imagine(h, actions):
        return {
            "reward": torch.zeros(1, actions.shape[1]),
            "continuation": torch.ones(1, actions.shape[1]),
            "reward_std": torch.zeros(1, actions.shape[1]),
            "latent_std": torch.zeros(1, actions.shape[1]),
            "final_state": h,
        }

    model.encode_obs_dict = _encode
    model.predict = _predict
    model.reconstruct = _reconstruct
    model.imagine = _imagine
    return model


def test_open_loop_reward_persistence_uses_last_observed_reward():
    """Reward baseline must be reward[start-1], not reward[start]."""
    cfg = _tiny_cfg()
    model = _model_with_patched_predict(cfg)

    T = 10  # max horizon 8 -> start is forced to 1
    ep = _FakeEpisode(T)
    # Make reward monotonic so reward[0] != reward[1].
    ep.arrays["reward"] = np.arange(T, dtype=np.float32) * 10.0

    # Mark each action so predict/reconstruct logic is exercised.
    for t in range(T):
        ep.arrays["action"][t, 0] = float(t)

    out = open_loop_eval(model, [ep], horizons=(1, 4, 8), samples_per_episode=4)

    for k in (1, 4, 8):
        true_r = ep.arrays["reward"][k]  # reward at start+k-1 with start=1
        base_r = ep.arrays["reward"][0]   # reward[start-1]
        expected = float((base_r - true_r) ** 2)
        actual = out["horizons"][str(k)]["persistence_reward"]
        assert pytest.approx(actual, abs=1e-6) == expected, (
            f"horizon {k}: persistence reward should use reward[start-1]={base_r}, "
            f"got {actual}; expected {expected}"
        )
        # If the buggy reward[start] baseline were used, horizon 1 would be 0.
        assert actual > 0.0, f"horizon {k}: persistence reward must be non-zero"


def test_open_loop_latent_metric_uses_k_step_prediction():
    """Latent error must compare the k-step predicted embedding to truth."""
    cfg = _tiny_cfg()
    model = _model_with_patched_predict(cfg)

    T = 10
    ep = _FakeEpisode(T)
    # start is forced to 1, so the final action consumed for horizon k is at
    # index start+k-1 == k and the true frame inspected is start+k == 1+k.
    # Mark that frame so its encoded embedding equals the k-step prediction.
    for k in (1, 4, 8):
        ep.arrays["proprio"][1 + k, 0] = float(k)
    for t in range(T):
        ep.arrays["action"][t, 0] = float(t)

    out = open_loop_eval(model, [ep], horizons=(1, 4, 8), samples_per_episode=4)

    for k in (1, 4, 8):
        # A 1-step latent metric would compare the embedding from action[start]
        # to the true start+k embedding and produce a non-zero MSE here.
        latent_mse = out["horizons"][str(k)]["latent"]
        assert pytest.approx(latent_mse, abs=1e-6) == 0.0, (
            f"horizon {k}: latent MSE should be zero when the k-step predicted "
            f"embedding matches the true start+k embedding; got {latent_mse}"
        )

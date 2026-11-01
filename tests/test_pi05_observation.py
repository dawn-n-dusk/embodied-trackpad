"""Software-only tests for the official pi0.5 Observation adapter.

These tests use clearly labeled lightweight dependency doubles injected into
``sys.modules`` for the real upstream import paths (``openpi``,
``openpi.transforms``, ``openpi.models.model``, ``openpi.models.tokenizer``,
``sentencepiece``). The doubles model call order, shapes and masks only — they
are NOT the official OpenPI code and passing these tests is not model
acceptance. Real upstream-runtime verification is a separate task.

The tokenizer asset is exercised with a small local double file whose SHA256
is computed at runtime; no download, no model, no GPU.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import pi05_observation as po

# --- labeled software-only doubles (order / shapes / masks, not upstream code) ---

CALLS: list[str] = []
TOKENIZE_STATE_SHAPES: list[tuple[int, ...]] = []
PAD_INPUT_DIMS: list[tuple[int, ...]] = []
LAST_TOKENIZER = {}


class _DoubleResizeImages:
    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width

    def __call__(self, data):
        CALLS.append("resize")
        assert (self.height, self.width) == (224, 224)
        out = dict(data)
        # Double semantics: deterministic crop to target shape (NOT the
        # official resize_with_pad); order and output shape are what is tested.
        out["image"] = {
            k: np.ascontiguousarray(v[: self.height, : self.width])
            for k, v in data["image"].items()
        }
        return out


class _DoubleTokenizePrompt:
    def __init__(self, tokenizer, discrete_state_input: bool = False) -> None:
        self.tokenizer = tokenizer
        self.discrete_state_input = discrete_state_input

    def __call__(self, data):
        CALLS.append("tokenize")
        prompt = data.pop("prompt")
        state = data.get("state") if self.discrete_state_input else None
        if self.discrete_state_input:
            assert state is not None
            TOKENIZE_STATE_SHAPES.append(tuple(state.shape))
        tokens, mask = self.tokenizer.tokenize(prompt, state)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
        }


class _DoublePadStatesAndActions:
    def __init__(self, model_action_dim: int) -> None:
        self.model_action_dim = model_action_dim

    def __call__(self, data):
        CALLS.append("pad")
        assert self.model_action_dim == 32
        PAD_INPUT_DIMS.append(
            (data["state"].shape[-1], data["actions"].shape[-1])
        )
        out = dict(data)
        out["state"] = np.pad(out["state"], (0, 32 - out["state"].shape[-1]))
        out["actions"] = np.pad(
            out["actions"], ((0, 0), (0, 32 - out["actions"].shape[-1]))
        )
        return out


class _DoubleObservation:
    """Double of the official Observation container; official uint8->[-1,1]."""

    @classmethod
    def from_dict(cls, data):
        CALLS.append("from_dict")
        assert ("tokenized_prompt" in data) == ("tokenized_prompt_mask" in data)
        images = {}
        for key, value in data["image"].items():
            value = np.asarray(value)
            if value.dtype == np.uint8:  # official scaling contract
                value = value.astype(np.float32) / 255.0 * 2.0 - 1.0
            images[key] = value
        obj = cls()
        obj.images = images
        obj.image_masks = data["image_mask"]
        obj.state = data["state"]
        obj.tokenized_prompt = data.get("tokenized_prompt")
        obj.tokenized_prompt_mask = data.get("tokenized_prompt_mask")
        return obj


class _DoublePaligemmaTokenizer:
    """Double base class; the adapter subclasses this exactly like upstream."""

    def __init__(self, max_len: int = 48) -> None:
        self._max_len = max_len

    def tokenize(self, prompt, state=None):
        LAST_TOKENIZER["prompt"] = prompt
        LAST_TOKENIZER["state_snapshot"] = None if state is None else state.copy()
        valid = 7 + (0 if state is None else len(state))
        tokens = np.arange(valid, dtype=np.int32)
        mask = np.ones(valid, dtype=bool)
        pad = self._max_len - valid
        return (
            np.pad(tokens, (0, pad)),
            np.pad(mask, (0, pad)),
        )


class _DoubleSentencePieceProcessor:
    def __init__(self, model_proto: bytes) -> None:
        LAST_TOKENIZER["model_proto"] = bytes(model_proto)


def _install_doubles(monkeypatch) -> None:
    """Install labeled doubles at the real upstream import paths."""

    def _module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    openpi = _module("openpi")
    models = _module("openpi.models")
    transforms = _module("openpi.transforms")
    model_mod = _module("openpi.models.model")
    tokenizer_mod = _module("openpi.models.tokenizer")
    sp_mod = _module("sentencepiece")

    transforms.ResizeImages = _DoubleResizeImages
    transforms.TokenizePrompt = _DoubleTokenizePrompt
    transforms.PadStatesAndActions = _DoublePadStatesAndActions
    model_mod.Observation = _DoubleObservation
    tokenizer_mod.PaligemmaTokenizer = _DoublePaligemmaTokenizer
    sp_mod.SentencePieceProcessor = _DoubleSentencePieceProcessor

    openpi.transforms = transforms
    openpi.models = models
    models.model = model_mod
    models.tokenizer = tokenizer_mod


@pytest.fixture
def doubles(monkeypatch):
    CALLS.clear()
    TOKENIZE_STATE_SHAPES.clear()
    PAD_INPUT_DIMS.clear()
    LAST_TOKENIZER.clear()
    # The adapter caches its tokenizer subclass across installs; drop it so
    # each test re-binds against the currently installed double modules.
    monkeypatch.delitem(po.__dict__, "_LocalPaligemmaTokenizer", raising=False)
    _install_doubles(monkeypatch)
    return monkeypatch


def _sample():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)
    state = rng.uniform(-1, 1, size=16).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(25, 16)).astype(np.float32)
    return rgb, state, actions


def _tokenizer_asset(tmp_path, payload: bytes = b"double-tokenizer-asset"):
    path = tmp_path / "paligemma_tokenizer.model"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _build(tmp_path, **overrides):
    rgb, state, actions = _sample()
    path, sha = _tokenizer_asset(tmp_path)
    kwargs = dict(
        rgb_uint8=rgb,
        state16=state,
        actions=actions,
        prompt="move the cursor to the target",
        tokenizer_path=path,
        expected_sha256=sha,
    )
    kwargs.update(overrides)
    return po.build_pi05_observation(**kwargs), (rgb, state, actions)


def test_module_import_path_has_no_heavy_imports():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(repo_root, "src"))
    code = (
        "import sys; "
        "import embodied_trackpad.leap_v1.pi05_observation as m; "
        "heavy = {'openpi', 'jax', 'torch', 'sentencepiece', 'flax'}; "
        "bad = [k for k in sys.modules if k.split('.')[0] in heavy]; "
        "assert not bad, bad; "
        "assert m.PINNED_TOKENIZER_SHA256 == "
        "'8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=env, check=True)


def test_transform_order_and_batch1_shapes(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    assert CALLS == ["resize", "tokenize", "pad", "from_dict"]
    obs = bundle.observation
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert obs.images[key].shape == (1, 224, 224, 3)
        assert obs.image_masks[key].shape == (1,)
    assert obs.state.shape == (1, 32)
    assert obs.tokenized_prompt.shape == (1, 200)
    assert obs.tokenized_prompt_mask.shape == (1, 200)
    assert bundle.actions.shape == (1, 25, 32)
    assert bundle.diagnostics["transform_order"][0] == "ResizeImages"


def test_tokenize_gets_16_state_before_padding(doubles, tmp_path):
    _build(tmp_path)
    assert TOKENIZE_STATE_SHAPES == [(16,)]
    assert PAD_INPUT_DIMS == [(16, 16)]
    assert LAST_TOKENIZER["prompt"] == "move the cursor to the target"
    # The double tokenizer saw the normalized state itself, not a padded copy.
    assert LAST_TOKENIZER["state_snapshot"].shape == (16,)


def test_masks_and_loss_mask32(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    assert bool(obs.image_masks["base_0_rgb"].all()) is True
    assert bool(obs.image_masks["left_wrist_0_rgb"].any()) is False
    assert bool(obs.image_masks["right_wrist_0_rgb"].any()) is False
    assert bundle.loss_mask32.dtype == bool
    assert bundle.loss_mask32.shape == (32,)
    assert bundle.loss_mask32[:16].all()
    assert not bundle.loss_mask32[16:].any()
    assert bundle.loss_mask32.sum() == 16


def test_action_padding_preserves_values(doubles, tmp_path):
    bundle, (_, state, actions) = _build(tmp_path)
    np.testing.assert_array_equal(bundle.actions[0, :, :16], actions)
    assert (bundle.actions[0, :, 16:] == 0.0).all()
    state_out = np.asarray(bundle.observation.state)
    assert state_out.shape == (1, 32)
    np.testing.assert_array_equal(state_out[0, :16], state)
    assert (state_out[0, 16:] == 0.0).all()


def test_image_scaling_and_missing_wrist(doubles, tmp_path):
    bundle, (rgb, _, _) = _build(tmp_path)
    obs = bundle.observation
    base = obs.images["base_0_rgb"]
    assert base.dtype == np.float32
    assert base.min() >= -1.0 and base.max() <= 1.0
    expected = rgb[:224, :224].astype(np.float32) / 255.0 * 2.0 - 1.0
    np.testing.assert_allclose(base[0], expected)
    # Missing wrist views are zero uint8 -> official scaling maps them to -1.
    np.testing.assert_array_equal(obs.images["left_wrist_0_rgb"], -1.0)
    np.testing.assert_array_equal(obs.images["right_wrist_0_rgb"], -1.0)


def test_normalized_inputs_unchanged(doubles, tmp_path):
    bundle, (rgb, state, actions) = _build(tmp_path)
    np.testing.assert_array_equal(np.asarray(bundle.observation.state)[0, :16], state)
    np.testing.assert_array_equal(bundle.actions[0, :, :16], actions)
    assert (np.asarray(bundle.observation.state)[0, 16:] == 0.0).all()


def test_original_arrays_not_mutated(doubles, tmp_path):
    rgb, state, actions = _sample()
    rgb_c, state_c, actions_c = rgb.copy(), state.copy(), actions.copy()
    path, sha = _tokenizer_asset(tmp_path)
    po.build_pi05_observation(
        rgb_uint8=rgb,
        state16=state,
        actions=actions,
        prompt="move the cursor to the target",
        tokenizer_path=path,
        expected_sha256=sha,
    )
    np.testing.assert_array_equal(rgb, rgb_c)
    np.testing.assert_array_equal(state, state_c)
    np.testing.assert_array_equal(actions, actions_c)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("rgb_uint8", np.zeros((224, 224, 3), np.uint8)),
        ("rgb_uint8", np.zeros((600, 800, 3), np.float32)),
        ("state16", np.zeros(16, np.float64)),
        ("state16", np.zeros(15, np.float32)),
        ("state16", np.full(16, np.nan, np.float32)),
        ("actions", np.zeros((25, 16), np.float64)),
        ("actions", np.zeros((24, 16), np.float32)),
        ("actions", np.full((25, 16), np.inf, np.float32)),
        ("prompt", ""),
        ("prompt", "   "),
    ],
)
def test_bad_inputs_rejected_before_heavy_import(doubles, tmp_path, field, bad_value):
    overrides = {field: bad_value}
    with pytest.raises(ValueError):
        _build(tmp_path, **overrides)
    assert "_LocalPaligemmaTokenizer" not in po.__dict__  # heavy phase never ran


def test_bad_hash_and_missing_file_rejected_before_heavy_import(doubles, tmp_path):
    path, _ = _tokenizer_asset(tmp_path)
    rgb, state, actions = _sample()
    common = dict(
        rgb_uint8=rgb,
        state16=state,
        actions=actions,
        prompt="move the cursor to the target",
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        po.build_pi05_observation(
            **common, tokenizer_path=path, expected_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="not found"):
        po.build_pi05_observation(
            **common,
            tokenizer_path=tmp_path / "missing.model",
            expected_sha256="0" * 64,
        )
    assert "_LocalPaligemmaTokenizer" not in po.__dict__  # heavy phase never ran


@pytest.mark.parametrize(
    "override",
    [
        {"image_size": (128, 128)},
        {"model_action_dim": 16},
        {"max_token_len": 48},
    ],
)
def test_unsupported_config_rejected_before_heavy_import(doubles, tmp_path, override):
    with pytest.raises(ValueError, match="must be"):
        _build(tmp_path, **override)
    assert "_LocalPaligemmaTokenizer" not in po.__dict__  # heavy phase never ran

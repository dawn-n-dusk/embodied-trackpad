"""Software-only tests for the explicit two-screen-view pi0.5 Observation adapter.

Uses clearly labeled lightweight doubles injected into ``sys.modules`` for the
real upstream import paths (``openpi``, ``openpi.transforms``,
``openpi.models.model``, ``openpi.models.tokenizer``, ``sentencepiece``).
The ``Observation`` double is a frozen dataclass so ``dataclasses.replace``
works exactly as on the official Flax struct. Doubles model call order,
shapes, masks and the explicit preprocess kwargs only — they are NOT the
official OpenPI code, and passing these tests is not trained-model
acceptance. No download, no model weights, no JAX, no GPU.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import pi05_observation as po
from embodied_trackpad.leap_v1 import pi05_screen_observation as pso
from embodied_trackpad.leap_v1.screen_views import prepare_screen_views

# --- labeled doubles (order / shapes / masks / preprocess kwargs only) ---

CALLS: list[str] = []
PREPROCESS_RECORD: dict = {}


@dataclasses.dataclass(frozen=True)
class _DoubleObservation:
    """Frozen-dataclass double: dataclasses.replace works like the Flax struct."""

    images: dict
    image_masks: dict
    state: object
    tokenized_prompt: object
    tokenized_prompt_mask: object
    # Optional in the official struct; exercised as None by default.
    token_ar_mask: object = None
    token_loss_mask: object = None

    @classmethod
    def from_dict(cls, data):
        CALLS.append("from_dict")
        images = {}
        for key, value in data["image"].items():
            value = np.asarray(value)
            if value.dtype == np.uint8:  # official scaling contract
                value = value.astype(np.float32) / 255.0 * 2.0 - 1.0
            images[key] = value
        return cls(
            images=images,
            image_masks={k: np.asarray(v) for k, v in data["image_mask"].items()},
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )


class _DoubleResizeImages:
    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width

    def __call__(self, data):
        CALLS.append("resize")
        assert (self.height, self.width) == (224, 224)
        out = dict(data)
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
        tokens, mask = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": mask}


class _DoublePadStatesAndActions:
    def __init__(self, model_action_dim: int) -> None:
        self.model_action_dim = model_action_dim

    def __call__(self, data):
        CALLS.append("pad")
        out = dict(data)
        out["state"] = np.pad(out["state"], (0, 32 - out["state"].shape[-1]))
        out["actions"] = np.pad(
            out["actions"], ((0, 0), (0, 32 - out["actions"].shape[-1]))
        )
        return out


class _DoublePaligemmaTokenizer:
    def __init__(self, max_len: int = 48) -> None:
        self._max_len = max_len

    def tokenize(self, prompt, state=None):
        valid = 7 + (0 if state is None else len(state))
        tokens = np.arange(valid, dtype=np.int32)
        mask = np.ones(valid, dtype=bool)
        return np.pad(tokens, (0, self._max_len - valid)), np.pad(mask, (0, self._max_len - valid))


class _DoubleSentencePieceProcessor:
    def __init__(self, model_proto: bytes) -> None:
        self.model_proto = bytes(model_proto)


def _double_preprocess_observation(rng=None, observation=None, train=None, image_keys=None):
    PREPROCESS_RECORD.update(
        rng=rng, observation=observation, train=train, image_keys=image_keys
    )
    return observation


def _install_doubles(monkeypatch) -> None:
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
    model_mod.preprocess_observation = _double_preprocess_observation
    tokenizer_mod.PaligemmaTokenizer = _DoublePaligemmaTokenizer
    sp_mod.SentencePieceProcessor = _DoubleSentencePieceProcessor

    openpi.transforms = transforms
    openpi.models = models
    models.model = model_mod
    models.tokenizer = tokenizer_mod


@pytest.fixture
def doubles(monkeypatch):
    CALLS.clear()
    PREPROCESS_RECORD.clear()
    monkeypatch.delitem(po.__dict__, "_LocalPaligemmaTokenizer", raising=False)
    _install_doubles(monkeypatch)
    return monkeypatch


# --- helpers ---

RED = (255, 59, 48)


def background_image() -> np.ndarray:
    rows = np.arange(600, dtype=np.uint16)[:, None]
    cols = np.arange(800, dtype=np.uint16)[None, :]
    img = np.empty((600, 800, 3), dtype=np.uint8)
    img[:, :, 0] = (40 + (cols % 16)).astype(np.uint8)
    img[:, :, 1] = (60 + (rows % 16)).astype(np.uint8)
    img[:, :, 2] = (80 + ((cols + rows) % 16)).astype(np.uint8)
    return img


def sample_inputs(with_cursor: bool = True):
    rng = np.random.default_rng(1)
    rgb = background_image()
    if with_cursor:
        rgb[300:312, 400:412] = RED  # 12x12 blob -> local view valid
    state = rng.uniform(-1, 1, size=16).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(25, 16)).astype(np.float32)
    return rgb, state, actions


def _tokenizer_asset(tmp_path, payload: bytes = b"double-tokenizer-asset"):
    path = tmp_path / "paligemma_tokenizer.model"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _build(tmp_path, with_cursor=True, **overrides):
    rgb, state, actions = sample_inputs(with_cursor=with_cursor)
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
    return pso.build_pi05_screen_observation(**kwargs), (rgb, state, actions)


# --- layout / provenance ---


def test_layout_has_exactly_two_screen_keys_in_order(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    assert tuple(obs.images.keys()) == pso.SCREEN_PAIR_KEYS == ("base_0_rgb", "screen_local_0_rgb")
    assert tuple(obs.image_masks.keys()) == pso.SCREEN_PAIR_KEYS
    for key in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert key not in obs.images
        assert key not in obs.image_masks


def test_wrist_absence_is_provenance_not_renamed_valid(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    d = bundle.diagnostics
    assert d["absent_sensor_masks"] == {
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    assert set(d["image_mask"].keys()) == set(pso.SCREEN_PAIR_KEYS)
    assert d["image_mask"]["base_0_rgb"] is True
    assert d["view_roles"] == {
        "base_0_rgb": "screen_full",
        "screen_local_0_rgb": "screen_local",
    }
    assert d["view_order"] == list(pso.SCREEN_PAIR_KEYS)
    assert d["view_valid"]["base_0_rgb"] is True
    assert d["layout"] == "explicit_two_screen_views_no_wrist"


# --- pixel parity / scaling ---


def test_local_view_pixel_parity_official_scaling(doubles, tmp_path):
    bundle, (rgb, _, _) = _build(tmp_path)
    obs = bundle.observation
    views = prepare_screen_views(rgb)
    expected_local = views.local_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    np.testing.assert_array_equal(obs.images["screen_local_0_rgb"][0], expected_local)
    assert obs.images["screen_local_0_rgb"].shape == (1, 224, 224, 3)
    assert obs.images["screen_local_0_rgb"].dtype == np.float32
    # Full view came through the reused builder's official resize+scaling.
    expected_full = rgb[:224, :224].astype(np.float32) / 255.0 * 2.0 - 1.0
    np.testing.assert_array_equal(obs.images["base_0_rgb"][0], expected_full)


def test_invalid_local_view_is_minus_one_with_false_mask(doubles, tmp_path):
    bundle, _ = _build(tmp_path, with_cursor=False)
    obs = bundle.observation
    assert bool(obs.image_masks["base_0_rgb"][0]) is True
    assert bool(obs.image_masks["screen_local_0_rgb"][0]) is False
    np.testing.assert_array_equal(obs.images["screen_local_0_rgb"], -1.0)
    d = bundle.diagnostics
    assert d["view_valid"]["screen_local_0_rgb"] is False
    assert d["image_mask"]["screen_local_0_rgb"] is False
    assert d["crop_origin_col_row"] is None
    assert d["cursor_col_row"] is None
    assert d["cursor_confidence"] == 0.0
    assert not d["local_pixel_valid"].any()
    assert "not attention mask" in d["local_pixel_valid_meaning"]


def test_valid_local_pixel_mask_matches_screen_views(doubles, tmp_path):
    bundle, (rgb, _, _) = _build(tmp_path)
    views = prepare_screen_views(rgb)
    np.testing.assert_array_equal(
        bundle.diagnostics["local_pixel_valid"], views.local_pixel_valid
    )
    assert bundle.diagnostics["local_pixel_valid"].all()  # interior crop


# --- preserved values / independence ---


def test_state_actions_tokens_and_loss_mask_preserved(doubles, tmp_path):
    bundle, (rgb, state, actions) = _build(tmp_path)
    obs = bundle.observation
    np.testing.assert_array_equal(np.asarray(obs.state)[0, :16], state)
    assert (np.asarray(obs.state)[0, 16:] == 0.0).all()
    np.testing.assert_array_equal(bundle.actions[0, :, :16], actions)
    assert (bundle.actions[0, :, 16:] == 0.0).all()
    assert bundle.loss_mask32[:16].all() and not bundle.loss_mask32[16:].any()
    assert obs.tokenized_prompt.shape == (1, 200)
    assert obs.tokenized_prompt_mask.shape == (1, 200)
    # Same values the legacy 3-camera builder produces for the shared fields.
    path, sha = _tokenizer_asset(tmp_path)
    legacy = po.build_pi05_observation(
        rgb_uint8=rgb,
        state16=state,
        actions=actions,
        prompt="move the cursor to the target",
        tokenizer_path=path,
        expected_sha256=sha,
    )
    np.testing.assert_array_equal(np.asarray(obs.state), np.asarray(legacy.observation.state))
    np.testing.assert_array_equal(
        np.asarray(obs.tokenized_prompt), np.asarray(legacy.observation.tokenized_prompt)
    )
    np.testing.assert_array_equal(bundle.actions, legacy.actions)
    np.testing.assert_array_equal(bundle.loss_mask32, legacy.loss_mask32)


def test_inputs_not_mutated_and_no_alias(doubles, tmp_path):
    rgb, state, actions = sample_inputs()
    rgb_c, state_c, actions_c = rgb.copy(), state.copy(), actions.copy()
    path, sha = _tokenizer_asset(tmp_path)
    bundle = pso.build_pi05_screen_observation(
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
    obs = bundle.observation
    assert not np.shares_memory(obs.images["base_0_rgb"], rgb)
    assert not np.shares_memory(obs.images["screen_local_0_rgb"], rgb)
    assert not np.shares_memory(obs.images["base_0_rgb"], obs.images["screen_local_0_rgb"])
    assert not np.shares_memory(bundle.actions, actions)


def test_builder_reuses_legacy_pipeline_order(doubles, tmp_path):
    _build(tmp_path)
    assert CALLS == ["resize", "tokenize", "pad", "from_dict"]


def test_builder_bad_input_rejected_before_heavy_phase(doubles, tmp_path):
    with pytest.raises(ValueError):
        _build(tmp_path, rgb_uint8=np.zeros((224, 224, 3), dtype=np.uint8))
    assert CALLS == []


# --- diagnostics describe the actual two-view Observation ---


@pytest.mark.parametrize("with_cursor", [True, False], ids=["valid_local", "invalid_local"])
def test_final_image_diagnostics_match_actual_observation(doubles, tmp_path, with_cursor):
    bundle, _ = _build(tmp_path, with_cursor=with_cursor)
    obs = bundle.observation
    d = bundle.diagnostics
    for key in ("image_mask", "image_resized_shape", "image_output_shape", "image_output_dtype"):
        assert set(d[key].keys()) == set(pso.SCREEN_PAIR_KEYS), key
        assert list(d[key].keys()) == list(pso.SCREEN_PAIR_KEYS), key
    for key in pso.SCREEN_PAIR_KEYS:
        assert d["image_output_shape"][key] == tuple(np.asarray(obs.images[key]).shape)
        assert d["image_output_dtype"][key] == str(np.asarray(obs.images[key]).dtype)
    # Full view always valid; local validity follows the cursor.
    assert d["image_mask"]["base_0_rgb"] is True
    assert d["image_mask"]["screen_local_0_rgb"] == with_cursor
    # Historical 3-camera transform outputs stay explicitly separate.
    base = d["base_builder"]
    for key in ("image_resized_shape", "image_output_shape", "image_output_dtype"):
        assert set(base[key].keys()) == {
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        }
    assert base["transform_order"][0] == "ResizeImages"


# --- no aliasing with the tracked internal base bundle ---


def _spy_on_internal_builder(monkeypatch, captured: dict, inject_masks: bool = False):
    original = pso.build_pi05_observation

    def spy(**kwargs):
        bundle = original(**kwargs)
        obs = bundle.observation
        if inject_masks:
            obs = dataclasses.replace(
                obs,
                token_ar_mask=np.ones((1, 200), dtype=bool),
                token_loss_mask=np.zeros((1, 200), dtype=bool),
            )
            bundle = po.Pi05ObservationBundle(
                observation=obs,
                actions=bundle.actions,
                loss_mask32=bundle.loss_mask32,
                diagnostics=bundle.diagnostics,
            )
        captured["base_observation"] = obs  # what the screen builder consumed
        return bundle

    monkeypatch.setattr(pso, "build_pi05_observation", spy)


def test_carried_fields_do_not_alias_tracked_base_bundle(doubles, monkeypatch, tmp_path):
    captured: dict = {}
    _spy_on_internal_builder(monkeypatch, captured)
    bundle, (_, state16, _) = _build(tmp_path)
    obs = bundle.observation
    base = captured["base_observation"]
    # Values preserved exactly...
    np.testing.assert_array_equal(np.asarray(obs.state), np.asarray(base.state))
    np.testing.assert_array_equal(np.asarray(obs.tokenized_prompt), np.asarray(base.tokenized_prompt))
    np.testing.assert_array_equal(
        np.asarray(obs.tokenized_prompt_mask), np.asarray(base.tokenized_prompt_mask)
    )
    # ...but nothing aliases the internal base bundle.
    assert not np.shares_memory(obs.state, base.state)
    assert not np.shares_memory(obs.tokenized_prompt, base.tokenized_prompt)
    assert not np.shares_memory(obs.tokenized_prompt_mask, base.tokenized_prompt_mask)
    assert not np.shares_memory(obs.images["base_0_rgb"], base.images["base_0_rgb"])
    assert not np.shares_memory(obs.images["screen_local_0_rgb"], base.images["base_0_rgb"])
    np.testing.assert_array_equal(np.asarray(obs.state)[0, :16], state16)


def test_optional_token_masks_copied_not_aliased(doubles, monkeypatch, tmp_path):
    captured: dict = {}
    _spy_on_internal_builder(monkeypatch, captured, inject_masks=True)
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    base = captured["base_observation"]
    assert obs.token_ar_mask is not None and obs.token_loss_mask is not None
    np.testing.assert_array_equal(obs.token_ar_mask, base.token_ar_mask)
    np.testing.assert_array_equal(obs.token_loss_mask, base.token_loss_mask)
    assert not np.shares_memory(obs.token_ar_mask, base.token_ar_mask)
    assert not np.shares_memory(obs.token_loss_mask, base.token_loss_mask)


# --- explicit preprocessing ---


def test_preprocess_invokes_official_with_explicit_kwargs(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    out = pso.preprocess_pi05_screen_observation(bundle.observation)
    assert PREPROCESS_RECORD["rng"] is None
    assert PREPROCESS_RECORD["train"] is False
    assert tuple(PREPROCESS_RECORD["image_keys"]) == pso.SCREEN_PAIR_KEYS
    assert out is bundle.observation  # both views retained, not re-ordered
    assert tuple(out.images.keys()) == pso.SCREEN_PAIR_KEYS


def _tampered(obs, **fields):
    return dataclasses.replace(obs, **fields)


def test_preprocess_rejects_legacy_wrist_keys(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    images = {**obs.images, "left_wrist_0_rgb": obs.images["base_0_rgb"]}
    with pytest.raises(ValueError, match="exactly"):
        pso.preprocess_pi05_screen_observation(_tampered(obs, images=images))
    assert PREPROCESS_RECORD == {}


@pytest.mark.parametrize(
    "images,masks",
    [
        ({**{"base_0_rgb": None, "screen_local_0_rgb": None}, "extra_rgb": None}, None),  # extra key
        ({"base_0_rgb": None}, None),  # missing local key
        ({"screen_local_0_rgb": None, "base_0_rgb": None}, None),  # wrong order
    ],
    ids=["extra_key", "missing_local", "wrong_order"],
)
def test_preprocess_rejects_bad_key_sets(doubles, tmp_path, images, masks):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    new_images = {k: (obs.images[k] if k in obs.images else obs.images["base_0_rgb"]) for k in images}
    new_masks = (
        {k: (obs.image_masks[k] if k in obs.image_masks else obs.image_masks["base_0_rgb"]) for k in masks}
        if masks is not None
        else obs.image_masks
    )
    with pytest.raises(ValueError, match="exactly"):
        pso.preprocess_pi05_screen_observation(
            _tampered(obs, images=new_images, image_masks=new_masks)
        )
    assert PREPROCESS_RECORD == {}


def test_preprocess_rejects_mask_key_mismatch(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    masks = {"base_0_rgb": obs.image_masks["base_0_rgb"], "other_rgb": obs.image_masks["base_0_rgb"]}
    with pytest.raises(ValueError, match="exactly"):
        pso.preprocess_pi05_screen_observation(_tampered(obs, image_masks=masks))
    assert PREPROCESS_RECORD == {}


def test_preprocess_rejects_base_mask_false(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    masks = {**obs.image_masks, "base_0_rgb": np.asarray([False])}
    with pytest.raises(ValueError, match="must be True"):
        pso.preprocess_pi05_screen_observation(_tampered(obs, image_masks=masks))
    assert PREPROCESS_RECORD == {}


@pytest.mark.parametrize(
    "field,transform,match",
    [
        ("shape", lambda a: np.zeros((1, 225, 224, 3), np.float32), "shape"),
        ("dtype", lambda a: a.astype(np.float64), "dtype"),
        ("range", lambda a: a * 2.0, "must lie in"),
        ("finite", lambda a: np.where(a == a.flat[0], np.float32(np.nan), a), "finite"),
    ],
)
def test_preprocess_rejects_bad_local_image(doubles, tmp_path, field, transform, match):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    bad = transform(np.array(obs.images["screen_local_0_rgb"], copy=True))
    images = {**obs.images, "screen_local_0_rgb": bad}
    with pytest.raises(ValueError, match=match):
        pso.preprocess_pi05_screen_observation(_tampered(obs, images=images))
    assert PREPROCESS_RECORD == {}


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((224, 224), dtype=bool),  # wrong shape
        np.asarray([1], dtype=np.int32),  # non-bool
    ],
    ids=["wrong_shape", "non_bool"],
)
def test_preprocess_rejects_bad_local_mask(doubles, tmp_path, mask):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    masks = {**obs.image_masks, "screen_local_0_rgb": mask}
    with pytest.raises(ValueError):
        pso.preprocess_pi05_screen_observation(_tampered(obs, image_masks=masks))
    assert PREPROCESS_RECORD == {}


def test_preprocess_rejects_bad_state_and_tokens(doubles, tmp_path):
    bundle, _ = _build(tmp_path)
    obs = bundle.observation
    with pytest.raises(ValueError, match="state"):
        pso.preprocess_pi05_screen_observation(
            _tampered(obs, state=np.zeros((1, 16), np.float32))
        )
    with pytest.raises(ValueError, match="tokenized_prompt"):
        pso.preprocess_pi05_screen_observation(
            _tampered(obs, tokenized_prompt=np.zeros((1, 48), np.int32))
        )
    assert PREPROCESS_RECORD == {}


# --- import lightness ---


def test_module_import_has_no_heavy_imports():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(repo_root, "src"))
    code = (
        "import sys; "
        "import embodied_trackpad.leap_v1.pi05_screen_observation as m; "
        "heavy = {'openpi', 'jax', 'torch', 'sentencepiece', 'flax'}; "
        "bad = [k for k in sys.modules if k.split('.')[0] in heavy]; "
        "assert not bad, bad; "
        "assert m.SCREEN_PAIR_KEYS == ('base_0_rgb', 'screen_local_0_rgb')"
    )
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=env, check=True)

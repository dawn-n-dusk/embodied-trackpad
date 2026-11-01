# Third-party materials

No model weights, full papers, or hand meshes are bundled in the initial repository.

The current LEAP v1 Full baseline uses MuJoCo Menagerie `leap_hand` at `a662de603224c47123b0783821c58ad4402ad3e3`, whose per-model license is MIT, Copyright 2023 Ananye Agarwal. The [manifest](assets/manifests/leap-v1-full.json) pins the right-hand XML, meshes and license; downloaded assets remain ignored. Project-authored diagnostic renders under `docs/assets/leap-v1-*` depict this attributed model. The official noncommercial LEAP hardware API is not incorporated. The [v1 guide](docs/leap-v1-model.md) discloses model adaptations and uncalibrated dynamics.

Current LEAP v2 Advanced import uses a pinned official URDF/STL download manifest. The repository's nested `leap_v2/LICENSE` contains Apache-2.0, while its official CAD pages separately state CC BY-NC-SA. These are distinct sources; no blanket license claim is made for every CAD/mesh asset. Keep upstream models and converted model outputs in ignored caches with provenance; public project-authored adapters and reports do not relicense those assets. The [latest asset audit](docs/research/2026-10-31-leap-latest-assets.zh-CN.md) records both sources.

- MuJoCo and MuJoCo Warp: follow their upstream licenses; pin runtime versions in experiments.
- MuJoCo Menagerie: individual robot models carry their own licenses. Allegro and Shadow assets require separate attribution and pinned source revisions if used.
- Historical LEAP v1 selection (superseded by ADR 0006): Menagerie `leap_hand` at `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` is MIT; MuJoCo Playground is Apache-2.0. The official `leap-hand/LEAP_Hand_API` LICENSE, checked 2026-10-31, is CC BY-NC 4.0. Model, task-code and hardware-API licenses are distinct; do not copy the API into this project and relabel it Apache-2.0. No LEAP assets or API code are bundled by the selection decision. See [the selection review](docs/research/2026-10-31-hand-selection.zh-CN.md).
- Qwen model weights: follow the specific model repository's license; record model/processor revisions. A model-card claim is not a local benchmark.
- OS-Atlas and any future datasets: retain dataset-level provenance and inspect component redistribution rights. A repository license alone does not establish rights to every screenshot.
- ACT, MEM, and related methods: cite the original research. If code is incorporated, preserve its own license and attribution.
- Tart 2.41.1 is an optional external VM runtime, under FSL-1.1-ALv2 (with the upstream future-license terms); its binary and macOS guest images are not redistributed in this Apache-2.0 repository. Record archive/signature verification and guest OCI digest separately. Reference: https://github.com/openai/tart/tree/2.41.1.

References and public source locations are listed in docs/references.md. Downloaded source snapshots and private audit receipts remain outside the release.

- The flow-loss body in `pi05_jax_loss.py` adapts Physical Intelligence OpenPI `Pi0.compute_loss` at commit `215abfb217dbac7d5f1273282331b9b1866c0479` ([source](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)). GUI image augmentation is disabled and only the first 16 action dimensions contribute to loss. The upstream Apache-2.0 [license is retained](third_party/licenses/openpi-LICENSE); model weights and datasets are not redistributed.

- `pi05_sampling.py` adapts the same pinned OpenPI `Pi0.sample_actions` under the retained Apache-2.0 license. It adds explicit image-layout preprocessing and writes the prefix-mask repeat as an equivalent broadcast; it preserves the upstream KV cache, attention, conditioning and Euler integration. No weights or datasets are bundled.

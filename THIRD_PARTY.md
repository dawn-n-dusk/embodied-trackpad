# Third-party materials

No model weights, full papers, or hand meshes are bundled in the initial repository.

- MuJoCo and MuJoCo Warp: follow their upstream licenses; pin runtime versions in experiments.
- MuJoCo Menagerie: individual robot models carry their own licenses. Allegro and Shadow assets require separate attribution and pinned source revisions if used.
- LEAP selection: Menagerie `leap_hand` at `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` is MIT; MuJoCo Playground is Apache-2.0. The official `leap-hand/LEAP_Hand_API` LICENSE, checked 2026-10-31, is CC BY-NC 4.0. Model, task-code and hardware-API licenses are distinct; do not copy the API into this project and relabel it Apache-2.0. No LEAP assets or API code are bundled by the selection decision. See [the selection review](docs/research/2026-10-31-hand-selection.zh-CN.md).
- Qwen model weights: follow the specific model repository's license; record model/processor revisions. A model-card claim is not a local benchmark.
- OS-Atlas and any future datasets: retain dataset-level provenance and inspect component redistribution rights. A repository license alone does not establish rights to every screenshot.
- ACT, MEM, and related methods: cite the original research. If code is incorporated, preserve its own license and attribution.
- Tart 2.41.1 is an optional external VM runtime, under FSL-1.1-ALv2 (with the upstream future-license terms); its binary and macOS guest images are not redistributed in this Apache-2.0 repository. Record archive/signature verification and guest OCI digest separately. Reference: https://github.com/openai/tart/tree/2.41.1.

References and public source locations are listed in docs/references.md. Downloaded source snapshots and private audit receipts remain outside the release.

# Third-party materials

No model weights, full papers, or hand meshes are bundled in the initial repository.

- MuJoCo and MuJoCo Warp: follow their upstream licenses; pin runtime versions in experiments.
- MuJoCo Menagerie: individual robot models carry their own licenses. Allegro and Shadow assets require separate attribution and pinned source revisions if used.
- Qwen model weights: follow the specific model repository's license; record model/processor revisions. A model-card claim is not a local benchmark.
- OS-Atlas and any future datasets: retain dataset-level provenance and inspect component redistribution rights. A repository license alone does not establish rights to every screenshot.
- ACT, MEM, and related methods: cite the original research. If code is incorporated, preserve its own license and attribution.
- Tart 2.41.1 is an optional external VM runtime, under FSL-1.1-ALv2 (with the upstream future-license terms); its binary and macOS guest images are not redistributed in this Apache-2.0 repository. Record archive/signature verification and guest OCI digest separately. Reference: https://github.com/openai/tart/tree/2.41.1.

References and public source locations are listed in docs/references.md. Downloaded source snapshots and private audit receipts remain outside the release.

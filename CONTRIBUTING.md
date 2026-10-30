# Contributing and recording work

1. State the concrete objective, allowed files/resources, acceptance checks, and exclusions before dispatch.
2. Implement on a dedicated branch. Keep implementation and coordinator documentation ownership separate when work runs concurrently.
3. Record the source revision, environment/configuration, commands, observed results, and limitations. Keep raw private logs outside Git; publish reviewed evidence summaries and checksums.
4. Run relevant checks. A failed run remains part of the record. Distinguish mocked, CPU-simulated, GPU-simulated, real-Mac, and physical evidence.
5. Update STATUS.md, relevant design/usage docs, CHANGELOG.md, and the experiment ledger. The ledger is append-only; corrections reference the superseded entry.
6. Review tracked files for credentials, private host/user details, personal screens, large artifacts, and third-party redistribution restrictions. Use explicit paths when staging.
7. The coordinator reviews, commits, pushes, and verifies the remote commit and any CI result. A worker report or successful push alone does not establish behavioral acceptance.

An experiment record uses the template in experiments/templates/record.json. Use `null` for unknown values, not invented zeros. The record's source commit identifies code used in the run; its result record is included in a subsequent Git commit, avoiding self-referential commit hashes.

Large artifacts belong in a separately managed dataset/artifact store. Store their immutable identifiers, checksums, license, and retention policy in reviewed manifests. Never put access tokens or signed private URLs into manifests.

Public publication is authorized for this project. It does not authorize exporting unrelated workspaces, exposing private repositories, publishing personal recordings, or changing other services.

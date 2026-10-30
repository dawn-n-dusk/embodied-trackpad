# 0004 — Comparable physics runs and observable live frames

Date: 2026-10-31 (Asia/Hong_Kong). Status: acceptance contract; execution results remain in `STATUS.md` and the experiment ledger.

The live viewer and H800 benchmark must consume a stable, hashed articulated-hand scene. A generic streaming self-test proves only infrastructure. It cannot establish the project's hand, pad contact, GUI control, or learned behavior.

## Shared scene and device contract

- The primary scene has the Panda's 7 arm actuators and Allegro's 16 hand actuators. A declared initializer sets joint position, velocity and control once at reset. Every consumer uses it: live rendering, CPU reference, GPU initial state and benchmarks.
- Subsequent motion comes from actuator controls and physics steps. Diagnostic IK may compute a teacher reference in a separate scratch state; it must not overwrite the executing world's pose.
- The pad is 160 × 110 mm. Touch coordinates are pad-centered meters, +x right and +y up. Bounds are ±0.080 and ±0.055 m. The physical scene and device model must use the same dimensions.
- Contacts identify actual robot bodies and pad geoms, separating fingertips, other phalanges, palm and unrelated/self contacts. A positive total contact count is not evidence that a fingertip touched the pad.
- The scene snapshot includes model source, assets manifest, mount transform, reset state, controller and control schedule hashes. Invalid mounting penetrations must be fixed geometrically; suppressing collisions against moving fingers is not an acceptable repair.

## CPU/GPU comparison

For the first reproducibility benchmark, record one explicit time-indexed actuator schedule and execute the identical rows from identical initial states on both backends. If a teacher generated that schedule using state feedback, the benchmark is a replay of its actuator references, not closed-loop GPU control. Future learned controllers have a separate acceptance stage.

Validate 1 and 32 worlds before admitting 128 and 512. A completed CPU test cannot stand in for an earlier GPU stage. Record finite-state failures throughout the trajectory, peak penetration, actual pad contacts, solver/capacity warnings and trajectory agreement. Report the coverage over worlds explicitly. Keep tolerances tied to joint units and contact scale; preserve failures rather than widening tolerances to pass.

Measure model upload, cold compilation, warmup and steady stepping separately. Both CPU and GPU throughput measurements use a declared matching schedule and initialization. Rendering, encoding and network transfer are separate costs. Repeated identical worlds are useful for throughput checks but are not diverse training data.

## Live viewing and role boundaries

Publish immutable frame bundles containing pixels, frame ID, wall capture time, simulation time and physics/contact metadata from the same state. HTTP handlers must not inspect `MjData` while the physics thread mutates it. The UI must expire its live indicator when either frames or state stop arriving, including a request that never resolves.

The viewer labels debug/teacher/learned control, source host/backend, scene identity, observed update rate and connection state. No learned policy or Mac task success is inferred from a moving debug hand. A prerecorded image or a generic arm must not be presented as the requested full-hand live stream.

Kimi owns implementation, installation, deployment, simulation, Slurm runs, training, VM operation and capture/input experiments. The coordinator owns contracts, budgets, read-only source/evidence/visual review, documentation, integration and publication. A timeout produces a saved checkpoint and a newly bounded Kimi continuation, not a coordinator takeover.

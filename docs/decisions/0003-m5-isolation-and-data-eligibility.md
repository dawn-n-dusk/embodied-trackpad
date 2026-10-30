# ADR 0003: M5 test host, confined inputs, and data eligibility

Date: 2026-10-30. Status: accepted design; runtime acceptance is separate.

The user cannot currently dedicate the M1 and explicitly authorized the M5 host, macOS virtualization, and continued experiments/training. The user subsequently authorized starting a small guest despite observed host memory pressure, and will report interactive slowdowns.

## Decisions

1. Use a project-owned native AppKit test surface for a small repeatable GUI environment. Restrict its permissions with App Sandbox and restrict the robot input route to its own content. Verify signed entitlements and actual denied operations separately. A signed app alone is not a proof of every security property.
2. In parallel, prepare one macOS guest with an initial budget of four CPUs and 6 GB memory. Keep the runtime/cache/disk in project-specific private storage. Do not share host folders, clipboard, USB, personal accounts or a host-wide input interface. Identify the base image by immutable digest after download. No unrelated VM deletion or automatic pruning.
3. Full macOS guest results and native test-surface results are different evidence classes. VM creation, boot, framebuffer capture, guest-only pointer input, physics-derived input and learned-policy performance are separate checks.
4. Robot actions remain actuator references. The simulator exports observed pad contacts as `touch.v1`; the device model derives relative movement, button transitions and scrolling. First contact has no previous position and must not teleport a pointer. Contact identity, omission/lift, phase changes, expiry and stopping require explicit semantics.
5. Actor observations are rendered pixels plus declared proprioception/history. Task layout, system pointer truth and success scores remain privileged. The bridge itself necessarily maintains device state but cannot pass hidden GUI state through ordinary actor responses.
6. M0 fixtures are software test data. They may exercise loaders or tensor shapes, but stay `training_eligible=false` and do not count as robot demonstrations. Real dynamic simulation data can train policies after provenance, time alignment and task-specific checks. Actual failed attempts may be valuable; fabricated outcomes are not.
7. Host permission changes or a guest application launch are not robot training results. Keep evidence of source/model versions, actual controls/states/contacts, consumed images, outcomes and experiment budgets before admitting a batch to training.

## Consequences

The native app remains useful even after the VM is ready, since it provides resets and controlled task distributions. A guest allows later Finder/Preview experiments without granting the policy a general host-desktop input path. Neither route establishes Apple native multitouch/Force Touch support.

Recorded host memory pressure is a point-in-time resource observation, not a claim that virtualization is unsupported. The user explicitly chose to try the VM; do not repeatedly seek the same authorization.

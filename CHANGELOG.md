# Changelog

- Complete local regression passed 1,148 tests with 31 optional skips. Preserved six initial native protocol failures and the stale ignored app, then rebuilt unchanged source and verified the corrected binary. No source assertions were relaxed.

- Added shared static screen-layout routing to the valid-16 loss, full-update closure and official action sampler. Independent focused tests: 103 passed, 8 dependency skips; real compiled JAX and full-model execution remain pending. Preserved the legacy default and upstream sampler attribution.

- Verified explicit full/local screen preprocessing in the pinned OpenPI CPU runtime: 121 actual checks and 99 independent checks of 36 arrays passed in a 31-second allocation. Pixels and state/token/action values match exactly; compiled policy routing and learned precision remain separate.

- Accepted complete pi0.5 step-1 checkpoint restoration on CPU: all 156 state leaves and the complete fingerprint dictionary match the frozen H800 state. A bounded 116-second storage copy and one 136-second CPU job preserved the original checkpoint, used no GPU, and released their resources. GPU continuation/replay and learned precision remain pending.

- Added explicit full/local screen Observation construction and host-side preprocessing with 138 independent software checks. Kept absent wrist sensors explicit; actual OpenPI runtime and compiled loss/sampler routing remain separate gates.

- Added standalone full-screen plus native-pixel cursor-local views with explicit edge padding and invalid-view masks. Independent checks: 88 software tests passed; two audited real frames preserved every cursor pixel. Model input-slot integration and policy evaluation remain pending.

- The CPU-only restore attempt failed its allocation preflight after 1 second because the selected CPU node could not see the preserved scratch checkpoint. Recorded the failed denominator, zero GPU use, and unchanged checkpoint inventory/metadata; no model launch or retry.

- Added a shape-only NNX/Optax restore-template helper and a bounded CPU recovery contract. Three new local checks passed and three JAX checks remain pending; no new GPU allocation or restoration result.

- Verified two contiguous complete pi0.5 updates and a 31.15 GB step-1 checkpoint save. The 452-second allocation hit its deadline during the second fingerprint; disk restore/replay remain unverified. Recorded full-state transfer timings and the 39-second remaining smoke budget.

- Reviewed the scratch recovery caller/launcher with 25 independent boundary checks. Runtime remains pending; one source-bound allocation is capped at 480 seconds within the existing smoke budget.

- Reviewed the checkpoint-first recovery sequence: nine local metadata/source checks passed, including bounded progress logs and atomic report publication. Full-model runtime and scratch launcher integration remain pending.

- Accepted the actual 8-second H800 fingerprint diagnostic: 12 real JAX callback checks and 32/256/1024 MiB full-hash parity passed. Kept the earlier complete-model timeout unexplained; prepared a checkpoint-first recovery design within the remaining 491 GPU seconds.

- Added optional per-leaf transfer/hash progress events without changing full-state fingerprint semantics, and corrected the host-cache memory guarantee. Independent local checks: 14 passed, 5 skipped; real JAX callback tests and performance measurements remain pending.

- Audited the corrected H800 retry: one full-model update passed, but full-state fingerprinting exhausted the 870-second internal deadline before checkpoint saving. Retained exact failure evidence and the 871-second allocation cost; no recovery, live training or precision-control claim.

- Repaired checkpoint-size monitoring to tolerate disappearing child entries while retaining other scan errors. Nine local filesystem checks plus isolated shell output/error checks passed; one reviewed retry keeps the original GPU, memory and time limits.

- Verified the first real 3.35B full-parameter pi0.5 update on H800. The 246-second allocation stopped during checkpoint saving after a directory-size scanner error; the partial checkpoint and all failure evidence are retained, and recovery remains unaccepted.

- Reviewed the complete pi0.5 update/recovery driver and a single-H800 15-minute contract. Corrected model-reference retention, interrupted-step accounting, staged-path assumptions and process-identity guards; real optimizer execution remains independently gated.

- Accepted the actual CPU checkpoint roundtrip: 22 checks in 6 seconds verify complete tiny NNX/Optax state restoration and exact continuation/replay parity. Retained preparation and caller timeouts separately; full pi0.5 updates remain pending.

- Added reviewed checkpoint helpers for full optimizer/RNG state and streamed full-state fingerprints. Ten helper checks passed; the actual JAX/Orbax roundtrip remains skipped locally and must pass in the allocated runtime before complete-model recovery acceptance.

- Verified the update core on the actual pinned JAX/Flax CPU runtime: a 4-second, 12-check synthetic test confirms real parameter changes and exact full-state preservation on NaN-gradient rejection. Complete-model gradients and disk restoration remain pending.

- Added a reviewed full-parameter pi0.5 update core with finite-gradient rejection, stored RNG, AdamW and per-leaf diagnostics. Ten local software checks passed; one JAX behavior check is skipped, with an actual pinned-runtime CPU gate still required.

- Accepted complete pi0.5 forward execution on two real image samples using one H800. Retained the failed probe assertion and extra submission outside its one-job handoff scope; rejected unbound GPU-memory telemetry. Gradients and optimizer updates remain pending.

- Added the reviewed forward-only diagnostic core (6 software checks); frozen a single-H800, five-minute real-image forward contract with zero backward passes or optimizer updates.

- Accepted official pi0.5 Observation transforms on two real training samples in a 30-second CPU allocation: 83 recorded checks plus independent pixel/state/action/mask and decoded-state parity. Model forward and optimizer checks remain separate.

- Added the official pi0.5 Observation adapter with local verified tokenizer initialization, 16-state tokenization before 32-channel padding, explicit loss mask and no second normalization. Independent 22 software tests passed; real upstream transform execution is the next gate.

- Verified complete pretrained pi0.5 loading on one H800: 3,353,433,872 FP32 parameters, strict tree restoration, 23.914 seconds load/count and a 26-second allocation. Recorded host RSS separately from unknown peak GPU memory; image forward, gradients and optimizer updates remain pending.

- Independently accepted the two-real-sample pi0.5 input pack; retained the first 54-second backend-initialization failure, requested/allocated CPU discrepancy, and a CUDA-only correction verified by 11 independent checks. Official tokenizer asset metadata/size/SHA256 were also verified. No full-model load or optimizer result is claimed.

- Accepted a strict complete-model pi0.5 loader after independent actual-report and portable CI admission checks; delegated one bounded H800 load-only job, with optimization still separate.

- Accepted source-bound train-only pi0.5 normalization and pure numeric valid-16 loss checks; integrated 72 passed/one private-fixture skip. Retained OpenPI attribution/license. Full-model runner timeout and tokenizer transport failure remain separate unaccepted steps.

- Froze the native JAX full-model smoke contract, retained preparation failures and scoped corrections, and independently found that twelve action channels are constant in the current teacher corpus. No pi0.5 model execution is accepted.

- Accepted native pi0.5 input repacking after 28 tests and six real-sample comparisons; preserved absent sensors and unapplied 16-of-32 loss mask. Audited 12 resized screenshots and recorded cursor detail loss before precision-control use.

- Verified isolated pinned OpenPI source and dependency staging from actual installer output and final package metadata. Base weight staging is delegated; pretrained model loading and training remain unverified.

- Accepted the pure native-to-coarse control resampling helper after 22 tests and six real-label interpolation checks; complete temporal assembly and physical equivalence remain unverified.

- Audited existing runtime metadata and pinned OpenPI dependency/weight inventories; delegated isolated setup. Recorded the temporal-module timeout and separated native-time model smoke from longer-window execution adaptation.

- Accepted the native RGB visual-policy corpus: full-resolution pixels, separated action/provenance, immutable index/image bindings and native time semantics. Independent 23 tests and 12 real-image comparisons passed; pretrained model training remains pending.

- Corrected model priority after user review: retain the coordinate-only MLP as a fixed reference; advance full pretrained pi0.5 adaptation and a visual temporal ACT-style comparison.

- Independently audited the fine-tuned candidate: diagonal 1/7 attempted, axial 0/2, one case unattempted due budget. Retained original policy; recorded contact, unintended click feedback, stale-frame and horizon failures.

- 2026-11-01: Accepted 500-update full fine-tuning from the existing small policy, exact original normalization and checkpoint restore. Recorded both diagonal offline improvement and old axial regression; candidate promotion waits for closed-loop evaluation.

- Independently accepted eight fresh diagonal teacher episodes and 550 samples (254/145/151), with 606 raw files unchanged. Issued a bounded 500-update full-network fine-tuning permit from the existing model; teacher success remains distinct from learned-policy evaluation.

- Accepted the diagonal collection CLI after 72 tests and 10 independent checks, preserving legacy index output and truthful rejected/missing episode reporting. Dispatched fresh collection and candidate export; training remains gated on actual data review.

- Accepted whole-network fine-tuning from the existing policy with frozen input normalization, step-zero best selection and checkpoint provenance/restore checks. Independent 127 tests passed with four conditional skips; GPU execution is still pending.

- Saved and independently reviewed the initial diagonal CLI and whole-network fine-tuning changes. Retained both worker timeouts and failed checks; focused corrections are in progress, with no new optimizer run.

- Accepted diagonal collection identity and profile-aware policy loading with 141 tests, 2 conditional skips and 26 independent checks; all 674 original feature vectors remain identical. Recorded user direction to fine-tune the existing policy with its original normalizer. CLI and warm-start training integration remain pending.

- Preserved two CLI continuation timeouts with no source changes; event logs show pending model responses, with cause unresolved. Accepted teacher-configuration source passed all four CI Python versions.

- Integrated strict teacher-configuration agreement and capture-source metadata into diagonal episode auditing; 127 independent tests and 11 integration checks passed. Legacy outputs, 50D actor input and 25×16 executed labels preserved. CLI and training-loader adaptation remain pending.

- Integrated per-frame clock validation into episode auditing, including terminal frames and accurate timing coverage. Fixed repo-relative PNG resolution and cross-episode symlink acceptance; 94 independent tests and a read-only 45-frame diagnostic audit passed. No new training index or optimizer run.

- Added profile/collection-scoped data export and a capture timing validator, with 82 independent tests plus saved-trace and legacy-parity checks. Complete CLI/configuration/training-loader wiring remains pending.

- Added optional task-profile-aware split mapping with 50 passing focused/regression tests; legacy axial mapping unchanged. Exporter wiring and fresh dataset acceptance remain pending. Updated the README to report the completed diagonal evaluation and distinguish the new teacher diagnostic from learned-policy results.

- Accepted one actual Pillow teacher trial:18.143 to0.917px,45fresh frames,peak0.942N,safe release/exit; decode median3.093ms, maximum age+uncertainty42.854ms. Preserved prior failures and learned2/8baseline.

- Integrated explicit teacher PNG backend selection, early dependency checks and backend/clock metadata. Independent106tests plus selector/clock audits passed; dispatched one seed2runtime diagnostic with unchanged limits.

- Added per-capture monotonic stage timings and clock provenance, independently checked with106focused tests and fresh/stale fake clocks. A300ms frame causes zero action with the unchanged250ms gate.

- Added explicit optional Pillow decoding with shared bounded PNG validation, unchanged stdlib default and no silent fallback. Independent59tests and six saved real-image parity checks passed; teacher integration remains separate.

- Updated follow-up cadence to45minutes as requested; preserved a240.1s decoder-source timeout and scoped test completion. Saved code remains unaccepted until independent checks.

- Accepted exact pixel/shape/dtype and extractor parity for six saved PNGs using Pillow12.0.0; same-batch median decode79.656→2.032ms. Dispatched an optional backend preserving strict validation; live pipeline unchanged.

- Verified saved-PNG decode timing (median78.979ms, maximum350.226ms), retained missing live causal evidence, and dispatched a bounded Pillow parity/performance measurement. No freshness threshold or policy behavior changed.

- Audited the remaining three lowtarget directions:two passed,one retained a322ms stale-frame rejection with no subsequent action. Combined teacher result3/4,success-only mean1.102px,all force/release checks passed; no learned-policy improvement claim. Dispatched offline decoder/timing comparison without relaxing freshness limits.

- Independently accepted one lowtarget teacher trial:18.143→1.852px,last three frames within3px,peak0.942N,no native press-threshold crossing,safe release/exit. Preserved single-case and teacher-only limits; remaining three directions precede any new training data.

- Accepted Kimi2.8 offline statistics after independent recomputation:902/3000zero gates,1.2987%mean tangential gate,no recorded bound saturation. Accepted an opt-in0.6N target candidate after47independentstub checks and dispatched one seed3trial; physical outcome remains pending.

- Switched shared Kimi handoffs to K2.8 Preview after user confirmation, three routing checks and a live read-only smoke. Recorded API-price proxies separately from membership billing; existing provider and billing controls remain unchanged.

- Independently audited the one pointing-light teacher trial: no1.0N pressure crossing, safe release/exit, but120cycles exhausted without pointing success. Preserved missing terminal capture and excluded the failed data from training; resumed bounded offline diagnosis with Kimi2.8.

- Added the opt-in pointing-light teacher mode with0.9N tangential-stop threshold and unchanged0.8N normal target. Standard mode remains unchanged. Independently passed41stub checks; a single source-bound trial is dispatched, with click-free checks covering control and cleanup, not just terminal pointing error.

- Independently audited four diagonal teacher trials:3met the original pointing criterion;1stopped after click feedback turned the target green. All four remained physically safe and released/exited. Preserved the failure and the distinction between pointing accuracy and click-free interaction; scoped a separate light-contact candidate without changing GUI or safety thresholds.

- Added bounded near_diagonal_v1 selection to the teacher runner while preserving defaults and all physical/success limits. Independently passed34stub checks and froze17source hashes plus the app/assets before dispatching four teacher-feasibility cases. Teacher outcomes remain separate from learned-policy results.

- Repaired the private offline diagnostic with JSON-only output and complete zero-contact force samples. Independently verified all eight contact summaries and453train-row feature ranges; unvalidated distance/causal claims remain excluded. Scoped four-direction teacher feasibility behind a named-profile source gate; no new optimizer.

- Preserved two offline-diagnosis timeouts and the saved analysis source. Independent offline execution produced partial JSON then hit a report-field error; contact missingness and feature-distance semantics also remain unaccepted. No new physics or optimizer was run. Named-suite integration CI passed on Python3.10–3.13.

- Independently reviewed all eight diagonal trials: 2 successes,4 penetration stops,2 contact losses. Preserved every failure and distinguished pre-abort observations from terminal error. Actual batch took58seconds; Kimi dispatch took310.6seconds. At the user's request, validation now defaults to12conditions×2resets with time/failure stopping rules;90is optional. Offline diagnosis precedes new data/training.

- Accepted named axis/diagonal evaluation suites after67focused checks; bound actual app bytes and native sources before execution. Preserved timeout, environment-resolution and fixture failures. The initially proposed90-episode comparison was superseded by the staged protocol above before any such trial ran.

- Added the fixed eight-target diagonal GUI profile while preserving default and axis profiles. Kimi passed 12 actual native protocol checks; coordinator verified source, signed app, preserved baseline app and three saved PNGs. Learned diagonal evaluation remains separately gated.

- Independently accepted the first learned physics/GUI trials: 2/2 fixed held-out near targets, final errors0.874/2.812px, 67 unassisted learned action blocks, complete contact/cursor evidence, safe release and app exit. Preserved narrow coverage, source/cache provenance and all failure denominators; dispatched a separate diagonal GUI test-profile module before broader evaluation.

- Added the frozen learned-policy physics/GUI evaluator. Independently passed 42 focused tests after correcting incomplete and mismatched stubs; preserved source/checkpoint binding, exact control anchors, pre-action logging and terminal safety. Two fixed held-out trials are dispatched separately, with no optimizer or teacher rescue.

- Accepted the first H800 small-policy run: 2,000 updates, exact checkpoint next-step comparison, verified NPZ export and independent held-out action-error recomputation. Preserved the failed first deployment and summary hash-order correction; learned physical/GUI evaluation remains pending.

- Preserved the first small-policy H800 deployment failure (zero optimizer updates) and the masked scheduler exit; required complete frozen-package staging, local import preflight and truthful exit propagation before a separate bounded retry. Source integration CI passed on Python 3.10–3.13.

- Added the source-reviewed small MLP trainer and NumPy deployment path, source/data permits, real disk checkpoint comparison and bounded H800 guards. Independently passed 81 local checks with 1 GPU skip. Fixed CI’s missing NumPy dev dependency. Actual optimizer, restore and GPU parity remain unrun until the bounded job.

- Accepted the exact 674-sample teacher pilot index after 84 auditor checks and independent raw/image/action comparison. Preserved 758 raw file hashes, whole-target splits and failed exports; corrected aggregated-finger counting and complete hash checks. Training and learned task success remain separate.

- Added a local LEAP v1 Full live physics viewer, independently inspected in the browser. Corrected displayed-frame freshness and narrow-panel fit; preserved timeout, restart and regression-test evidence. Training-attached monitoring remains a separate next stage.

## Unreleased — 2026-10-31

- Verified the NumPy adapter on all 674 actual candidate samples and 33 focused tests; corrected cumulative-label semantics and normalization. A saved 29-action episode now passes terminal audit, while 4 synthetic fixture failures and trainer restore/provenance defects remain under bounded repair. Preserved both timeout receipts; no optimizer has started.

- Preserved auditor and small-policy implementation timeouts plus independent failing tests. Reproduced real-index metadata/label incompatibility, kept unaccepted source private, and split correction work into two disjoint bounded Kimi tasks. No optimizer or new GPU job was started.

- Independently verified the complete12-episode raw pilot:47,450physics steps,686images,674action groups and1,214touch messages. All12 fixed-target teacher episodes passed. Rejected the first reusable auditor despite25passing tests after reproducing7false-acceptance counterexamples; preserved it and delegated bounded repairs. Added a gated small-policy BC/checkpoint/independent-evaluation contract.

- Collected12 fresh fixed-target pilot episodes after four-direction teacher acceptance. Independently checked source hashes, row counts, raw byte inventory and exited application identities; full causal/pixel/data acceptance remains pending. Preserved prospective target-class splits and raw ineligibility, and corrected the collection report's byte-unit/wall-time interpretation.

- Added a force-governed teacher mode that distributes planar increments over25substeps and preserves the normal correction before bounding tangential motion. Independently passed121source checks and4000pure randomized boundary cases; retained previous modes and fixed safety criteria. One actual seed2 episode then passed: error24.32→1.67px,2700causal visual substeps,5.4scontact,0.714mmmaximum penetration and measured safe release. Three additional directions dispatched; no data/training acceptance inferred.

- Added privileged per-substep force admittance to the visual teacher, with prior-record causality and actual-command logging. Independently passed 107 pure/static/mock checks. Preserved two timeouts and a separately stopped attempt; fixed a floating-point envelope validation defect and corrected a historical source line count. One actual force-feedback episode then failed the unchanged 1 mm penetration limit after136visualsteps. All136 corrected commands were executed;101 reached the normal-speed cap. Preserved trace and delegated static dynamics diagnosis before another controller change.

- Added a normal-priority visual teacher with common-vector bound scaling and a per-substep contact-loss stop. Independently passed 83 pure/static/mock checks; preserved the legacy DLS option and failed episode. The actual diagnostic then failed the unchanged 1 mm penetration limit despite maintaining contact for 302 visual substeps; preserved the failed trace and stopped all physics at the first breach. Reviewed four saved-state reconstructions after a diagnostic timeout, identifying commanded/actual motion mismatch; delegated a separate bounded normal-force feedback helper before further dynamics.

- Executed and independently reviewed the first actual near-target visual robot episode:120 frame/control groups,5550 physics substeps,165 actual touch messages. Precision failed after contact loss (24.32→14.89px), with safe final clearance and app exit. Preserved raw evidence, diagnosed the normal-motion error and delegated a constrained teacher correction; no unchanged retry or learning claim.

- Integrated a source-reviewed fixed-parameter near-pointing probe with65 independently passing input/physics/probe checks. Fixed finalization, input acknowledgement, release timing and evidence-label defects, preserving two timeouts and the original counterexamples. Froze source/model/native hashes and delegated one bounded actual episode; its outcome requires separate runtime review.

- Recorded a600s generic visual-runner timeout with no source delivered. Delegated a smaller fixed-parameter experiment script in a fresh Kimi session, retaining per-step evidence and external source-bound execution review. No robot episode or learning result is claimed.

- Integrated the reviewed P1 scene/contact library and per-substep physics core, with17 static/stub checks and52 integrated input/physics checks. Added the confined native near-pointing reset profile,54 executed native tests and independent saved-image review. Preserved failed attempts and kept actual visual robot episodes/data/training pending.

- Integrated reviewed visual input utilities: complete PNG stream validation, fixed-palette localisation, clock uncertainty and frame checks. Independently passed35 focused tests and verified cursor displacement in two saved native PNGs. Recorded the measured decode cost and kept controller/data/training claims separate.

- Preserved three visual-implementation timeouts and their available usage, checkpointed incomplete source privately, and split recovery into fresh, disjoint Kimi utility and physics-core tasks. No completed controller or learning result is claimed.

- Verified the new P1 live producer, advancing actual images, process identities and per-step contact telemetry. Recorded its two-hour deadline and preserved the previous stream as history; it remains a scripted diagnostic.

- Reviewed the first LEAP v1 Full continuous-contact profile (P1): 8.994 mm actual fingertip sliding, 2.558 s continuous contact, 1.583 N peak and 0.734 mm penetration; repeated and 1 ms checks passed. Preserved all failed candidates, cumulative-budget correction and uncalibrated limits. Visual controller remains under review; no learning result.

- Restored a bounded LEAP v1 Full hand-and-pad diagnostic stream and verified actual motion in the browser. Kept DEMO labels and unresolved contact losses explicit. Enabled periodic project continuation, documented bounded resources, and resumed Kimi contact-stability diagnostics; no policy training claim.

- Added the reviewed bounded frame publisher and loopback training-monitor viewer, with monotonic identities, PNG verification, displayed-frame freshness and terminal shutdown. Inspected a finite TEST-source sequence in the browser; no actual training source attached. Preserved failed v1 contact diagnostics and prepared the next contact-stability investigation.

- Fixed Retina capture scaling in the confined native GUI: actual PNGs are normalized to 800×600 and cursor pixel movement matches logical coordinates. Independently passed 48 native tests and inspected two real app images; no host desktop capture or physical robot operation.

- Verified a separately installed LEAP v1 H800 runtime at 1/32 worlds and a PyTorch BF16 diagnostic update; published source/hash, sampling and budget limits. The one-GPU job completed in 150 seconds. Recorded automatic expiry of the earlier hand-only live demo. No project policy training or 8192-world acceptance.

- Integrated Kimi's independent LEAP v1 Full hand-only baseline and 14-file pinned asset manifest. Reviewed 16-channel motion over 11,000 steps, actual renders and raw traces; retained self-contact and tracking limitations. Corrected eager optional-dependency imports after independent review; 19 focused worker checks passed. V1 pad contact, online feedback and training remain pending.

- Confirmed full fine-tuning on allocated H800/other compute and simulation-first validation, with physical FSR/conductive-fingertip checks deferred. Documented the current physics-contact-to-device pipeline, idealized recognition assumptions and missing gesture support.

- Recorded the user's final LEAP v1 Full hardware choice (four fingers / 16 XC330 motors), preserving all v2 measurements as v2. Added source-grounded pi0.5 empirical fine-tuning analysis, GR00T N1.7 / LingBot-VLA 2.0 architecture tradeoffs, optional FSR fingertip design and a prepared v1 rebaseline contract. No model migration, hardware modification, GPU allocation or training occurred.
- Retained RL as the user-confirmed conditional middle/later-stage path: use it for remaining performance limitations if demonstration learning, corrections and execution tuning are insufficient; omit it if task criteria are met.

- Integrated Kimi's executed LEAP contact-to-native-GUI replay: 105 frames accepted in each role, four actual window captures, verified press/release and cursor displacement. Independent source/hash/pixel review passed. Preserved actor/evaluator isolation, two prior dispatch timeouts and offline-only limitations; added the next online-feedback pilot contract.

- Validated the unchanged full LEAP hand on H800 at1/32/128/512 worlds:42/42 checks. Added the reproducible batch harness and exact backend-representation reset audit, preserving the failed first reset-hash attempt and sampled-coverage/timing limitations.

- Added the reviewed LEAP static-pad contact cycle, continuous contact recording and thirteen focused tests. Recorded the failed initial mounting/force accounting and the first H800 reset-audit failure separately; no GUI or learning success is inferred.
- Integrated Kimi's LEAP v2 Advanced CPU adapter and pinned asset fetcher. Thirty-four focused tests and a 14,000-step scripted hand run passed; fixed false CAD-hull assembly contacts, preserved 17-motor/21-joint coupling, improved pose tracking, corrected observation timestamps, rejected invalid commands atomically and protected run artifacts. Published failed candidates, narrow acceptance limits and delegation/rework accounting. No LEAP pad/GUI, GPU or learning result is claimed.

- Updated the research target to the latest available LEAP v2 Advanced full configuration, following the user's preference over the earlier v1 selection. Audited official URDF/STL/CAD sources, 17-motor/21-joint coupling, inertial-marker discrepancies and Metal calibration limits. Added MacBook/external-Trackpad hardware-readiness boundaries; dispatched Kimi's real CPU import and preserved its initial 600s source-only timeout. Corrected stale native-GUI pad dimensions in documentation to match the existing implementation.

- Selected LEAP v1 Full as the single new research-hand target, with Allegro retained until import acceptance and future G1 DFX/Revo2 options separated from current simulation. Added model/motor/license qualifications and a prepared Kimi import contract; focused the research plan on visual decisions, action chunks and long-horizon memory. No new model import, GPU job, training or purchase occurred in this selection step.

- Verified the corrected full-hand H800 teacher replay: 43/43 checks, 1/32/128/512 identical worlds, unchanged reset/model/control, actual completed allocation and full-cycle timing. Published sanitized results and independent review; retained sampled-contact limits, unknown solver warnings, remaining failure-gate work and the later Kimi CLI timeout. No training or production source integration is claimed.

- Added exact humanoid/hand compatibility evidence, including official G1 support, H2 dealer-only configurations, Dex5-S power/physical-layer constraints, and wheel-based Linker platform distinctions. Kept hardware fit separate from policy transfer and real acceptance.

- Preserved a600-second Kimi correction timeout with saved viewer/contact source, no deployment acceptance and no second GPU job; confirmed the scheduler queue is empty.

- Expanded hand pricing beyond official pages with 37 price records, domestic interviews, retail configurations, historical costs and public used-market coverage. Added RMB reference conversions and practical budget tiers; preserved original research and unresolved stock/unit differences.
- Recorded the first full-hand H800 contact-classification failure at 1/32 worlds and independent viewing of the corrected CPU source after manual reload; stale-image false-LIVE behavior returned to Kimi.

- Recorded independent review of one corrected CPU light-touch cycle, the live full-hand diagnostic stream, and actual guest fixture click/scroll pixels. Preserved timeouts, old failed-precision stream provenance and missing cursor/release acceptance; corrected viewer/H800 continuation remains Kimi-owned.

- Added a source-grounded dexterous-hand survey with exact hardware/model versions, humanoid integration boundaries, public price references and a LEAP v1/Tesollo S shortlist. Raw vendor sources remain private.
- Verified Kimi invocation recovery and preserved the subsequent contact/release review rejection: passive hold sag and 6.328 mm penetration remain unsuitable for training acceptance.

- Integrated Kimi's corrected sandboxed AppKit GUI, contact client and 47 real-app integration tests (136 total locally), plus 18 smoke checks. Preserved initial failures and timestamp corrections. Coordinator reviewed actual output images and source/receipt hashes; physics-to-GUI and learned tasks remain pending.
- Added explicit shared reset/control, 160 × 110 mm pad, immutable live-frame and staged GPU acceptance contracts. Preserved additional native-GUI and compute-harness review findings for Kimi correction before integration.
- User selected the M5 host and explicitly authorized a small macOS guest. Guest boot/desktop is verified; native GUI, full-hand dynamics, live stream, and full-hand H800 work remain separate acceptance stages.
- Clarified that Kimi owns operational execution as well as implementation; coordinator owns architecture, budgets, read-only review and Git/documentation integration.
- Preserved native-app review failures and incomplete CPU-model checkpoints; extended narrowly scoped Kimi continuations instead of treating timeout or initial tests as acceptance.
- Accepted architecture v2: articulated hand, visual closed-loop operation, multi-scale memory, and short action chunks.
- Established a dedicated public project boundary, reproducibility policy, and append-only experiment records.
- Preserved sanitized historical design snapshots and authoritative source links.
- Initially planned protocol/data foundations and a bounded H800 probe; execution and acceptance followed in the steps below.
- Published the initial public repository and verified its remote commit. Dispatched isolated Kimi foundation and GPU-probe work.
- Specified staged data acquisition budgets, provenance/quality gates, and a dedicated M1 Mac setup plan.
- Verified and integrated Kimi's single-H800 MuJoCo Warp probe: 12/12 checks, 1/32 worlds, exact executed-source and result hashes matched independently. Retained the initial device-API failure and correction.
- Corrected the probe's interpretation: batching increased GPU aggregate throughput, but no GPU-over-CPU crossover was measured in this tiny scene.
- Reviewed the initial foundation package (54 tests passed) and returned independently reproduced boundary gaps for correction before acceptance.
- Integrated the corrected protocol/recording foundation after 89 local tests, 9 independent boundary checks, and deterministic mock replay; 12 negative CLI cases are rejected.
- Bound actions to actuator layout and execution windows; enforced observation/evidence references, strict actor contracts, safe episode paths, mock-data exclusion, and explicit interrupted recording.
- Preserved the correction worker's 20-minute timeout and coordinator verification-harness correction. Coordinator completed documentation/provenance; production code remains Kimi's implementation.
- Added runnable setup instructions, protocol-integrity ADR, CI matrix for Python 3.10–3.13, and measured delegation/rework records. CI execution is tracked separately in STATUS.md.
- Verified public remote revision and the first GitHub Actions run: all four Python 3.10–3.13 jobs passed tests and deterministic CLI replay; appended the run receipt and source revision.

# ADR 0002: Protocol and evidence integrity

Date: 2026-10-30. Status: accepted design requirements; implementation acceptance is tracked separately in STATUS.md.

Independent review of the first M0 implementation found that its existing 54 tests passed while several important boundary cases remained accepted. Preserve this finding and the subsequent corrections in the experiment record.

## Decisions

1. An action chunk binds the exact actuator layout/order of its episode. Equal array width and equal embodiment name alone are insufficient. Zero-valued references remain valid; validity comes from an explicit mask.
2. Live execution validation and historical recording are different operations. Live admission checks intent issuance, clock identity, goal/observation identity, and a half-open action interval. A chunk cannot execute after `start + horizon * dt`, even if an optional external expiry is later. Cross-host monotonic timestamps require an explicit mapping rather than direct subtraction.
3. Observation and evidence references must resolve within the recorded episode. Intents/chunks refer to known observations; outcomes refer to known intents; success facts require recorded observation evidence. A nonempty invented frame ID is not sufficient. Multiple decisions may legitimately reuse the same observation; conflicting duplicate definitions are rejected.
4. Actor payloads follow strict known contracts, with no arbitrary unknown fields. Teacher/evaluator records remain separate. Filtering only suspicious field-name substrings is not a field whitelist and does not establish observation isolation.
5. Software fixtures are always marked `training_eligible=false`. Eligibility is a strict Boolean and is distinct from success: genuine failures can be valuable training data, while a synthetically fabricated success is still not a physics trajectory.
6. Episode output stays under its intended root; episode identifiers cannot be absolute paths, traversal, or platform-specific path separators.
7. Accepted events are flushed and partial/interrupted recordings remain recognizable. Closing a file or completing a control attempt does not establish GUI or task success. An exception-terminated context must retain its interrupted status.

## Consequences

Contract, JSON Schema, example, recorder, and test changes must remain consistent. Corrections are recorded with the initial findings instead of rewriting the first pass as if it had established acceptance.

These checks establish protocol/data integrity only. They do not validate a physical hand, verify the semantic contents of an image, create Mac native multitouch support, or demonstrate a trained task policy.

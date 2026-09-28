# SRS display alignment and remaining work

Reviewed against the supplied SonicSentinel AI SRS PDF. This is a display/workflow update, not a claim that every SRS requirement is complete.

| SRS-facing area | Current implementation |
|---|---|
| Upload and batch | Actual decoding/inference; format, duration and signal rejection; configured request size limit; per-file progress/errors |
| Recording metadata | Audio ID, filename, format, duration, original sample rate/channels/bit depth when available, size, UTC timestamp |
| Audio inspection | Original audio playback, full-clip waveform and spectrogram generated from stored bytes |
| Python results | Latest selected CNN; original candidate/score, top-three/all-ten scores, top-two margin, quality and decision note |
| Final result | Reviewed class if present, preserved original output, status, original model severity, alert state, advice and review state |
| GTM/comparison | Explicitly deferred at the user's request; no fabricated prediction, agreement or difference |
| History | Filename/ID/date/original-class/confidence/quality/severity/status filters, pagination; admin user-ID filter |
| Duplicate display | Exact uploaded-file SHA-256 match within the owner's history, linked to the prior event |
| Manual review | Playback, confirm/correct, required comment, optional advice, immutable original prediction fields and appended action timeline |
| Alerts | High/critical queue; role checks for acknowledge/dismiss/escalate; local action history |
| Dashboard | Saved upload counts, open/tracked alert count, pending review, poor quality and recent events |
| Model evaluation | Real test metrics for three Python models, selected-on-validation label, per-class precision/recall/F1/support and confusion image |
| Live monitoring | Existing actual microphone pipeline, measured visuals, class scores, confidence/quality, repeated-window confirmation and session-only history |
| Reports/export | User-scoped upload charts/filters/CSV, per-event JSON with corrections/actions, browser Print / Save PDF |
| Profile/settings | Editable display name; role and account details; admin-only threshold editing with CSRF protection |

## Scope limits that remain

- GTM integration is deliberately postponed. Independent dual-model comparisons cannot be marked passed.
- Accuracy/recall targets remain unmet; the trained weights and measurements are unchanged. Full SRS verified: false.
- Real microphone windows are not persistently logged; historical dashboards and reports cover uploaded clips only.
- One highest-energy model window does not provide exhaustive multi-event timeline/overlapping-window detection for longer uploads.
- Exact duplicate matching does not implement acoustic near-duplicate detection.
- New analysis records have detailed evidence; pre-update records only contain their original saved columns. Do not invent missing metadata or probability vectors.
- Review correction does not retrain the model, invent a new confidence, or automatically recalculate original severity. The display labels original model severity explicitly.
- Actions are recorded locally. There are no email/SMS/push notification deliveries, shared operator assignment queues or escalation delivery guarantees.
- Administrators have all-event access; other roles are owner-scoped. Fine-grained per-category operator assignments, full account management and organization-wide report aggregation remain pending.
- The action timeline covers review/alert actions. It is not a complete immutable audit log of every access or settings change.
- Retention automation, user-data deletion workflows, privacy-policy configuration and production deployment hardening are not implemented by this UI update.
- Thresholds are development rules, not validation-calibrated safety guarantees. Settings changes affect new predictions only.
- Noise robustness and split artifacts remain in the exported reports; this page does not re-run training/evaluation or produce new measured accuracy from unlabelled uploads.
- Full desktop OS/browser compatibility, physical microphone behavior and the local MySQL installation need verification on the user's machine.

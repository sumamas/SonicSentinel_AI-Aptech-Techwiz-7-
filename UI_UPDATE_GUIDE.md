# Modern workspace UI — 28 September 2026

## Run this update

Extract the ZIP into a new folder. Start XAMPP MySQL (default port 3307), check `.env`, run `setup_project_test.bat`, then `start_project.bat`.

**Important:** this update introduces two additional tables: `event_analyses` and `event_actions`. Run setup once to create them. Existing event/user data is not reset. If the environment is already installed, use `.venv\Scripts\python.exe scripts\init_db.py` before starting the app.

## What changed

Upload, Batch Upload, Event History, Alerts, Manual Review, Profile, Settings and Model Evaluation now share a responsive teal/white workspace style, navigation, cards, forms, status badges and footer. An overview and a complete per-recording result page have been added.

After analysis, Upload opens the saved result automatically. Batch processing shows each file's progress, actual result or error, with a result link. Stop finishes the current request and leaves remaining files unprocessed. Starting again submits the selected files again; exact duplicates are flagged, not silently removed.

## Result display

- Original class and original confidence remain preserved after a human correction.
- Final class, current status, original severity, alert state and recommended action are visible.
- Top-three predictions, all ten class scores and the top-two margin use actual inference output.
- Audio playback includes native seek and volume controls where the browser supports the format.
- Waveform and spectrogram are generated from the stored recording, not placeholder graphics.
- Metadata includes original sample rate/channel count/bit depth when the decoder can read them. Compressed or older records may show unavailable fields.
- The model uses its unchanged v5 selected 3-second window; displayed visuals cover the full accepted clip.
- Save JSON downloads the evidence and action history. Print / Save PDF uses the browser dialog.

## Review and alert actions

Audio Reviewers and Administrators can confirm/correct a result, add a comment and record advice. Authorized operators can acknowledge/dismiss/escalate high or critical severity records. Actions are appended to the event timeline with actor ID and UTC time. Escalation is a local record; external messaging is not configured.

New accounts remain normal users. Use the local role-management command in README.md to grant your registered test account an appropriate role. Non-admin operators are scoped to their own recordings; administrators can inspect all events. Unauthenticated or unrelated-user requests cannot retrieve another user's audio, analysis JSON or visuals.

## Existing Live Monitoring and Reports

The supplied designs remain integrated. Live Start/Pause/Resume/Stop, device selection, measured waveforms/spectrograms and repeated-window confirmation remain connected to the latest CNN. Live class score details now include all ten probabilities and top-two margin. Live windows are session-only.

Reports use saved uploads owned by the current account. Date filters use UTC. Charts, category filters and CSV use database events; test model metrics are independent of these filters. Confidence is not measured accuracy.

## Deferred and unchanged

Google Teachable Machine stays pending. No agreement or confidence difference is fabricated. Model weights, source-contract hashes, label order, package pins and exported test metrics are unchanged. This UI update does not certify full SRS compliance. See SRS_UI_ALIGNMENT.md for remaining work.

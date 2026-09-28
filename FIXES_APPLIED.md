# Fixes applied — 28 September 2026

## 1. "The analysis could not be saved. Check MySQL and run scripts/init_db.py"
Reproduced with the original code on MariaDB 10.11. Two causes:
* The old `database/sonicsentinel.sql` created only `users` and `audio_events`; the code also needs `event_analyses` and `event_actions`.
* When XAMPP MySQL is stopped or on another port, every save fails with the same message.

Fix (`database/bootstrap.py`, `app.py`, `config.py`): on start-up the app creates the database, every table and any missing column. If MySQL is unreachable it switches to `instance/sonicsentinel.db` (SQLite) and shows the active database in the page header. JSON snapshots are stored as text (`models_db/json_type.py`), so any MySQL/MariaDB version works. Save errors now show the real database reason. Batch upload used the same endpoint and is fixed by the same change.

## 2. Results always "Uncertain"
`src/ml/predict.py` added the review reason `clip_exceeds_model_window` to every clip longer than 3 s, so even a 99 % Glass Breaking prediction became "Uncertain". It is now an informational note. A quiet Background Noise result is no longer sent to review.

## 3. Teachable Machine integration
`src/ml/gtm.py` loads the exported TF.js files and runs the network with NumPy. Features reproduce the Teachable Machine browser pipeline: 44.1 kHz, 1024-point Blackman-windowed FFT frames, dB magnitude, first 232 bins, 43 frames (~1 s), per-example normalisation. Label names with stray spaces in `metadata.json` (e.g. `"Gunshot  "`) are mapped to the SRS names. The 3-second window is covered by ~1 s windows averaged by loudness. Zero padding is removed before GTM so digital silence does not distort normalisation.

## 4. Comparison and final decision (`src/ml/comparison.py`, `config/model_policy.json`)
* Confidence difference = |Python top-class confidence − GTM top-class confidence| (SRS formula).
* Acceptable Match: same class, difference ≤ 35 %, both ≥ 30 %. Weak Match: same class with larger gap, or each model's answer is the other's runner-up. Model Disagreement otherwise. Uncertain Result when the final confidence is below 35 %.
* Final scores = 0.65 × Python + 0.35 × GTM (configurable in Settings). Raw outputs of both models are stored and displayed unchanged.
* Manual review: low confidence, small top-two margin, poor quality, overlapping sounds, model disagreement, unknown pattern.
* Alerts: High/Critical class with ≥ 80 % confidence and ≥ 20 % margin; 65 % is enough when both models predict the same class (SRS Step 15: model agreement is a confirmation method). Live alerts also need two matching windows.

## 5. Measured behaviour of the supplied GTM model (honest note)
On the 56 gunshot/glass clips in `data/starter_3class`, the GTM model's top-1 was correct for about 50 % and its top-2 for about 82 %. It often confuses glass breaking with "Person Asking for Help". The Python CNN is much stronger on these clips. To make the two models agree more often, retrain the GTM project with more samples per class taken from the same training recordings (SRS Step 9), especially Glass Breaking and Person Asking for Help, then replace the three files in `gtm_model/`.

## 6. Pages
New modern design (`static/css/modern.css`) for Analyze (drag and drop, progress steps, inline result), Batch upload (queue, summary counts, Python/GTM/agreement per file), Event history (quick filters, agreement filter), Alerts (Open / Pending review / Handled with inline actions), Manual review (inline player and review form), Recording analysis (side-by-side scores for all 10 classes), About and How it works. Live monitoring shows the GTM result and agreement for every window.

## 7. Setup
`run_setup.bat` installs everything, prepares the database, creates default accounts and runs `scripts/check_setup.py`. `start_project.bat` starts the server and opens the browser.

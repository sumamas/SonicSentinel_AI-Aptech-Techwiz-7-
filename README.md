# SonicSentinel AI

Sound-event detection with **two independent models**: a custom Python CNN and a Google Teachable Machine (GTM) audio model, compared on the same audio window.

## Quick start (Windows)

1. Install **Python 3.11 (64-bit)** and tick "Add python.exe to PATH".
2. Optional: start **XAMPP MySQL** (port 3307). If MySQL is not running, the app automatically uses a local SQLite database (`instance/sonicsentinel.db`).
3. Double-click **`run_setup.bat`**. It creates `.venv`, installs packages, creates the database/tables/columns, creates the default accounts and runs a self-test of both models.
4. Next time, just double-click **`start_project.bat`** and open http://127.0.0.1:5000.

| Role | Email | Password |
|---|---|---|
| Administrator | admin@sonicsentinel.local | Admin@12345 |
| Audio reviewer | reviewer@sonicsentinel.local | Review@12345 |
| Security operator | security@sonicsentinel.local | Security@12345 |
| Normal user | user@sonicsentinel.local | User@12345 |

Change these passwords before any public deployment. Run tests with `.venv\Scripts\python.exe -m pytest -q tests`.

## Release: dual-model update (28 Sep 2026)

See **FIXES_APPLIED.md** for the full list. Summary:

- **Teachable Machine connected.** The exported model (`gtm_model/model.json`, `weights.bin`, `metadata.json`, project https://teachablemachine.withgoogle.com/models/wziUFlrWa/) runs in Python (`src/ml/gtm.py`) on the same 3-second window the Python model uses. It never receives the Python prediction.
- **Comparison and final decision** (`src/ml/comparison.py`): class match, |Python top − GTM top| confidence difference, top-two margins, status (Acceptable Match / Weak Match / Model Disagreement / Uncertain Result) and a documented weighted final decision. Raw model outputs are stored unchanged.
- **Fixed:** clips longer than 3 s were always forced to "Uncertain / Manual Review".
- **Fixed:** "The analysis could not be saved" — missing tables/columns are now created automatically, and SQLite is used when MySQL is unreachable.
- New/redesigned pages: Analyze, Batch upload, Event history, Alerts, Manual review, Recording analysis, About, How it works; Live monitoring now shows GTM and agreement.

---

## Previous release notes

# SonicSentinel — SRS display update

The latest trained **10-class Custom CNN** is connected. This release modernizes the remaining workspace pages and adds saved analysis details and review/action history. Google Teachable Machine remains deferred, as requested.

Start with **START_PROJECT_TEST.md**. For this release, read **UI_UPDATE_GUIDE.md** and **SRS_UI_ALIGNMENT.md**.

## Start on Windows

1. Extract into a new folder; keep your previous project backup.
2. Start XAMPP MySQL. The existing default port is **3307**.
3. Check your local database credentials in `.env`.
4. Run `setup_project_test.bat` once. This installs the pinned environment, creates any missing tables and checks the trained model.
5. Run `start_project.bat` and open http://127.0.0.1:5000.

Use Python 3.11. Do not copy an older `.venv` or replace the included `src`/`models` folders.
The model is already trained; no Colab run or retraining is required.

## Database upgrade

This release adds `event_analyses` and `event_actions`. Run the setup script even if you already have a database, or run `.venv\Scripts\python.exe scripts\init_db.py` from the project folder.
The existing `users` and `audio_events` columns are unchanged. `create_all` adds missing tables without resetting existing rows. Existing recordings remain visible; detailed scores that were never stored cannot be reconstructed retroactively.

## Pages

- Home and existing sign-in/register pages.
- Workspace overview, accessible from the account menu and workspace tabs.
- Audio upload and batch processing.
- Recording analysis: original/final class, confidence, top-three and all ten scores, decision margin, quality, severity, alert state and recommended action.
- Native audio playback, waveform, spectrogram, original metadata where available, UTC timestamps and model fingerprint.
- Searchable event history, review queue, high/critical severity queue and action timeline.
- Model evaluation with the real exported test metrics and confusion matrices.
- Live Monitoring and Reports using the supplied designs, with the shared footer.
- Profile and role-protected decision threshold settings.

Reports contain saved uploads only. Live windows are temporary and are not inserted into the database. CSV downloads contain filtered original model event rows; per-record JSON includes human corrections and actions. Print / Save PDF uses the browser print dialog.

## Account roles

Registration creates a `normal_user`. To test review and settings using your own local account, register first, then have the project administrator run:

```bat
.venv\Scripts\python.exe scripts\set_user_role.py --email "your-email@example.com" --role administrator
```

Sign out and sign in again. Supported roles: `normal_user`, `audio_reviewer`, `security_operator`, `maintenance_operator`, `administrator`.
Normal users and non-admin operators see their own recordings; administrators can inspect all users' event records. The Reports page intentionally remains scoped to the signed-in account. Full shared operator assignments and a user-management UI are not implemented.

## Measured model performance

| Python model | Test accuracy | Macro F1 |
|---|---:|---:|
| Custom CNN (selected on validation) | 74.96% | 0.6727 |
| Random Forest | 77.67% | 0.7135 |
| SVM | 76.71% | 0.6566 |

The UI update does not improve these metrics. Full SRS verification remains false. In particular, selected-model recall for Panic Scream is 34.48%, Glass Breaking 76.09%, and Person Asking for Help 77.08%, below the 85% critical recall target. See the Model Evaluation page for all classes.

## Development and evidence

- `SRS_UI_ALIGNMENT.md`: implemented display features and remaining gaps.
- `VALIDATION_UI.md`: checks performed and environment limitations.
- `AI_USAGE.md`: AI-assisted development record; student review remains required.
- `ML_TRAINING_GUIDE.md`: existing training workflow, only if you later choose to retrain.
- `reports/model_evaluation/`: original exported metrics, confusion matrices and supporting evaluation artifacts.
- `scripts/check_project_model.py`: installed-model and optional database/audio checks.
- `tests/test_srs_portal.py`: isolated SQLite functional regression checks with the real model.

Bootstrap, Bootstrap Icons and Chart.js are bundled locally. External web fonts have system fallbacks. This is a local development prototype, not a hosted or certified emergency-response system.

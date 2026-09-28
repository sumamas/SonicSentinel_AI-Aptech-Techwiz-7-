# AI-assisted development declaration

## 28 September 2026 — Anthropic Claude (claude.ai)

Assistance requested: fix database save failures and batch upload, integrate the team's exported Google Teachable Machine audio model, implement Python-vs-GTM comparison and final decision rules, fix long clips always being marked uncertain, redesign Event History / Alerts / Manual Review / Analyze / Batch pages, add About and How-it-works pages, and add one-click setup scripts.

Files affected: `src/ml/gtm.py` (new), `src/ml/comparison.py` (new), `services/analysis.py` (new), `database/bootstrap.py` (new), `models_db/json_type.py` (new), `routes/api.py` (rewritten), `src/ml/predict.py`, `src/ml/provenance.py`, `app.py`, `config.py`, `models_db/*.py`, `routes/workspace.py`, `routes/main.py`, `scripts/init_db.py`, `scripts/check_setup.py` (new), `config/model_policy.json`, templates, `static/css/modern.css` (new), `static/js/portal-upload.js`, `static/js/portal-batch.js`, `static/js/live_monitoring.js`, `run_setup.bat`, `start_project.bat`, `tests/test_dual_model.py` (new).

Not changed: trained model weights, `src/audio/preprocess.py`, `decode.py`, `features.py`, `training_config.py` (the hashed preprocessing contract), the GTM model files, and the dataset. No classification is produced by a generative-AI API; both predictions come from the team's trained models. No confidence value is invented or altered.

Testing done in the development environment: 25 automated tests (pytest), a MariaDB 10.11 run on the old 2-table schema (tables and columns created automatically, analysis saved), SQLite fallback, browser checks of the new pages, upload and batch flows.

Student/team verification: pending. Each member must read and be able to explain `gtm.py` (TM feature extraction and NumPy forward pass), `comparison.py` (rules) and `analysis.py`, re-test on Windows with XAMPP and a real microphone, and list their own modifications here.


## 28 September 2026 — OpenAI ChatGPT / Codex

Assistance: integrating the supplied latest ten-class model export into the Flask project; adapting supplied Live Monitoring, Reports and footer UI references; interpreting the uploaded SRS for display requirements; implementing responsive workspace templates and associated event storage, review/action routes, visualizations and tests.

Main affected areas: `templates/`, `static/css/`, `static/js/`, `static/vendor/`, `services/event_display.py`, `services/reporting.py`, `routes/api.py`, `routes/workspace.py`, `models_db/event_analysis.py`, `models_db/audio_event.py`, `app.py`, setup/role/check scripts and release documentation. Latest exported model assets and their associated preprocessing were integrated during the preceding update. This display update did not retrain or modify the trained model weights or hashed preprocessing files.

Verification by the automated development environment: actual CNN WAV/MP3 and live-window inference, upload persistence, detailed-score storage, protected event access, review preservation of original predictions, action history, threshold guards, invalid audio handling and browser UI checks. See VALIDATION_UI.md for the exact scope and limitations.

Student/team verification: pending. The team must inspect, understand and explain the implementation, check it on Windows/MySQL and real microphones, review dataset provenance and complete any competition-required AI disclosure form. No manual student understanding or testing is asserted here. GTM remains deferred and full SRS compliance is not claimed.

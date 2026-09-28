# Verification of the SRS UI release

Verified in the development environment on 28 September 2026.

- Real latest CNN inference on WAV and MP3 uploads; all ten scores are returned and saved. Selected model/contract checks pass.
- All files under `models/`, protected preprocessing source, selected-model metadata, Windows ML pins and the original decision-policy file are byte-identical to the previously delivered integrated model ZIP.
- Functional checks with isolated SQLite: all new pages render; metadata, full scores, exact duplicates, waveform/spectrogram data, owner isolation for detail/audio/JSON/visuals, role/CSRF checks, review preserves original prediction and confidence, alert action history, legacy records, profile update and settings guards. Threshold test restores the original policy afterward.
- Silent, malformed, too-short and over-60-second audio is rejected. A mixed browser batch correctly saves the valid file and reports the broken file.
- New portal pages checked at 1440, 768 and 390 pixels: no document-wide horizontal overflow or broken image assets. Wide tables scroll inside their container.
- Actual browser HTTP workflow: sign-in, native file upload, automatic result navigation, all ten scores/top three, review correction timeline, batch processing, profile save and history filtering. No JavaScript errors in these workflows.
- Browser live monitoring: actual backend inference with a synthetic microphone fixture; start/pause/resume/stop, all ten live scores, sample rate, mobile navigation, report charts, filters and CSV download passed.
- Saved-upload reports: real database totals, date/class filtering, account isolation, CSV formula escaping and latest model metrics passed.
- Desktop and mobile screenshots were visually inspected for the upload and recording-analysis layouts.

Environment: Linux, Python 3.12, isolated SQLite databases, Chromium 134. The project targets the included pinned Windows Python 3.11 environment. XAMPP/MySQL and a physical microphone must still be checked on the user's machine. QA users, fixture databases and browser test uploads are outside the delivered project.

These checks validate integration and UI behavior. They do not re-evaluate model accuracy, certify all SRS requirements, or implement GTM. See `SRS_UI_ALIGNMENT.md` and `UI_VERIFICATION.json`.

To repeat the included functional checks after installing dependencies:

```bat
.venv\Scripts\python.exe tests\test_srs_portal.py
```

The script creates a temporary SQLite database and temporary upload folder. It tests an administrator threshold edit and restores the original policy file afterward; run it with the local app stopped.

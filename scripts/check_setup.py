"""Quick self-test after setup: both models load and classify one sample file."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

sample = next((p for p in [ROOT / 'sample_audio' / 'gunshot_sample.wav', ROOT / 'cat_model_input.wav'] if p.is_file()), None)
ok = True
try:
    from src.ml.gtm import load_gtm
    m = load_gtm()
    print(f'[OK]   Teachable Machine model loaded ({m.version}, {len(m.labels)} classes)')
except Exception as exc:
    ok = False
    print(f'[FAIL] Teachable Machine model: {exc}')
try:
    from src.ml.predict import get_selected_model_name
    print(f'[OK]   Python model selected: {get_selected_model_name()}')
except Exception as exc:
    ok = False
    print(f'[FAIL] Python model: {exc}')
if sample:
    try:
        from services.analysis import analyze_file
        r = analyze_file(sample)
        py, tm, c, f = r['python'], r['gtm'], r['comparison'], r['final']
        print(f"[OK]   Test file {sample.name}: Python={py.get('prediction', {}).get('label')} "
              f"GTM={tm.get('prediction', {}).get('label')} -> {c.get('status')} -> final {f.get('display_label')} "
              f"({f.get('confidence', 0) * 100:.1f}%) in {r['processing_ms']:.0f} ms")
        if py.get('status') != 'ready':
            ok = False
            print(f"[FAIL] Python prediction error: {py.get('error')}")
    except Exception as exc:
        ok = False
        print(f'[FAIL] Test analysis: {exc}')
print('Self-test passed.' if ok else 'Self-test found problems (see above).')
sys.exit(0 if ok else 1)

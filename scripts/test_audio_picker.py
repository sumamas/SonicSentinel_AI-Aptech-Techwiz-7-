"""Choose a file and compare the three Python models on Windows."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk();root.withdraw()
    try:path=filedialog.askopenfilename(title='SonicSentinel: audio select karein',filetypes=[('Audio files','*.wav *.mp3 *.flac *.ogg *.m4a'),('All files','*.*')])
    finally:root.destroy()
    if not path:
        print('Koi audio select nahi hui.');return
    from src.ml.predict import predict_models,compare_predictions
    result=compare_predictions(predict_models(path))
    result['audio_path']=path
    text=json.dumps(result,indent=2);print(text)
    out=ROOT/'reports/manual_tests'/('comparison_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8');print('SAVED:',out)

if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError,ValueError) as exc:raise SystemExit(str(exc)) from exc

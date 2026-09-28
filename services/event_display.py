"""Display/storage helpers. Never modifies the evaluated preprocessing contract."""
from datetime import datetime
from pathlib import Path
import hashlib
import io
import json
import secrets
import hmac
import numpy as np
from flask import session, current_app, abort, request
from extensions import db
from models_db import User, AudioEvent

ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROLES = {'administrator','admin','audio_reviewer','reviewer'}
ALERT_ROLES = REVIEW_ROLES | {'security_operator','maintenance_operator'}
RECOMMENDATIONS = {
    'Machinery Fault':'Request an equipment inspection by maintenance staff.',
    'Glass Breaking':'Verify the recording and ask authorized staff to inspect the area.',
    'Alarm or Siren':'Check the source of the alarm and follow your site procedure.',
    'Gunshot':'Verify the alert promptly and follow the established site safety procedure.',
    'Panic Scream':'Review the recording and ask authorized staff to verify the situation.',
    'Aggression':'Review the event and notify authorized staff if appropriate.',
    'Person Asking for Help':'Review the possible help request and verify the situation.',
    'Vehicle Horn':'Record as a traffic or environmental sound event.',
    'Animal Sound':'Record the event and review only if the context requires it.',
    'Background Noise':'No automatic escalation; continue monitoring.',
}

def current_user():
    user=db.session.get(User,session.get('user_id'))
    if not user or not user.is_active: abort(403)
    return user

def is_admin(): return current_user().role in {'admin','administrator'}

def scoped_events():
    query=AudioEvent.query
    return query if is_admin() else query.filter_by(user_id=session['user_id'])

def event_for_user(event_id):
    event=scoped_events().filter_by(id=event_id).first()
    if event is None: abort(404)
    return event

def csrf_token():
    if 'csrf_token' not in session: session['csrf_token']=secrets.token_urlsafe(32)
    return session['csrf_token']

def check_csrf():
    expected=session.get('csrf_token',''); supplied=request.form.get('csrf_token') or request.headers.get('X-CSRF-Token','')
    if not expected or not hmac.compare_digest(expected,supplied):abort(400,description='Your session form expired. Refresh the page and try again.')

def audio_path(event):
    root=Path(current_app.config['UPLOAD_FOLDER']).resolve()
    path=(root/event.stored_filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():abort(404,description='The stored audio file is unavailable.')
    return path

def audio_metadata(path, original_filename):
    import soundfile as sf
    meta={'filename':original_filename,'format':path.suffix.lstrip('.').upper(),'file_size_bytes':path.stat().st_size,'channels':None,'sample_rate':None,'bit_depth':None,'duration_seconds':None}
    try:
        info=sf.info(str(path))
        meta.update(channels=info.channels,sample_rate=info.samplerate,duration_seconds=float(info.duration),encoding=info.subtype)
        meta['bit_depth']={'PCM_16':16,'PCM_24':24,'PCM_32':32,'PCM_U8':8,'PCM_S8':8,'FLOAT':32,'DOUBLE':64}.get(info.subtype)
    except (RuntimeError,OSError):pass
    return meta

def recommendation(label,status):
    if status in {'Manual Review','Uncertain'}:return 'Listen to the recording and confirm or correct the candidate class before taking action.'
    return RECOMMENDATIONS.get(label,'Inspect the recording and confirm the result manually.')

def visual_data(path):
    from src.audio.preprocess import load_audio_native
    from scipy.signal import stft
    y,sr=load_audio_native(str(path))
    # Display the complete accepted clip, not the model's selected 3-second window.
    chunks=np.array_split(y,min(900,len(y)))
    wave=[round(float(np.max(np.abs(c))),5) for c in chunks if len(c)]
    frequencies,times,z=stft(y,fs=sr,nperseg=min(512,len(y)),noverlap=min(384,max(0,len(y)//2)))
    magnitude=20*np.log10(np.maximum(np.abs(z),1e-7));magnitude=np.clip((magnitude+80)/80,0,1)
    rows=np.linspace(0,len(frequencies)-1,min(100,len(frequencies))).astype(int)
    cols=np.linspace(0,len(times)-1,min(280,len(times))).astype(int)
    return {'duration_seconds':round(len(y)/sr,3),'sample_rate':sr,'waveform':wave,'spectrogram':np.round(magnitude[np.ix_(rows,cols)],3).tolist(),'max_frequency_hz':float(frequencies[-1])}

def training_summary():
    selected=json.loads((ROOT/'models/selected_model.json').read_text())
    models=[]
    for key,name in [('custom_cnn','Custom CNN'),('random_forest','Random Forest'),('svm','SVM')]:
        metrics=json.loads((ROOT/f'reports/model_evaluation/{key}_test_metrics.json').read_text())
        models.append(dict(key=key,name=name,selected=key==selected['selected_model'],**metrics))
    return selected,models


def system_chips():
    """Cheap file-based status for the page header (does not load TensorFlow)."""
    info = {'python': False, 'python_name': 'model', 'python_note': '', 'gtm': False, 'gtm_note': ''}
    try:
        selected = json.loads((ROOT / 'models/selected_model.json').read_text())
        key = selected.get('selected_model', '')
        file = ROOT / 'models' / ('custom_cnn.keras' if key == 'custom_cnn' else f'{key}.joblib')
        info['python'] = file.is_file()
        info['python_name'] = selected.get('display_name', key)
        info['python_note'] = 'Selected on the validation split' if info['python'] else 'Model file missing in models/'
    except (OSError, ValueError):
        info['python_note'] = 'models/selected_model.json missing'
    files = [ROOT / 'gtm_model' / n for n in ('model.json', 'weights.bin', 'metadata.json')]
    info['gtm'] = all(f.is_file() for f in files)
    info['gtm_note'] = 'Exported TF.js model in gtm_model/' if info['gtm'] else 'Copy the GTM export into gtm_model/'
    return info

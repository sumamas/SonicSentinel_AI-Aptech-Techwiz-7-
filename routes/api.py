import hashlib
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, request, jsonify, current_app, session
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename

from extensions import db
from models_db import AudioEvent, EventAnalysis
from services.analysis import analyze_file, AudioRejected
from utils.auth import login_required

api_bp = Blueprint('api', __name__, url_prefix='/api')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file, folder: Path, prefix: str = '') -> tuple[Path, str, str]:
    extension = file.filename.rsplit('.', 1)[1].lower()
    original = secure_filename(file.filename) or f'recording.{extension}'
    stored = f'{prefix}{uuid4().hex}.{extension}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / stored
    file.save(path)
    return path, original, stored


def _event_from_analysis(result: dict, original: str, stored: str, source: str) -> AudioEvent:
    final = result.get('final') or {}
    python = result.get('python') or {}
    gtm = result.get('gtm') or {}
    comparison = result.get('comparison') or {}
    quality = result['quality']
    return AudioEvent(
        user_id=session.get('user_id'),
        original_filename=original[:255],
        stored_filename=stored,
        sample_rate=int(result['sample_rate']),
        duration_seconds=float(result['duration_seconds']),
        quality_label=quality['label'],
        silence_ratio=float(quality['silence_ratio']),
        clipping_ratio=float(quality['clipping_ratio']),
        rms=float(quality['rms']),
        predicted_class=final.get('label'),
        confidence=final.get('confidence'),
        severity=final.get('severity'),
        status=final.get('status') or 'Inspected',
        python_class=(python.get('prediction') or {}).get('label'),
        python_confidence=(python.get('prediction') or {}).get('confidence'),
        gtm_class=(gtm.get('prediction') or {}).get('label'),
        gtm_confidence=(gtm.get('prediction') or {}).get('confidence'),
        agreement_status=comparison.get('status'),
        confidence_difference=comparison.get('confidence_difference'),
        source=source,
    )


def _public_result(event: AudioEvent, result: dict, snapshot: dict) -> dict:
    final = result.get('final') or {}
    return {
        'event_id': event.id,
        'filename': event.original_filename,
        'detail_url': f'/workspace/events/{event.id}',
        'duration_seconds': result['duration_seconds'],
        'sample_rate': result['sample_rate'],
        'quality': result['quality'],
        'python': result['python'],
        'gtm': result['gtm'],
        'comparison': result['comparison'],
        'final': final,
        'prediction_enabled': bool(final.get('label')),
        'prediction': {'label': final.get('display_label'), 'confidence': final.get('confidence')} if final else None,
        'severity': final.get('severity'),
        'status': final.get('status'),
        'active_alert': final.get('active_alert', False),
        'decision_note': final.get('decision_note'),
        'duplicate_of': snapshot.get('duplicate_of'),
        'processing_ms': result.get('processing_ms'),
        'metadata': snapshot.get('metadata'),
    }


@api_bp.post('/audio/inspect')
@login_required
def inspect_audio():
    file = request.files.get('audio')
    if not file or not file.filename:
        return jsonify({'error': 'Please choose an audio file first.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported format. Use WAV, MP3, FLAC, OGG or M4A.'}), 400

    source = request.form.get('source', 'upload')[:20]
    path, original, stored = _save_upload(file, Path(current_app.config['UPLOAD_FOLDER']))
    try:
        result = analyze_file(path)
    except AudioRejected as exc:
        path.unlink(missing_ok=True)
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        current_app.logger.exception('Analysis failed')
        path.unlink(missing_ok=True)
        return jsonify({'error': f'Analysis failed: {exc}'}), 500

    try:
        from services.event_display import audio_metadata, recommendation
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        duplicate = (EventAnalysis.query.join(AudioEvent)
                     .filter(AudioEvent.user_id == session['user_id'], EventAnalysis.audio_sha256 == digest)
                     .order_by(AudioEvent.id).first())
        event = _event_from_analysis(result, original, stored, source)
        db.session.add(event)
        db.session.flush()
        final = result.get('final') or {}
        snapshot = {
            'metadata': audio_metadata(path, original),
            'quality': result['quality'],
            'python': result['python'],
            'gtm': result['gtm'],
            'comparison': result['comparison'],
            'final': final,
            'prediction': result['python'] if result['python'].get('status') == 'ready' else None,
            'original_status': event.status,
            'original_severity': event.severity,
            'recommended_action': recommendation(final.get('label'), event.status),
            'duplicate_of': duplicate.event_id if duplicate else None,
            'policy_version': result.get('policy_version'),
            'processing_ms': result.get('processing_ms'),
            'model_versions': {
                'python': result['python'].get('model_sha256'),
                'gtm': result['gtm'].get('model_version'),
                'gtm_sha256': result['gtm'].get('model_sha256'),
            },
        }
        db.session.add(EventAnalysis(event_id=event.id, data=snapshot, audio_sha256=digest))
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        path.unlink(missing_ok=True)
        current_app.logger.exception('Could not save audio analysis')
        reason = str(getattr(exc, 'orig', exc)).splitlines()[0][:220]
        return jsonify({'error': f'Database error while saving: {reason}. Run run_setup.bat again or check that MySQL is running.'}), 500

    return jsonify(_public_result(event, result, snapshot))


@api_bp.post('/live/inspect')
@login_required
def inspect_live_audio():
    """Classify one temporary live-microphone window (not stored; file deleted)."""
    file = request.files.get('audio')
    if not file or not file.filename:
        return jsonify({'error': 'No live audio window provided.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported live audio format. The browser should send WAV audio.'}), 400
    path, original, _ = _save_upload(file, Path(current_app.config['UPLOAD_FOLDER']) / 'live_tmp', 'live_')
    try:
        result = analyze_file(path, live=True)
    except AudioRejected as exc:
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        current_app.logger.exception('Live analysis failed')
        return jsonify({'error': f'Live analysis failed: {exc}'}), 500
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    python, gtm, comparison = result['python'], result['gtm'], result['comparison']
    final = result.get('final') or {}

    def block(model):
        if model.get('status') != 'ready':
            return {'status': 'unavailable', 'prediction': None, 'confidence': None, 'top3': [],
                    'all_confidences': {}, 'error': model.get('error')}
        return {'status': 'ready', 'model': model.get('model'), 'prediction': model['prediction']['label'],
                'confidence': model['prediction']['confidence'], 'top3': model.get('top3', []),
                'all_confidences': model.get('all_confidences', {}), 'top_two_margin': model.get('top_two_margin'),
                'model_sha256': model.get('model_sha256'), 'model_version': model.get('model_version')}

    final_model = None
    if final.get('label'):
        final_model = {'status': 'ready', 'model': 'final decision', 'prediction': final['label'],
                       'confidence': final['confidence'], 'top3': final['top3'],
                       'all_confidences': final['all_confidences'], 'top_two_margin': final['top_two_margin']}
    return jsonify({
        'filename': original,
        'sample_rate': result['sample_rate'],
        'duration_seconds': result['duration_seconds'],
        'quality': result['quality'],
        'prediction_enabled': final_model is not None,
        'python_model': block(python),
        'gtm_model': block(gtm),
        'final_model': final_model,
        'model_agreement': comparison.get('status'),
        'comparison': comparison,
        'severity': final.get('severity'),
        'event_status': 'Confirmation Required' if final.get('candidate_alert') else final.get('status', 'Classified'),
        'raw_event_status': final.get('status'),
        'active_alert': False,
        'candidate_alert': bool(final.get('candidate_alert')) and not ({'audio_quality_requires_review', 'unknown_sound_pattern'} & set(final.get('reasons', []))),
        'decision_note': final.get('decision_note'),
        'reasons': final.get('reasons', []),
        'processing_ms': result.get('processing_ms'),
        'live_capture': {'mode': request.form.get('capture_mode', 'unknown'),
                         'event_token': request.form.get('event_token', '')},
        'confirmation_required': True,
    })


@api_bp.get('/system/status')
@login_required
def system_status():
    status = {'database': current_app.config.get('DB_ACTIVE_ENGINE')}
    try:
        from src.ml.predict import get_selected_model_name
        status['python_model'] = {'ready': True, 'name': get_selected_model_name()}
    except Exception as exc:
        status['python_model'] = {'ready': False, 'error': str(exc)}
    try:
        from src.ml.gtm import load_gtm, GTM_PROJECT_URL
        m = load_gtm()
        status['gtm_model'] = {'ready': True, 'version': m.version, 'labels': m.labels, 'url': GTM_PROJECT_URL}
    except Exception as exc:
        status['gtm_model'] = {'ready': False, 'error': str(exc)}
    return jsonify(status)

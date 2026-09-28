from pathlib import Path
from uuid import uuid4

from flask import Blueprint, request, jsonify, current_app, session
from werkzeug.utils import secure_filename

from extensions import db
from models_db import AudioEvent
from src.audio.preprocess import load_audio
from src.audio.quality import assess_quality
from src.ml.predict import predict_file
from utils.auth import login_required

api_bp = Blueprint('api', __name__, url_prefix='/api')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Starter 3-class decision rules. These can be tuned again when the final
# 10-class dataset is trained.
MANUAL_REVIEW_THRESHOLD = 0.60
ALERT_CONFIDENCE_THRESHOLD = 0.80
CRITICAL_LABELS = {'Gunshot'}
HIGH_LABELS = {'Glass Breaking', 'Alarm or Siren'}
POOR_QUALITY_LABELS = {'Poor', 'Unusable'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _classify_decision(label: str, confidence: float, quality_label: str) -> dict:
    """Turn a model prediction into starter severity/status/alert values."""
    confidence = float(confidence or 0.0)

    if quality_label in POOR_QUALITY_LABELS:
        return {
            'severity': 'Medium',
            'status': 'Manual Review',
            'active_alert': False,
            'decision_note': 'Prediction received, but audio quality requires manual review.',
        }

    if confidence < MANUAL_REVIEW_THRESHOLD:
        return {
            'severity': 'Low',
            'status': 'Manual Review',
            'active_alert': False,
            'decision_note': 'Low-confidence result. Manual review is recommended.',
        }

    if confidence < ALERT_CONFIDENCE_THRESHOLD:
        return {
            'severity': 'Medium',
            'status': 'Uncertain',
            'active_alert': False,
            'decision_note': 'Prediction is below the starter alert threshold.',
        }

    if label in CRITICAL_LABELS:
        return {
            'severity': 'Critical',
            'status': 'Alert Generated',
            'active_alert': True,
            'decision_note': 'High-confidence critical sound detected.',
        }

    if label in HIGH_LABELS:
        return {
            'severity': 'High',
            'status': 'Alert Generated',
            'active_alert': True,
            'decision_note': 'High-confidence security/attention sound detected.',
        }

    return {
        'severity': 'Informational',
        'status': 'Classified',
        'active_alert': False,
        'decision_note': 'Sound classified successfully.',
    }


def _prediction_payload(path: Path, quality_label: str) -> dict:
    """Run the currently selected Python model and normalize API output."""
    result = predict_file(str(path), model_name='selected')
    predicted = result['prediction']
    label = str(predicted['label'])
    confidence = float(predicted['confidence'])
    decision = _classify_decision(label, confidence, quality_label)

    return {
        'model': str(result.get('model', 'selected')),
        'label': label,
        'confidence': confidence,
        'top3': result.get('top3', []),
        'all_confidences': result.get('all_confidences', {}),
        **decision,
    }


@api_bp.post('/audio/inspect')
@login_required
def inspect_audio():
    file = request.files.get('audio')
    if not file or not file.filename:
        return jsonify({'error': 'No audio file provided.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported audio format. Use WAV, MP3, FLAC, OGG or M4A.'}), 400

    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit('.', 1)[1].lower()
    stored_filename = f"{uuid4().hex}.{extension}"

    upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / stored_filename
    file.save(path)

    try:
        y, sr = load_audio(str(path))
        quality = assess_quality(y)
        duration = round(len(y) / sr, 3)

        prediction = None
        prediction_error = None
        try:
            prediction = _prediction_payload(path, quality['label'])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            # Keep upload/quality inspection usable even if the local trained
            # model files were not copied yet.
            prediction_error = str(exc)

        event = AudioEvent(
            user_id=session.get('user_id'),
            original_filename=original_filename,
            stored_filename=stored_filename,
            sample_rate=sr,
            duration_seconds=duration,
            quality_label=quality['label'],
            silence_ratio=quality['silence_ratio'],
            clipping_ratio=quality['clipping_ratio'],
            rms=quality['rms'],
            predicted_class=prediction['label'] if prediction else None,
            confidence=prediction['confidence'] if prediction else None,
            severity=prediction['severity'] if prediction else None,
            status=prediction['status'] if prediction else 'Inspected',
        )
        db.session.add(event)
        db.session.commit()

        payload = {
            'event_id': event.id,
            'filename': original_filename,
            'sample_rate': sr,
            'duration_seconds': duration,
            'quality': quality,
            'prediction_enabled': prediction is not None,
        }

        if prediction:
            payload.update({
                'prediction': {
                    'label': prediction['label'],
                    'confidence': prediction['confidence'],
                    'model': prediction['model'],
                },
                'top3': prediction['top3'],
                'all_confidences': prediction['all_confidences'],
                'severity': prediction['severity'],
                'status': prediction['status'],
                'active_alert': prediction['active_alert'],
                'decision_note': prediction['decision_note'],
                'message': 'Audio preprocessing, quality analysis and Python model classification completed successfully.',
            })
        else:
            payload.update({
                'prediction': None,
                'severity': None,
                'status': 'Inspected',
                'active_alert': False,
                'message': f'Quality inspection succeeded, but prediction is unavailable: {prediction_error}',
            })

        return jsonify(payload)
    except Exception as exc:
        db.session.rollback()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return jsonify({'error': str(exc)}), 422


@api_bp.post('/live/inspect')
@login_required
def inspect_live_audio():
    """Classify one temporary live-microphone audio window.

    Live windows are not inserted into the database on every 3-second cycle;
    this avoids flooding event history during continuous monitoring. The file is
    deleted immediately after preprocessing/classification.
    """
    file = request.files.get('audio')
    if not file or not file.filename:
        return jsonify({'error': 'No live audio window provided.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported live audio format. The browser should send WAV audio.'}), 400

    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit('.', 1)[1].lower()
    stored_filename = f"live_{uuid4().hex}.{extension}"

    live_dir = Path(current_app.config['UPLOAD_FOLDER']) / 'live_tmp'
    live_dir.mkdir(parents=True, exist_ok=True)
    path = live_dir / stored_filename
    file.save(path)

    try:
        y, sr = load_audio(str(path))
        quality = assess_quality(y)
        duration = round(len(y) / sr, 3)

        try:
            prediction = _prediction_payload(path, quality['label'])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return jsonify({
                'filename': original_filename,
                'sample_rate': sr,
                'duration_seconds': duration,
                'quality': quality,
                'prediction_enabled': False,
                'python_model': {
                    'status': 'unavailable',
                    'model': None,
                    'prediction': None,
                    'confidence': None,
                    'top3': [],
                },
                'gtm_model': {
                    'status': 'not_connected',
                    'prediction': None,
                    'confidence': None,
                },
                'model_agreement': None,
                'severity': None,
                'event_status': 'Inspected',
                'active_alert': False,
                'message': f'Live quality inspection succeeded, but Python prediction is unavailable: {exc}',
            })

        # Live monitoring uses repeated-window confirmation in the browser.
        # A single window is treated as a candidate only; the frontend will
        # confirm an event after consistent overlapping/adjacent windows.
        raw_alert = bool(prediction['active_alert'])
        live_status = prediction['status']
        if raw_alert:
            live_status = 'Confirmation Required'

        capture_mode = request.form.get('capture_mode', 'unknown')
        event_token = request.form.get('event_token', '')
        try:
            peak_score = float(request.form.get('peak_score', '0') or 0)
        except ValueError:
            peak_score = 0.0

        return jsonify({
            'filename': original_filename,
            'sample_rate': sr,
            'duration_seconds': duration,
            'quality': quality,
            'prediction_enabled': True,
            'python_model': {
                'status': 'ready',
                'model': prediction['model'],
                'prediction': prediction['label'],
                'confidence': prediction['confidence'],
                'top3': prediction['top3'],
            },
            'gtm_model': {
                'status': 'not_connected',
                'prediction': None,
                'confidence': None,
            },
            'model_agreement': None,
            'severity': prediction['severity'],
            'event_status': live_status,
            'active_alert': False,
            'candidate_alert': raw_alert,
            'raw_event_status': prediction['status'],
            'decision_note': prediction['decision_note'],
            'live_capture': {
                'mode': capture_mode,
                'event_token': event_token,
                'peak_score': peak_score,
            },
            'confirmation_required': True,
            'message': 'Live candidate classified. Repeated-window confirmation is applied before an alert is activated.',
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 422
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

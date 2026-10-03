"""Tests for the Teachable Machine integration, comparison rules and API saving."""
import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Machinery Fault', 'Glass Breaking', 'Alarm or Siren', 'Vehicle Horn', 'Animal Sound',
           'Gunshot', 'Panic Scream', 'Aggression', 'Person Asking for Help', 'Background Noise']


def scores(label, conf):
    d = {c: (1 - conf) / 9 for c in CLASSES}
    d[label] = conf
    return {'status': 'ready', 'prediction': {'label': label, 'confidence': conf}, 'all_confidences': d}


# ---------------------------------------------------------------- GTM model
def test_gtm_labels_are_cleaned_to_srs_names():
    from src.ml.gtm import canonical_label, load_gtm
    assert canonical_label('Gunshot  ') == 'Gunshot'
    assert canonical_label('Glass Breaking ') == 'Glass Breaking'
    assert set(load_gtm().labels) == set(CLASSES)


def test_gtm_scores_every_class_and_sums_to_one():
    from src.ml.gtm import classify_signal
    y = (np.random.default_rng(0).standard_normal(22050 * 2) * 0.1).astype(np.float32)
    r = classify_signal(y, 22050)
    assert set(r['all_confidences']) == set(CLASSES)
    assert abs(sum(r['all_confidences'].values()) - 1) < 1e-4
    assert len(r['top3']) == 3


def test_gtm_recognises_sample_gunshot():
    from src.ml.gtm import classify_file
    r = classify_file(ROOT / 'sample_audio' / 'gunshot_sample.wav')
    assert 'Gunshot' in [x['label'] for x in r['top3'][:2]]


# ---------------------------------------------------------- comparison rules
def test_confidence_difference_formula():
    from src.ml.comparison import compare
    r = compare(scores('Gunshot', 0.9), scores('Gunshot', 0.7))
    assert r['comparison']['confidence_difference'] == pytest.approx(0.2)
    assert r['comparison']['status'] == 'Acceptable Match'


def test_disagreement_goes_to_manual_review():
    from src.ml.comparison import compare
    r = compare(scores('Gunshot', 0.9), scores('Glass Breaking', 0.8))
    assert r['comparison']['status'] == 'Model Disagreement'
    assert r['final']['manual_review_required'] and 'model_disagreement' in r['final']['reasons']
    assert not r['final']['active_alert']


def test_agreed_critical_event_raises_alert():
    from src.ml.comparison import compare
    r = compare(scores('Gunshot', 0.99), scores('Gunshot', 0.85))
    assert r['final']['status'] == 'Alert Generated' and r['final']['severity'] == 'Critical'


def test_quiet_background_is_not_uncertain():
    from src.ml.comparison import compare
    r = compare(scores('Background Noise', 0.5), scores('Background Noise', 0.45))
    assert r['final']['status'] == 'Classified' and r['final']['display_label'] == 'Background Noise'


def test_poor_quality_needs_review():
    from src.ml.comparison import compare
    r = compare(scores('Vehicle Horn', 0.95), scores('Vehicle Horn', 0.9), 'Poor')
    assert 'audio_quality_requires_review' in r['final']['reasons']


def test_raw_model_scores_are_not_modified():
    from src.ml.comparison import compare
    py, tm = scores('Gunshot', 0.9), scores('Glass Breaking', 0.6)
    before = (dict(py['all_confidences']), dict(tm['all_confidences']))
    compare(py, tm)
    assert (py['all_confidences'], tm['all_confidences']) == before


# ------------------------------------------------------------------ API + DB
@pytest.fixture(scope='module')
def client(tmp_path_factory):
    os.environ['DATABASE_URL'] = 'sqlite:///' + str(tmp_path_factory.mktemp('db') / 't.db')
    import importlib, app_config
    importlib.reload(app_config)
    from app import create_app
    from extensions import db
    from models_db import User
    app = create_app(app_config.Config)
    app.config['TESTING'] = True
    with app.app_context():
        u = User(full_name='Tester', email='t@example.com', role='administrator')
        u.set_password('Password1'); db.session.add(u); db.session.commit()
    c = app.test_client()
    c.post('/auth/login', data={'email': 't@example.com', 'password': 'Password1'})
    yield c
    os.environ.pop('DATABASE_URL', None)


def wav(y, sr=22050):
    b = io.BytesIO(); sf.write(b, y, sr, format='WAV'); b.seek(0); return b


def test_upload_is_saved_with_both_models(client):
    with open(ROOT / 'sample_audio' / 'gunshot_sample.wav', 'rb') as fh:
        r = client.post('/api/audio/inspect', data={'audio': (fh, 'g.wav')}, content_type='multipart/form-data')
    j = r.get_json()
    assert r.status_code == 200, j
    assert j['python']['status'] == 'ready' and j['gtm']['status'] == 'ready'
    assert j['event_id'] and j['final']['label']
    assert client.get(j['detail_url']).status_code == 200


def test_duplicate_upload_is_flagged(client):
    for _ in range(2):
        with open(ROOT / 'sample_audio' / 'glass_breaking_sample.wav', 'rb') as fh:
            j = client.post('/api/audio/inspect', data={'audio': (fh, 'g.wav')}, content_type='multipart/form-data').get_json()
    assert j['duplicate_of'] is not None


@pytest.mark.parametrize('payload,name,code', [
    (lambda: wav(np.zeros(66150, dtype=np.float32)), 'silent.wav', 422),
    (lambda: wav((np.random.randn(1500) * 0.2).astype(np.float32)), 'short.wav', 422),
    (lambda: io.BytesIO(b'not audio' * 50), 'broken.mp3', 422),
    (lambda: io.BytesIO(b'x'), 'notes.txt', 400),
])
def test_invalid_audio_is_rejected(client, payload, name, code):
    r = client.post('/api/audio/inspect', data={'audio': (payload(), name)}, content_type='multipart/form-data')
    assert r.status_code == code and r.get_json()['error']


@pytest.mark.parametrize('url', ['/workspace/event-history', '/workspace/alerts', '/workspace/manual-review',
                                 '/workspace/batch-upload', '/audio/upload', '/about', '/how-it-works'])
def test_pages_render(client, url):
    assert client.get(url).status_code == 200

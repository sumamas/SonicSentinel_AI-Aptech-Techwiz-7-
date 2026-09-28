from flask import Blueprint, render_template, session
from models_db import AudioEvent

main_bp = Blueprint('main', __name__)


@main_bp.get('/')
def home():
    recent_events = []
    if session.get('user_id'):
        recent_events = (AudioEvent.query
                         .filter_by(user_id=session['user_id'])
                         .order_by(AudioEvent.created_at.desc())
                         .limit(5).all())
    return render_template('home.html', recent_events=recent_events)


@main_bp.get('/health')
def health():
    return {'status': 'ok'}


@main_bp.get('/about')
def about():
    from src.ml.reliability import SEVERITY
    return render_template('pages/about.html', severity=SEVERITY)


@main_bp.get('/how-it-works')
def how_it_works():
    from src.ml.reliability import read_policy, SEVERITY
    from src.ml.comparison import DEFAULT_COMPARISON_POLICY
    from src.ml.training_config import ROOT
    policy = read_policy(ROOT)
    comparison = dict(DEFAULT_COMPARISON_POLICY)
    comparison.update(policy.get('comparison') or {})
    return render_template('pages/how_it_works.html', policy=policy, comparison=comparison, severity=SEVERITY)

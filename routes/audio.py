from flask import Blueprint, render_template
from utils.auth import login_required

audio_bp = Blueprint('audio', __name__, url_prefix='/audio')


@audio_bp.get('/upload')
@login_required
def upload():
    return render_template('audio/upload.html')

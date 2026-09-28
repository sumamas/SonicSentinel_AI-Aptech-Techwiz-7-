"""Check the exact model used by the Flask app; optionally classify one file."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', help='Optional WAV/MP3/FLAC/OGG/M4A file')
    parser.add_argument('--database', action='store_true', help='Check database connection and tables')
    args = parser.parse_args()
    from src.audio.decode import ffmpeg_executable
    from src.ml.predict import get_selected_model_name, cnn_labels, _load_cnn, predict_file
    print('Project:', ROOT)
    print('FFmpeg:', ffmpeg_executable())
    selected = get_selected_model_name()
    if selected != 'custom_cnn':
        raise RuntimeError('This integrated release expects the exported custom_cnn selection.')
    labels = cnn_labels()
    model = _load_cnn()
    print('Selected model:', selected)
    print('Classes:', len(labels), '|', ', '.join(labels))
    print('Model loaded. Input:', model.input_shape, '| Output:', model.output_shape)
    if args.audio:
        print(json.dumps(predict_file(args.audio), indent=2))
    if args.database:
        from app import create_app
        from extensions import db
        from models_db import User, AudioEvent
        from sqlalchemy import select
        app = create_app()
        with app.app_context():
            db.session.execute(select(User.id).limit(1)).first()
            db.session.execute(select(AudioEvent.id).limit(1)).first()
        print('Database connection and app tables: OK')
    print('READY for local project testing. This check does not certify model accuracy.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('CHECK FAILED:', exc, file=sys.stderr)
        print('Use Python 3.11 and install requirements.txt in this project folder.', file=sys.stderr)
        raise SystemExit(1)

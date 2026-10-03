import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from datetime import timedelta
from pathlib import Path
from flask import Flask, session, jsonify, request, render_template
from app_config import Config
from extensions import db

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.permanent_session_lifetime = timedelta(days=30)
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

    from database.bootstrap import resolve_database_uri, ensure_schema
    app.config.setdefault('DB_ACTIVE_ENGINE', 'Custom DATABASE_URL')
    if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite'):
        app.config['DB_ACTIVE_ENGINE'] = 'SQLite'
    app.config['SQLALCHEMY_DATABASE_URI'] = resolve_database_uri(app)
    db.init_app(app)

    import models_db  # noqa: F401

    from routes.main import main_bp
    from routes.auth import auth_bp
    from routes.audio import audio_bp
    from routes.api import api_bp
    from routes.workspace import workspace_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(workspace_bp)

    with app.app_context():
        ensure_schema(db)

    from services.event_display import csrf_token, system_chips
    app.jinja_env.globals['csrf_token'] = csrf_token
    app.jinja_env.globals['system_chips'] = system_chips

    @app.context_processor
    def inject_session_user():
        return {
            'session_user_name': session.get('user_name'),
            'session_user_role': session.get('user_role'),
            'db_engine': app.config.get('DB_ACTIVE_ENGINE'),
        }

    @app.errorhandler(413)
    def too_large(_):
        msg = f"File is larger than the {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB upload limit."
        if request.path.startswith('/api/'):
            return jsonify(error=msg), 413
        return msg, 413

    @app.route('/health')
    def health_check():
        return {'status': 'ok', 'service': 'sonicsentinel'}, 200

    return app


app = create_app()


if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
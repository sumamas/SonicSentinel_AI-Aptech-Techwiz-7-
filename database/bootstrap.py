"""Create the database/tables automatically and add columns introduced later.

Fixes: "The analysis could not be saved. Check MySQL and run scripts/init_db.py".
That error appeared because event_analyses / event_actions (and new columns)
were missing from databases created with the old SQL file.
"""
import logging
from sqlalchemy import inspect, text

log = logging.getLogger('sonicsentinel')

NEW_COLUMNS = {
    'audio_events': {
        'python_class': 'VARCHAR(100) NULL',
        'python_confidence': 'FLOAT NULL',
        'gtm_class': 'VARCHAR(100) NULL',
        'gtm_confidence': 'FLOAT NULL',
        'agreement_status': 'VARCHAR(40) NULL',
        'confidence_difference': 'FLOAT NULL',
        'source': "VARCHAR(20) NULL",
    },
    'event_analyses': {'audio_sha256': 'VARCHAR(64) NULL'},
}


def resolve_database_uri(app):
    """Make sure MySQL is reachable and the schema exists; otherwise use SQLite."""
    from config import sqlite_uri
    uri = app.config['SQLALCHEMY_DATABASE_URI']

    if not uri.startswith('mysql'):
        return uri

    try:
        import pymysql
        from urllib.parse import urlparse

        # SQLAlchemy URL se credentials parse karein
        parsed = urlparse(uri)
        host = parsed.hostname or '127.0.0.1'
        port = parsed.port or 3306
        user = parsed.username or 'root'
        password = parsed.password or ''
        database = parsed.path.lstrip('/') or 'sonicsentinel'

        conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            charset='utf8mb4', autocommit=True, connect_timeout=5
        )
        try:
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        finally:
            conn.close()

        # Config mein individual variables bhi set karein (dusre code ke liye)
        app.config['DB_HOST'] = host
        app.config['DB_PORT'] = str(port)
        app.config['DB_USER'] = user
        app.config['DB_PASSWORD'] = password
        app.config['DB_NAME'] = database
        app.config['DB_ACTIVE_ENGINE'] = 'MySQL'
        return uri
    except Exception as exc:
        if not app.config.get('SQLITE_FALLBACK', True):
            raise
        log.warning('MySQL not reachable (%s). Using local SQLite database instead.', exc)
        app.config['DB_ACTIVE_ENGINE'] = 'SQLite (MySQL fallback)'
        app.config['DB_FALLBACK_REASON'] = str(exc)
        return sqlite_uri()


def ensure_schema(db):
    db.create_all()
    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())
    with db.engine.begin() as conn:
        for table, columns in NEW_COLUMNS.items():
            if table not in tables:
                continue
            existing = {c['name'] for c in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {ddl}'))
                    log.info('Added column %s.%s', table, name)
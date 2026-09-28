"""Create the database, all tables, missing columns and default accounts.

Usage:  python scripts/init_db.py
MySQL (XAMPP) is used when it is running; otherwise a local SQLite database is
created in instance/sonicsentinel.db (set DB_ENGINE=sqlite in .env to force it).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from app import create_app          # noqa: E402  (create_app runs the schema bootstrap)
from extensions import db           # noqa: E402
from models_db import User          # noqa: E402

DEFAULT_ACCOUNTS = [
    (os.getenv('ADMIN_NAME', 'System Administrator'), os.getenv('ADMIN_EMAIL', 'admin@sonicsentinel.local'),
     os.getenv('ADMIN_PASSWORD', 'Admin@12345'), 'administrator'),
    ('Audio Reviewer', 'reviewer@sonicsentinel.local', 'Review@12345', 'audio_reviewer'),
    ('Security Operator', 'security@sonicsentinel.local', 'Security@12345', 'security_operator'),
    ('Evaluator User', 'user@sonicsentinel.local', 'User@12345', 'normal_user'),
]

app = create_app()
with app.app_context():
    print(f"Database engine : {app.config.get('DB_ACTIVE_ENGINE')}")
    if app.config.get('DB_FALLBACK_REASON'):
        print(f"MySQL not reachable -> using SQLite. Reason: {app.config['DB_FALLBACK_REASON']}")
    print(f"Tables          : {', '.join(sorted(__import__('sqlalchemy').inspect(db.engine).get_table_names()))}")
    for name, email, password, role in DEFAULT_ACCOUNTS:
        user = User.query.filter(db.func.lower(User.email) == email.lower()).first()
        if user:
            print(f"Account exists  : {email} ({user.role})")
            continue
        user = User(full_name=name, email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        print(f"Account created : {email} / {password} ({role})")
    print('Database ready.')

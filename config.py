import os
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')


def sqlite_uri() -> str:
    folder = BASE_DIR / 'instance'
    folder.mkdir(parents=True, exist_ok=True)
    return 'sqlite:///' + (folder / 'sonicsentinel.db').as_posix()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-secret-key')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_UPLOAD_MB', '50')) * 1024 * 1024
    UPLOAD_FOLDER = str((BASE_DIR / os.getenv('UPLOAD_FOLDER', 'uploads')).resolve())

    # DB_ENGINE=mysql (XAMPP) or sqlite (zero-setup). With mysql, the app falls
    # back to SQLite automatically when MySQL is not running (SQLITE_FALLBACK=1).
    DB_ENGINE = os.getenv('DB_ENGINE', 'mysql').strip().lower()
    SQLITE_FALLBACK = os.getenv('SQLITE_FALLBACK', '1') == '1'
    DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
    DB_PORT = os.getenv('DB_PORT', '3307')
    DB_NAME = os.getenv('DB_NAME', 'sonicsentinel')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')

    _database_url = os.getenv('DATABASE_URL')
    if _database_url:
        SQLALCHEMY_DATABASE_URI = _database_url
    elif DB_ENGINE == 'sqlite':
        SQLALCHEMY_DATABASE_URI = sqlite_uri()
    else:
        SQLALCHEMY_DATABASE_URI = (
            f"mysql+pymysql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
        )
    SQLALCHEMY_ENGINE_OPTIONS = {'pool_pre_ping': True}
    SQLALCHEMY_TRACK_MODIFICATIONS = False

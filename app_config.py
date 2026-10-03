import os
from pathlib import Path
from urllib.parse import quote_plus, urlparse
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

    DB_ENGINE = os.getenv('DB_ENGINE', 'mysql').strip().lower()
    SQLITE_FALLBACK = os.getenv('SQLITE_FALLBACK', '1') == '1'
    DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
    DB_PORT = os.getenv('DB_PORT', '3307')
    DB_NAME = os.getenv('DB_NAME', 'sonicsentinel')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')

    _mysql_url = os.getenv('MYSQL_URL') or os.getenv('DATABASE_URL')
    _mysql_host = os.getenv('MYSQLHOST')
    _mysql_port = os.getenv('MYSQLPORT') or os.getenv('MYSQLPOR')
    _mysql_user = os.getenv('MYSQLUSER')
    _mysql_password = os.getenv('MYSQLPASSWORD') or os.getenv('MYSQL_ROOT_PASSWORD')
    _mysql_database = os.getenv('MYSQLDATABASE') or os.getenv('MYSQL_DATABASE')

    if _mysql_url:
        if _mysql_url.startswith('mysql://') and '+pymysql' not in _mysql_url:
            _mysql_url = _mysql_url.replace('mysql://', 'mysql+pymysql://', 1)
        SQLALCHEMY_DATABASE_URI = _mysql_url
    elif _mysql_host and _mysql_user and _mysql_database:
        _port = str(_mysql_port or 3306)
        _pwd = quote_plus(_mysql_password or '')
        SQLALCHEMY_DATABASE_URI = (
            f"mysql+pymysql://{quote_plus(_mysql_user)}:{_pwd}"
            f"@{_mysql_host}:{_port}/{_mysql_database}?charset=utf8mb4"
        )
    elif DB_ENGINE == 'sqlite':
        SQLALCHEMY_DATABASE_URI = sqlite_uri()
    else:
        SQLALCHEMY_DATABASE_URI = (
            f"mysql+pymysql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
        )
    SQLALCHEMY_ENGINE_OPTIONS = {'pool_pre_ping': True}
    SQLALCHEMY_TRACK_MODIFICATIONS = False
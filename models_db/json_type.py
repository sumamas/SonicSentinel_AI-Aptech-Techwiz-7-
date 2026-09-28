"""Portable JSON column: LONGTEXT on MySQL/MariaDB, TEXT elsewhere.

Works on old XAMPP MariaDB/MySQL versions without a native JSON type and on
tables that were created earlier with a JSON column.
"""
import json
from sqlalchemy.types import TypeDecorator, Text
from sqlalchemy.dialects import mysql


class JSONText(TypeDecorator):
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'mysql':
            return dialect.type_descriptor(mysql.LONGTEXT())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        from services.analysis import to_jsonable
        return json.dumps(to_jsonable(value), ensure_ascii=False)

    def process_result_value(self, value, dialect):
        if value is None or isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}

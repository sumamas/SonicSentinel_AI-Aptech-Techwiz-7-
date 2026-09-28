from datetime import datetime
from extensions import db
from .json_type import JSONText

class EventAnalysis(db.Model):
    __tablename__ = 'event_analyses'
    event_id = db.Column(db.Integer, db.ForeignKey('audio_events.id'), primary_key=True)
    data = db.Column(JSONText, nullable=False)
    audio_sha256 = db.Column(db.String(64), index=True)
    event = db.relationship('AudioEvent', backref=db.backref('analysis', uselist=False, cascade='all, delete-orphan'))

class EventAction(db.Model):
    __tablename__ = 'event_actions'
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey('audio_events.id'), nullable=False, index=True)
    actor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action = db.Column(db.String(30), nullable=False)
    corrected_class = db.Column(db.String(100))
    comment = db.Column(db.Text, nullable=False, default='')
    recommended_action = db.Column(db.String(1000), nullable=False, default='')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    event = db.relationship('AudioEvent', backref=db.backref('actions', order_by='EventAction.id', cascade='all, delete-orphan'))
    actor = db.relationship('User')

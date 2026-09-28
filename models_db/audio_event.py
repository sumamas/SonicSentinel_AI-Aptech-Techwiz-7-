from datetime import datetime
from extensions import db


class AudioEvent(db.Model):
    __tablename__ = 'audio_events'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    sample_rate = db.Column(db.Integer, nullable=False)
    duration_seconds = db.Column(db.Float, nullable=False)
    quality_label = db.Column(db.String(30), nullable=False)
    silence_ratio = db.Column(db.Float, nullable=False, default=0)
    clipping_ratio = db.Column(db.Float, nullable=False, default=0)
    rms = db.Column(db.Float, nullable=False, default=0)

    predicted_class = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    severity = db.Column(db.String(30), nullable=True)
    status = db.Column(db.String(40), nullable=False, default='Inspected')
    # Independent model outputs (raw, never modified) and their comparison.
    python_class = db.Column(db.String(100), nullable=True)
    python_confidence = db.Column(db.Float, nullable=True)
    gtm_class = db.Column(db.String(100), nullable=True)
    gtm_confidence = db.Column(db.Float, nullable=True)
    agreement_status = db.Column(db.String(40), nullable=True)
    confidence_difference = db.Column(db.Float, nullable=True)
    source = db.Column(db.String(20), nullable=True, default='upload')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    user = db.relationship('User', back_populates='audio_events')

    @property
    def latest_review(self):
        return next((a for a in reversed(self.actions) if a.action in {'confirm', 'correct'}), None)

    @property
    def final_class(self):
        review = self.latest_review
        return review.corrected_class if review else self.predicted_class

    @property
    def alert_state(self):
        action = next((a for a in reversed(self.actions) if a.action in {'acknowledge', 'dismiss', 'escalate'}), None)
        if action:
            return {'acknowledge':'Acknowledged','dismiss':'Dismissed','escalate':'Escalated'}[action.action]
        data = (self.analysis.data or {}) if self.analysis else {}
        original_status = data.get('original_status', self.status)
        if self.status == 'Closed':
            return 'Closed'
        if original_status == 'Alert Generated':
            return 'Open'
        final = data.get('final') or {}
        if final.get('candidate_alert') and final.get('manual_review_required') and self.status != 'Reviewed':
            return 'Pending review'
        return 'No active alert'

    @property
    def display_class(self):
        review = self.latest_review
        if review:
            return review.corrected_class
        data = (self.analysis.data or {}) if self.analysis else {}
        return (data.get('final') or {}).get('display_label') or self.predicted_class

from datetime import datetime
import json

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    preferred_language = db.Column(db.String(5), default="en", nullable=False)
    biometric_template = db.Column(db.Text, nullable=True)
    biometric_reference_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    analyses = db.relationship("Analysis", backref="user", lazy=True, cascade="all, delete-orphan")
    biometric_profiles = db.relationship(
        "BiometricProfile",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="BiometricProfile.updated_at.desc()",
    )
    logs = db.relationship("SystemLog", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == "admin"

    def biometric_template_dict(self):
        if not self.biometric_template:
            return {}
        try:
            return json.loads(self.biometric_template)
        except json.JSONDecodeError:
            return {}

    def latest_biometric_profile(self):
        return (
            BiometricProfile.query.filter_by(user_id=self.id)
            .order_by(BiometricProfile.updated_at.desc())
            .first()
        )


class BiometricProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    enrollment_image_path = db.Column(db.String(255), nullable=False)
    processed_image_path = db.Column(db.String(255), nullable=True)
    biometric_template = db.Column(db.Text, nullable=False)
    feature_summary = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def biometric_template_dict(self):
        try:
            return json.loads(self.biometric_template)
        except (TypeError, json.JSONDecodeError):
            return {}

    def feature_summary_dict(self):
        if not self.feature_summary:
            return {}
        try:
            return json.loads(self.feature_summary)
        except json.JSONDecodeError:
            return {}


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    result_status = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False, default=0.0)
    processing_time = db.Column(db.Float, nullable=False, default=0.0)
    feature_summary = db.Column(db.Text, nullable=True)
    annotated_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def feature_summary_dict(self):
        if not self.feature_summary:
            return {}
        try:
            return json.loads(self.feature_summary)
        except json.JSONDecodeError:
            return {}


class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(80), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    description = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

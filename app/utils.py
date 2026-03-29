from datetime import datetime
import json
import os
import secrets

from flask import current_app
from werkzeug.utils import secure_filename

from .models import SystemLog


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def unique_filename(filename):
    sanitized = secure_filename(filename or "upload")
    stem, _, ext = sanitized.rpartition(".")
    safe_stem = stem or "upload"
    safe_ext = ext.lower() if ext else "png"
    return f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_stem}_{secrets.token_hex(6)}.{safe_ext}"


def to_public_path(path_value):
    normalized = str(path_value).replace("\\", "/")
    base = str(current_app.root_path).replace("\\", "/")
    project_root = os.path.dirname(base)
    return normalized.replace(project_root, "").lstrip("/")


def log_event(action, description, user_id=None):
    entry = SystemLog(action=action, description=description, user_id=user_id)
    from .extensions import db
    db.session.add(entry)


def dump_feature_summary(data):
    return json.dumps(data, ensure_ascii=False)

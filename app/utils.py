from datetime import datetime
import json
import os
from pathlib import Path
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
    if not path_value:
        return None

    candidate = Path(str(path_value)).expanduser()
    storage_root = Path(current_app.config["STORAGE_ROOT"]).resolve()
    upload_root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    processed_root = Path(current_app.config["PROCESSED_FOLDER"]).resolve()

    try:
        return candidate.resolve().relative_to(storage_root).as_posix()
    except ValueError:
        pass

    try:
        relative_to_upload = candidate.resolve().relative_to(upload_root)
        return Path("uploads", relative_to_upload).as_posix()
    except ValueError:
        pass

    try:
        relative_to_processed = candidate.resolve().relative_to(processed_root)
        return Path("uploads", "processed", relative_to_processed).as_posix()
    except ValueError:
        pass

    normalized = str(path_value).replace("\\", "/").lstrip("/")
    for marker in ("instance/storage/", "uploads/", "processed/"):
        if normalized.startswith(marker):
            return normalized
    return normalized


def log_event(action, description, user_id=None):
    entry = SystemLog(action=action, description=description, user_id=user_id)
    from .extensions import db
    db.session.add(entry)


def dump_feature_summary(data):
    return json.dumps(data, ensure_ascii=False)

from pathlib import Path

from flask import Flask, g, redirect, render_template, request, send_from_directory, session, url_for
from flask_login import current_user
from sqlalchemy import inspect, text
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from .blueprints.admin import admin_bp
from .blueprints.auth import auth_bp
from .blueprints.dashboard import dashboard_bp
from .blueprints.main import main_bp
from .extensions import db, login_manager
from .models import SystemLog, User
from .translations import LANG_CONTENT, translate


def create_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(Config)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["PROCESSED_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(admin_bp)

    @app.before_request
    def resolve_language():
        lang = request.args.get("lang")
        if lang in app.config["LANGUAGES"]:
            session["lang"] = lang
            if current_user.is_authenticated and current_user.preferred_language != lang:
                current_user.preferred_language = lang
                db.session.commit()
        if current_user.is_authenticated and current_user.preferred_language:
            g.lang = current_user.preferred_language
        else:
            g.lang = session.get("lang", app.config["DEFAULT_LANGUAGE"])

    @app.context_processor
    def inject_globals():
        def media_url(path_value):
            if not path_value:
                return None
            normalized = str(path_value).replace("\\", "/").lstrip("/")
            storage_root = Path(app.config["STORAGE_ROOT"]).resolve().as_posix().rstrip("/")
            upload_root = Path(app.config["UPLOAD_FOLDER"]).resolve().as_posix().rstrip("/")
            processed_root = Path(app.config["PROCESSED_FOLDER"]).resolve().as_posix().rstrip("/")

            if normalized.startswith(storage_root + "/"):
                normalized = normalized[len(storage_root) + 1 :]
            elif normalized.startswith(upload_root + "/"):
                normalized = f"uploads/{normalized[len(upload_root) + 1 :]}"
            elif normalized.startswith(processed_root + "/"):
                normalized = f"uploads/processed/{normalized[len(processed_root) + 1 :]}"
            elif normalized.startswith("instance/storage/"):
                normalized = normalized[len("instance/storage/") :]

            if normalized.startswith("static/"):
                return url_for("static", filename=normalized[len("static/") :])

            return url_for("uploaded_media", filename=normalized)

        def enum_label(prefix, value):
            if value is None:
                return ""
            lang = getattr(g, "lang", app.config["DEFAULT_LANGUAGE"])
            key = f"{prefix}_{value}"
            translated = translate(lang, key)
            if translated == key:
                return str(value).replace("_", " ").title()
            return translated

        return {
            "tr": lambda key, **kwargs: translate(getattr(g, "lang", app.config["DEFAULT_LANGUAGE"]), key, **kwargs),
            "current_lang": getattr(g, "lang", app.config["DEFAULT_LANGUAGE"]),
            "available_languages": app.config["LANGUAGES"],
            "site_titles": LANG_CONTENT["titles"],
            "media_url": media_url,
            "enum_label": enum_label,
        }

    @app.route("/set-language/<lang>")
    def set_language(lang):
        if lang in app.config["LANGUAGES"]:
            session["lang"] = lang
            if current_user.is_authenticated:
                current_user.preferred_language = lang
                db.session.commit()
        next_url = request.args.get("next") or request.referrer or url_for("main.home")
        return redirect(next_url)

    @app.route("/media/<path:filename>")
    def uploaded_media(filename):
        normalized = str(filename).replace("\\", "/").lstrip("/")
        return send_from_directory(app.config["STORAGE_ROOT"], normalized)

    @app.errorhandler(404)
    def not_found(error):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(error):
        db.session.rollback()
        return render_template("500.html"), 500

    with app.app_context():
        db.create_all()
        ensure_runtime_schema()
        ensure_default_admin(app)

    return app


def ensure_default_admin(app):
    admin = User.query.filter_by(email=app.config["DEFAULT_ADMIN_EMAIL"]).first()
    if admin:
        return
    admin = User(
        username=app.config["DEFAULT_ADMIN_USERNAME"],
        email=app.config["DEFAULT_ADMIN_EMAIL"],
        role="admin",
        preferred_language=app.config["DEFAULT_LANGUAGE"],
    )
    admin.set_password(app.config["DEFAULT_ADMIN_PASSWORD"])
    db.session.add(admin)
    db.session.flush()
    db.session.add(
        SystemLog(
            action="bootstrap_admin",
            user_id=admin.id,
            description="Default administrator account created during initialization.",
        )
    )
    db.session.commit()


def ensure_runtime_schema():
    inspector = inspect(db.engine)
    user_columns = {column["name"] for column in inspector.get_columns("user")}
    if "biometric_template" not in user_columns:
        db.session.execute(text("ALTER TABLE user ADD COLUMN biometric_template TEXT"))
    if "biometric_reference_path" not in user_columns:
        db.session.execute(text("ALTER TABLE user ADD COLUMN biometric_reference_path VARCHAR(255)"))
    db.session.commit()

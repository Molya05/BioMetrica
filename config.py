from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
DEFAULT_STORAGE_ROOT = INSTANCE_DIR / "storage"


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _running_on_render():
    return _env_flag("RENDER", default=False) or os.environ.get("RENDER_SERVICE_ID") is not None


def _database_uri():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return f"sqlite:///{INSTANCE_DIR / 'biometric_auth.db'}"

    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    if database_url.startswith("postgresql://") and "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


def _path_from_env(name, default_path):
    raw_value = os.environ.get(name)
    if not raw_value:
        return default_path
    return Path(raw_value).expanduser()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "diploma-project-secret-key")
    DEBUG = _env_flag("FLASK_DEBUG", default=False)
    SQLALCHEMY_DATABASE_URI = _database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STORAGE_ROOT = _path_from_env("STORAGE_ROOT", DEFAULT_STORAGE_ROOT)
    UPLOAD_FOLDER = _path_from_env("UPLOAD_FOLDER", STORAGE_ROOT / "uploads")
    PROCESSED_FOLDER = _path_from_env("PROCESSED_FOLDER", UPLOAD_FOLDER / "processed")
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
    DEFAULT_LANGUAGE = "en"
    LANGUAGES = {
        "en": "English",
        "ru": "Русский",
        "kk": "Қазақша",
    }
    DEFAULT_ADMIN_USERNAME = os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
    DEFAULT_ADMIN_EMAIL = os.environ.get("DEFAULT_ADMIN_EMAIL", "admin@biometrica.local")
    DEFAULT_ADMIN_PASSWORD = os.environ.get("DEFAULT_ADMIN_PASSWORD", "Admin123!")
    SESSION_COOKIE_NAME = "biometrica_session"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = _env_flag("SESSION_COOKIE_SECURE", default=_running_on_render())
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_DOMAIN = None
    PREFERRED_URL_SCHEME = os.environ.get(
        "PREFERRED_URL_SCHEME",
        "https" if SESSION_COOKIE_SECURE else "http",
    )
    REMEMBER_COOKIE_SECURE = _env_flag("REMEMBER_COOKIE_SECURE", default=SESSION_COOKIE_SECURE)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_DOMAIN = None

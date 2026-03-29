from functools import wraps

from flask import Blueprint, flash, redirect, render_template, url_for
from flask_login import current_user, login_required

from ..models import Analysis, SystemLog, User


admin_bp = Blueprint("admin", __name__)


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        if not current_user.is_admin:
            flash("message_admin_only", "danger")
            return redirect(url_for("dashboard.dashboard"))
        return view(*args, **kwargs)

    return wrapped_view


@admin_bp.route("/admin")
@admin_required
def admin_panel():
    users = User.query.order_by(User.created_at.desc()).all()
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(10).all()
    logs = SystemLog.query.order_by(SystemLog.timestamp.desc()).limit(12).all()
    return render_template("admin.html", users=users, analyses=analyses, logs=logs)

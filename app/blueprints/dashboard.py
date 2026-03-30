import base64
import binascii
from datetime import datetime, timedelta
import time

import cv2
import numpy as np

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import RequestEntityTooLarge

from ..extensions import db
from ..models import Analysis
from ..services.recognition import analyze_image, compare_templates
from ..translations import translate
from ..utils import allowed_file, dump_feature_summary, log_event, to_public_path, unique_filename


dashboard_bp = Blueprint("dashboard", __name__)


VALIDATION_MESSAGE_MAP = {
    "image_load_error": ("validation_image_load_error", "validation_recapture"),
    "invalid_frame": ("validation_image_load_error", "validation_recapture"),
    "no_face_detected": ("validation_no_reliable_eye_detected", "validation_face_camera"),
    "eye_not_detected": ("validation_eye_not_detected", "validation_move_closer"),
    "eye_region_too_small": ("validation_eye_region_too_small", "validation_move_closer"),
    "face_detected_eyes_too_small": ("validation_face_detected_eyes_too_small", "validation_move_closer"),
    "no_reliable_eye_detected": ("validation_no_reliable_eye_detected", "validation_face_camera"),
    "image_too_blurry": ("validation_image_too_blurry", "validation_use_sharper"),
    "strong_glare": ("validation_strong_glare", "validation_reduce_reflection"),
    "poor_contrast": ("validation_poor_contrast", "validation_improve_lighting"),
    "segmentation_failed": ("validation_eye_partially_blocked", "validation_open_eye"),
    "eye_partially_blocked": ("validation_eye_partially_blocked", "validation_open_eye"),
    "no_biometric_template": ("message_biometric_template_missing", "message_biometric_template_missing_hint"),
}


def _decode_request_frame(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    if image_array.size == 0:
        return None
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame


def _safe_commit(action, *, filename=None):
    try:
        db.session.commit()
        return True
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception(
            "%s database failure user_id=%s filename=%s",
            action,
            current_user.id,
            filename,
        )
        return False


@dashboard_bp.route("/recognition", methods=["GET", "POST"])
@login_required
def recognition():
    analysis_result = None
    if request.method == "POST":
        image = request.files.get("eye_image")
        webcam_image_data = (request.form.get("webcam_image_data") or "").strip()
        captured_frame = None
        saved_path = None
        original_filename = None

        current_app.logger.info(
            "recognition POST received user_id=%s content_length=%s has_file=%s has_camera_data=%s",
            current_user.id,
            request.content_length,
            bool(image and image.filename),
            bool(webcam_image_data),
        )

        try:
            if webcam_image_data:
                try:
                    _, encoded = webcam_image_data.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                except (ValueError, binascii.Error):
                    flash("message_upload_invalid", "danger")
                    return render_template("recognition.html", analysis_result=analysis_result)

                captured_frame = _decode_request_frame(image_bytes)
                if captured_frame is None:
                    flash("message_upload_invalid", "danger")
                    return render_template("recognition.html", analysis_result=analysis_result)

                filename = unique_filename("webcam_capture.png")
                saved_path = current_app.config["UPLOAD_FOLDER"] / filename
                saved_path.write_bytes(image_bytes)
                original_filename = "webcam_capture.png"
            else:
                if not image or not image.filename:
                    flash("message_upload_missing", "danger")
                    return render_template("recognition.html", analysis_result=analysis_result)
                if not allowed_file(image.filename):
                    flash("message_upload_invalid", "danger")
                    return render_template("recognition.html", analysis_result=analysis_result)

                image_bytes = image.read()
                captured_frame = _decode_request_frame(image_bytes)
                if captured_frame is None:
                    flash("message_upload_invalid", "danger")
                    return render_template("recognition.html", analysis_result=analysis_result)

                filename = unique_filename(image.filename)
                saved_path = current_app.config["UPLOAD_FOLDER"] / filename
                saved_path.write_bytes(image_bytes)
                original_filename = image.filename
        except RequestEntityTooLarge:
            flash("message_upload_invalid", "danger")
            return render_template("recognition.html", analysis_result=analysis_result)
        except Exception:
            current_app.logger.exception("recognition input processing crashed user_id=%s", current_user.id)
            flash("message_registration_failed", "danger")
            return render_template("recognition.html", analysis_result=analysis_result)

        analysis_start = time.perf_counter()
        try:
            result = analyze_image(captured_frame, current_app.config["PROCESSED_FOLDER"])
        except Exception:
            current_app.logger.exception("recognition analysis crashed user_id=%s", current_user.id)
            flash("message_registration_failed", "danger")
            return render_template("recognition.html", analysis_result=analysis_result)
        current_app.logger.info(
            "recognition analysis user_id=%s success=%s timings=%s elapsed=%.4fs",
            current_user.id,
            result.get("success"),
            result.get("timings", {}),
            time.perf_counter() - analysis_start,
        )
        error_code = result.get("error_code")
        message_key, recommendation_key = VALIDATION_MESSAGE_MAP.get(
            error_code,
            ("message_analysis_failed", "validation_recapture"),
        )

        if result["success"]:
            biometric_profile = current_user.latest_biometric_profile()
            enrolled_template = biometric_profile.biometric_template_dict() if biometric_profile else current_user.biometric_template_dict()
            if not enrolled_template:
                localized_error = translate(current_user.preferred_language, "message_biometric_template_missing")
                localized_recommendation = translate(current_user.preferred_language, "message_biometric_template_missing_hint")
                analysis_result = {
                    **result,
                    "success": False,
                    "is_valid_image": False,
                    "error_code": "no_biometric_template",
                    "image_path": to_public_path(saved_path),
                    "annotated_path": to_public_path(result["processed_path"]),
                    "error_message": localized_error,
                    "recommendation_message": localized_recommendation,
                }
                flash("message_biometric_template_missing", "danger")
                return render_template("recognition.html", analysis_result=analysis_result)

            comparison_start = time.perf_counter()
            verification_result = compare_templates(result["biometric_template"], enrolled_template)
            comparison_time = round(time.perf_counter() - comparison_start, 4)
            result.setdefault("timings", {})["database_comparison"] = comparison_time
            result["timings"]["final_decision"] = comparison_time
            decision = verification_result["decision"]
            status = "authenticated" if decision == "matched" else "rejected"
            matched_user = current_user.username if decision == "matched" else None
            result["stages"] = {
                **result.get("stages", {}),
                "comparison": "completed",
                "final_result": "completed" if decision == "matched" else "failed",
            }
            record = Analysis(
                user_id=current_user.id,
                filename=original_filename,
                image_path=to_public_path(saved_path),
                annotated_path=to_public_path(result["processed_path"]),
                result_status=status,
                confidence_score=verification_result["similarity_score"],
                processing_time=result["processing_time"],
                feature_summary=dump_feature_summary(
                    {
                        "features": result["features"],
                        "biometric_template": result["biometric_template"],
                        "similarity_score": verification_result["similarity_score"],
                        "decision": decision,
                        "matched_user": matched_user,
                        "stage_images": {key: to_public_path(value) for key, value in result.get("stage_images", {}).items()},
                        "component_scores": verification_result.get("component_scores", {}),
                        "timings": result.get("timings", {}),
                    }
                ),
            )
            current_app.logger.info(
                "recognition compare user_id=%s decision=%s similarity=%s timings=%s",
                current_user.id,
                decision,
                verification_result.get("similarity_score"),
                result.get("timings", {}),
            )
            db.session.add(record)
            log_event(
                "authentication_success",
                f"Authentication completed for {original_filename} with similarity {verification_result['similarity_score']}%.",
                current_user.id,
            )
            if not _safe_commit("recognition_success", filename=original_filename):
                flash("message_registration_failed", "danger")
                return render_template("recognition.html", analysis_result=analysis_result)
            analysis_result = {
                **result,
                **verification_result,
                "status": status,
                "matched_user": matched_user,
                "image_path": to_public_path(saved_path),
                "annotated_path": to_public_path(result["processed_path"]),
                "stage_images": {key: to_public_path(value) for key, value in result.get("stage_images", {}).items()},
                "component_scores": verification_result.get("component_scores", {}),
                "stages": result.get("stages", {}),
                "timings": result.get("timings", {}),
                "error_message": None,
                "recommendation_message": None,
            }
            flash("message_authentication_success", "success")
        else:
            localized_error = translate(current_user.preferred_language, message_key)
            localized_recommendation = translate(current_user.preferred_language, recommendation_key)
            record = Analysis(
                user_id=current_user.id,
                filename=original_filename,
                image_path=to_public_path(saved_path),
                annotated_path=to_public_path(result["processed_path"]) if result.get("processed_path") else None,
                result_status="failed",
                confidence_score=0.0,
                processing_time=result.get("processing_time", 0.0),
                feature_summary=dump_feature_summary(
                    {
                        "error_code": error_code,
                        "error_message": localized_error,
                        "recommendation_message": localized_recommendation,
                        "details": result.get("details", {}),
                        "stage_images": {key: to_public_path(value) for key, value in result.get("stage_images", {}).items()},
                        "timings": result.get("timings", {}),
                    }
                ),
            )
            db.session.add(record)
            log_event("authentication_failed", f"Authentication failed for {original_filename}: {error_code}.", current_user.id)
            if not _safe_commit("recognition_failure", filename=original_filename):
                flash("message_registration_failed", "danger")
                return render_template("recognition.html", analysis_result=analysis_result)
            analysis_result = {
                **result,
                "image_path": to_public_path(saved_path),
                "annotated_path": to_public_path(result["processed_path"]) if result.get("processed_path") else None,
                "stage_images": {key: to_public_path(value) for key, value in result.get("stage_images", {}).items()},
                "stages": result.get("stages", {}),
                "timings": result.get("timings", {}),
                "error_message": localized_error,
                "recommendation_message": localized_recommendation,
            }

    return render_template("recognition.html", analysis_result=analysis_result)


@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    user_analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    total = len(user_analyses)
    successes = len([item for item in user_analyses if item.result_status == "authenticated"])
    avg_confidence = round(sum(item.confidence_score for item in user_analyses) / total, 2) if total else 0
    latest = user_analyses[0] if user_analyses else None
    return render_template(
        "dashboard.html",
        total=total,
        successes=successes,
        avg_confidence=avg_confidence,
        latest=latest,
        recent=user_analyses[:5],
    )


@dashboard_bp.route("/history")
@login_required
def history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    return render_template("history.html", analyses=analyses)


@dashboard_bp.route("/analytics")
@login_required
def analytics():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.asc()).all()
    last_7_days = datetime.utcnow() - timedelta(days=6)
    per_day = (
        db.session.query(func.date(Analysis.created_at), func.count(Analysis.id))
        .filter(Analysis.user_id == current_user.id, Analysis.created_at >= last_7_days)
        .group_by(func.date(Analysis.created_at))
        .all()
    )

    day_map = {}
    for i in range(7):
        day = (last_7_days + timedelta(days=i)).date().isoformat()
        day_map[day] = 0
    for day, count in per_day:
        day_map[str(day)] = count

    authenticated = len([item for item in analyses if item.result_status == "authenticated"])
    rejected = len([item for item in analyses if item.result_status == "rejected"])
    failed = len([item for item in analyses if item.result_status == "failed"])

    chart_data = {
        "days": list(day_map.keys()),
        "counts": list(day_map.values()),
        "confidence_labels": [item.created_at.strftime("%d.%m") for item in analyses[-10:]],
        "confidence_values": [item.confidence_score for item in analyses[-10:]],
        "status_values": [authenticated, rejected, failed],
        "status_labels": [
            translate(current_user.preferred_language, "status_authenticated"),
            translate(current_user.preferred_language, "status_rejected"),
            translate(current_user.preferred_language, "status_failed"),
        ],
        "analysis_dataset_label": translate(current_user.preferred_language, "analytics_dataset_analyses"),
        "confidence_dataset_label": translate(current_user.preferred_language, "analytics_dataset_confidence"),
    }
    return render_template("analytics.html", chart_data=chart_data, analyses=analyses)


@dashboard_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        current_user.username = request.form.get("username", current_user.username).strip()
        current_user.email = request.form.get("email", current_user.email).strip().lower()
        current_user.preferred_language = request.form.get("preferred_language", current_user.preferred_language)
        new_password = request.form.get("password", "").strip()
        if new_password:
            current_user.set_password(new_password)
        log_event("profile_update", f"Profile updated for {current_user.username}.", current_user.id)
        db.session.commit()
        flash("message_profile_updated", "success")
        return redirect(url_for("dashboard.profile"))
    return render_template("profile.html")

import base64
import binascii
from datetime import datetime, timedelta
import time

import cv2
import numpy as np

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db
from ..models import BiometricProfile, User
from ..services.recognition import analyze_image, compare_templates
from ..translations import translate
from ..utils import allowed_file, dump_feature_summary, log_event, to_public_path, unique_filename


auth_bp = Blueprint("auth", __name__)
BIOMETRIC_LOGIN_THRESHOLD = 65.0

REGISTER_VALIDATION_MESSAGE_MAP = {
    "image_load_error": ("validation_image_load_error", "validation_recapture"),
    "invalid_frame": ("validation_image_load_error", "validation_recapture"),
    "no_face_detected": ("validation_no_reliable_eye_detected", "validation_face_camera"),
    "both_eyes_not_visible": ("validation_both_eyes_not_visible", "validation_face_camera"),
    "eye_not_detected": ("validation_eye_not_detected", "validation_move_closer"),
    "eye_region_too_small": ("validation_eye_region_too_small", "validation_move_closer"),
    "face_detected_eyes_too_small": ("validation_face_detected_eyes_too_small", "validation_move_closer"),
    "no_reliable_eye_detected": ("validation_no_reliable_eye_detected", "validation_face_camera"),
    "image_too_blurry": ("validation_image_too_blurry", "validation_use_sharper"),
    "strong_glare": ("validation_strong_glare", "validation_reduce_reflection"),
    "poor_contrast": ("validation_poor_contrast", "validation_improve_lighting"),
    "segmentation_failed": ("validation_eye_partially_blocked", "validation_open_eye"),
    "eye_partially_blocked": ("validation_eye_partially_blocked", "validation_open_eye"),
}


IDENTIFIER_FEEDBACK = {
    "en": {
        "not_found": "No user was found with this username or email.",
        "hint": "Enter a valid username or email to start biometric verification.",
    },
    "ru": {
        "not_found": "Пользователь с таким именем или email не найден.",
        "hint": "Введите корректное имя пользователя или email, чтобы начать биометрическую проверку.",
    },
    "kk": {
        "not_found": "Мұндай пайдаланушы аты немесе email бар пайдаланушы табылмады.",
        "hint": "Биометриялық тексеруді бастау үшін дұрыс пайдаланушы атын немесе email енгізіңіз.",
    },
}


def _registration_context(form_data=None, error_key=None, error_message=None):
    form_data = form_data or {}
    preferred_language = form_data.get("preferred_language", "en")
    if preferred_language not in current_app.config["LANGUAGES"]:
        preferred_language = current_app.config["DEFAULT_LANGUAGE"]
    return {
        "form_error_key": error_key,
        "form_error_message": error_message,
        "form_values": {
            "username": form_data.get("username", "").strip(),
            "email": form_data.get("email", "").strip().lower(),
            "preferred_language": preferred_language,
        },
    }


def _login_context(form_data=None, feedback_key=None, feedback_message=None, feedback_detail_key=None, is_error=True):
    form_data = form_data or {}
    return {
        "login_values": {
            "identifier": form_data.get("identifier", "").strip(),
        },
        "login_feedback": {
            "message_key": feedback_key,
            "message": feedback_message,
            "detail_key": feedback_detail_key,
            "is_error": is_error,
        },
    }


def _save_biometric_input(upload_file, webcam_image_data, fallback_name):
    def _decode_frame(image_bytes):
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        if image_array.size == 0:
            return None
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None
        return frame

    if webcam_image_data:
        try:
            _, encoded = webcam_image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        except (ValueError, binascii.Error):
            return None, None, None, "message_biometric_upload_invalid"

        frame = _decode_frame(image_bytes)
        if frame is None:
            return None, None, None, "message_biometric_upload_invalid"

        filename = unique_filename(fallback_name)
        saved_path = current_app.config["UPLOAD_FOLDER"] / filename
        saved_path.write_bytes(image_bytes)
        return saved_path, fallback_name, frame, None

    if not upload_file or not upload_file.filename:
        return None, None, None, "message_biometric_enrollment_required"

    if not allowed_file(upload_file.filename):
        return None, None, None, "message_biometric_upload_invalid"

    image_bytes = upload_file.read()
    frame = _decode_frame(image_bytes)
    if frame is None:
        return None, None, None, "message_biometric_upload_invalid"

    filename = unique_filename(upload_file.filename)
    saved_path = current_app.config["UPLOAD_FOLDER"] / filename
    saved_path.write_bytes(image_bytes)
    return saved_path, upload_file.filename, frame, None


def _resolve_biometric_feedback(lang, error_code):
    message_key, recommendation_key = REGISTER_VALIDATION_MESSAGE_MAP.get(
        error_code,
        ("message_biometric_enrollment_failed", "validation_recapture"),
    )
    return (
        translate(lang, message_key),
        translate(lang, recommendation_key),
    )


def _resolve_identifier_feedback(lang):
    localized = IDENTIFIER_FEEDBACK.get(lang, IDENTIFIER_FEEDBACK["en"])
    return localized["not_found"], localized["hint"]


def _biometric_login_response(
    *,
    preferred_language,
    http_status,
    success,
    biometric_verified,
    user_found=False,
    threshold=BIOMETRIC_LOGIN_THRESHOLD,
    match_passed=False,
    error_code=None,
    error_message=None,
    recommendation_message=None,
    similarity_score=None,
    matched_user=None,
    result=None,
    saved_path=None,
    extra=None,
):
    result = result or {}
    stages = result.get(
        "stages",
        {
            "frame_received": "pending",
            "face_detection": "pending",
            "eye_detection": "pending",
            "eye_selection": "pending",
            "pupil_segmentation": "pending",
            "iris_segmentation": "pending",
            "feature_extraction": "pending",
            "comparison": "pending",
            "final_result": "pending",
        },
    )
    details = result.get("details", {})
    frame_received = stages.get("frame_received") == "completed" or bool(saved_path)
    face_detected = bool(details.get("face_detected"))
    eye_detected = stages.get("eye_detection") == "completed"
    best_eye_selected = stages.get("eye_selection") == "completed"
    pupil_detected = stages.get("pupil_segmentation") == "completed"
    iris_segmented = stages.get("iris_segmentation") == "completed"
    features_extracted = stages.get("feature_extraction") == "completed"
    comparison_started = stages.get("comparison") in {"completed", "failed"}
    template_found = not (error_code == "biometric_template_missing")
    current_step = extra.get("current_step") if extra else None
    if not current_step:
        if success and biometric_verified:
            current_step = "login_successful"
        elif error_code == "invalid_credentials":
            current_step = "credentials_invalid"
        elif error_code == "user_not_found":
            current_step = "user_lookup_failed"
        elif error_code == "biometric_template_missing":
            current_step = "template_lookup_failed"
        elif error_code == "biometric_match_not_found":
            current_step = "match_not_found"
        elif stages.get("eye_detection") in {"pending", "failed"}:
            current_step = "detecting_eye"
        elif stages.get("pupil_segmentation") in {"pending", "failed"}:
            current_step = "segmenting_pupil"
        elif stages.get("iris_segmentation") in {"pending", "failed"}:
            current_step = "segmenting_iris"
        elif stages.get("feature_extraction") in {"pending", "failed"}:
            current_step = "extracting_features"
        elif stages.get("comparison") in {"pending", "failed"}:
            current_step = "comparing_database"
        else:
            current_step = "login_failed"

    payload = {
        "success": success,
        "biometric_verified": biometric_verified,
        "user_found": user_found,
        "threshold": threshold,
        "match_passed": match_passed,
        "matched": biometric_verified,
        "login_allowed": success and biometric_verified,
        "login_denied": not (success and biometric_verified),
        "is_valid_image": result.get("is_valid_image", success),
        "current_step": current_step,
        "camera_opened": True,
        "frame_captured": frame_received,
        "frame_received": frame_received,
        "face_detected": face_detected,
        "detected_faces": details.get("detected_faces", 0),
        "detected_eyes": details.get("detected_eyes", 0),
        "eye_detected": eye_detected,
        "best_eye_selected": best_eye_selected,
        "selected_eye_side": details.get("selected_eye_side"),
        "selected_eye_score": details.get("selected_eye_score"),
        "pupil_detected": pupil_detected,
        "iris_segmented": iris_segmented,
        "features_extracted": features_extracted,
        "template_found": template_found,
        "comparison_started": comparison_started,
        "error_code": error_code,
        "error_message": error_message,
        "recommendation_message": recommendation_message,
        "similarity_score": similarity_score,
        "matched_user": matched_user,
        "message": error_message if not success else translate(preferred_language, "message_biometric_login_success"),
        "processing_time": result.get("processing_time", 0.0),
        "timings": result.get("timings", {}),
        "debug_details": details,
        "stages": stages,
        "stage_images": {key: to_public_path(value) for key, value in result.get("stage_images", {}).items()},
        "annotated_path": to_public_path(result["processed_path"]) if result.get("processed_path") else None,
        "image_path": to_public_path(saved_path) if saved_path else None,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload), http_status


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")
        form_data = {"identifier": identifier}
        user = User.query.filter(or_(User.username == identifier, User.email == identifier)).first()
        if not user or not user.check_password(password):
            session.pop("pending_biometric_user_id", None)
            session.pop("pending_biometric_verified_at", None)
            session.pop("pending_biometric_similarity", None)
            flash("message_invalid_credentials", "danger")
            return render_template(
                "login.html",
                **_login_context(
                    form_data,
                    feedback_key="message_invalid_credentials",
                    feedback_detail_key="message_enter_credentials_first",
                ),
            )

        login_user(user)
        session.pop("pending_biometric_user_id", None)
        session.pop("pending_biometric_verified_at", None)
        session.pop("pending_biometric_similarity", None)
        log_event("login", f"User {user.username} signed in.", user.id)
        db.session.commit()
        flash("message_login_success", "success")
        next_page = request.args.get("next")
        return redirect(next_page or url_for("dashboard.dashboard"))

    return render_template("login.html", **_login_context())


@auth_bp.route("/login/biometric", methods=["POST"])
def biometric_login():
    upload_file = request.files.get("biometric_image")
    webcam_image_data = (request.form.get("webcam_image_data") or "").strip()
    identifier = request.form.get("identifier", "").strip()
    preferred_language = request.form.get("preferred_language", current_app.config["DEFAULT_LANGUAGE"])
    if preferred_language not in current_app.config["LANGUAGES"]:
        preferred_language = current_app.config["DEFAULT_LANGUAGE"]

    session.pop("pending_biometric_user_id", None)
    session.pop("pending_biometric_verified_at", None)
    session.pop("pending_biometric_similarity", None)

    if not identifier:
        error_message, recommendation_message = _resolve_identifier_feedback(preferred_language)
        return _biometric_login_response(
            preferred_language=preferred_language,
            http_status=400,
            success=False,
            biometric_verified=False,
            user_found=False,
            match_passed=False,
            error_code="user_not_found",
            error_message=error_message,
            recommendation_message=recommendation_message,
            result={
                "is_valid_image": False,
                "stages": {
                    "eye_detection": "failed",
                    "pupil_segmentation": "failed",
                    "iris_segmentation": "failed",
                    "feature_extraction": "failed",
                    "comparison": "failed",
                    "final_result": "failed",
                },
            },
            extra={"current_step": "user_lookup_failed"},
        )

    user = User.query.filter(or_(User.username == identifier, User.email == identifier)).first()
    if not user:
        error_message, recommendation_message = _resolve_identifier_feedback(preferred_language)
        return _biometric_login_response(
            preferred_language=preferred_language,
            http_status=401,
            success=False,
            biometric_verified=False,
            user_found=False,
            match_passed=False,
            error_code="user_not_found",
            error_message=error_message,
            recommendation_message=recommendation_message,
            result={
                "is_valid_image": False,
                "stages": {
                    "eye_detection": "failed",
                    "pupil_segmentation": "failed",
                    "iris_segmentation": "failed",
                    "feature_extraction": "failed",
                    "comparison": "failed",
                    "final_result": "failed",
                },
            },
            extra={"current_step": "user_lookup_failed"},
        )

    saved_path, _, captured_frame, input_error = _save_biometric_input(upload_file, webcam_image_data, "biometric_login_capture.png")
    if input_error:
        return _biometric_login_response(
            preferred_language=preferred_language,
            http_status=400,
            success=False,
            biometric_verified=False,
            user_found=True,
            match_passed=False,
            error_code=input_error,
            error_message=translate(preferred_language, input_error),
            recommendation_message=translate(preferred_language, "validation_recapture"),
            result={
                "is_valid_image": False,
                "stages": {
                    "eye_detection": "failed",
                    "pupil_segmentation": "failed",
                    "iris_segmentation": "failed",
                    "feature_extraction": "failed",
                    "comparison": "failed",
                    "final_result": "failed",
                },
            },
            extra={"current_step": "capturing_image"},
        )

    analysis_start = time.perf_counter()
    result = analyze_image(captured_frame, current_app.config["PROCESSED_FOLDER"], require_both_eyes=True)
    current_app.logger.info(
        "biometric_login analysis identifier=%s success=%s timings=%s elapsed=%.4fs",
        identifier,
        result.get("success"),
        result.get("timings", {}),
        time.perf_counter() - analysis_start,
    )
    if not result["success"]:
        error_message, recommendation_message = _resolve_biometric_feedback(preferred_language, result.get("error_code"))
        return _biometric_login_response(
            preferred_language=preferred_language,
            http_status=400,
            success=False,
            biometric_verified=False,
            user_found=True,
            match_passed=False,
            error_code=result.get("error_code"),
            error_message=error_message,
            recommendation_message=recommendation_message,
            result=result,
            saved_path=saved_path,
            extra={},
        )

    biometric_profile = user.latest_biometric_profile()
    if not biometric_profile:
        return _biometric_login_response(
            preferred_language=preferred_language,
            http_status=400,
            success=False,
            biometric_verified=False,
            user_found=True,
            match_passed=False,
            error_code="biometric_template_missing",
            error_message=translate(preferred_language, "message_biometric_template_missing"),
            recommendation_message=translate(preferred_language, "message_biometric_template_missing_hint"),
            result={
                **result,
                "stages": {
                    **result.get("stages", {}),
                    "comparison": "failed",
                    "final_result": "failed",
                },
            },
            saved_path=saved_path,
            extra={"current_step": "template_lookup_failed"},
        )

    enrolled_template = biometric_profile.biometric_template_dict()
    comparison_start = time.perf_counter()
    match_result = compare_templates(result["biometric_template"], enrolled_template, threshold=BIOMETRIC_LOGIN_THRESHOLD)
    comparison_time = round(time.perf_counter() - comparison_start, 4)
    result.setdefault("timings", {})["database_comparison"] = comparison_time
    result["timings"]["final_decision"] = comparison_time
    current_app.logger.info(
        "biometric_login compare identifier=%s decision=%s similarity=%s timings=%s",
        identifier,
        match_result.get("decision"),
        match_result.get("similarity_score"),
        result.get("timings", {}),
    )

    if match_result["decision"] != "matched":
        return _biometric_login_response(
            preferred_language=preferred_language,
            http_status=401,
            success=False,
            biometric_verified=False,
            user_found=True,
            match_passed=False,
            threshold=BIOMETRIC_LOGIN_THRESHOLD,
            error_code="biometric_similarity_below_threshold",
            error_message=translate(preferred_language, "message_biometric_match_failed_threshold"),
            recommendation_message=translate(
                preferred_language,
                "message_biometric_similarity_detail",
                similarity=match_result["similarity_score"],
                threshold=int(BIOMETRIC_LOGIN_THRESHOLD),
            ),
            similarity_score=match_result["similarity_score"],
            matched_user=user.username,
            result={
                **result,
                "stages": {
                    **result.get("stages", {}),
                    "comparison": "completed",
                    "final_result": "failed",
                },
            },
            saved_path=saved_path,
            extra={
                "component_scores": match_result.get("component_scores", {}),
                "current_step": "match_not_found",
            },
        )

    login_user(user)
    session["pending_biometric_user_id"] = user.id
    session["pending_biometric_verified_at"] = datetime.utcnow().isoformat()
    session["pending_biometric_similarity"] = match_result["similarity_score"]
    log_event(
        "biometric_login",
        f"User {user.username} signed in with biometric verification at similarity {match_result['similarity_score']}%.",
        user.id,
    )
    db.session.commit()
    return _biometric_login_response(
        preferred_language=preferred_language,
        http_status=200,
        success=True,
        biometric_verified=True,
        user_found=True,
        match_passed=True,
        threshold=BIOMETRIC_LOGIN_THRESHOLD,
        similarity_score=match_result["similarity_score"],
        matched_user=user.username,
        result={
            **result,
            "stages": {
                **result.get("stages", {}),
                "comparison": "completed",
                "final_result": "completed",
            },
        },
        saved_path=saved_path,
        extra={
            "matched_profile_id": biometric_profile.id,
            "component_scores": match_result.get("component_scores", {}),
            "redirect_url": url_for("dashboard.dashboard"),
            "current_step": "login_successful",
            "message": translate(preferred_language, "message_biometric_match_success_threshold"),
            "recommendation_message": translate(
                preferred_language,
                "message_biometric_similarity_detail",
                similarity=match_result["similarity_score"],
                threshold=int(BIOMETRIC_LOGIN_THRESHOLD),
            ),
        },
    )


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        biometric_image = request.files.get("biometric_image")
        webcam_image_data = (request.form.get("webcam_image_data") or "").strip()
        preferred_language = request.form.get("preferred_language", current_app.config["DEFAULT_LANGUAGE"])
        if preferred_language not in current_app.config["LANGUAGES"]:
            preferred_language = current_app.config["DEFAULT_LANGUAGE"]

        form_data = {
            "username": username,
            "email": email,
            "preferred_language": preferred_language,
        }

        if not username or not email or not password or not confirm_password:
            flash("message_required_fields", "danger")
            return render_template("register.html", **_registration_context(form_data, "message_required_fields"))

        if password != confirm_password:
            flash("message_password_mismatch", "danger")
            return render_template("register.html", **_registration_context(form_data, "message_password_mismatch"))

        existing = User.query.filter(or_(User.username == username, User.email == email)).first()
        if existing:
            flash("message_user_exists", "danger")
            return render_template("register.html", **_registration_context(form_data, "message_user_exists"))

        biometric_path, original_name, captured_frame, input_error = _save_biometric_input(
            biometric_image,
            webcam_image_data,
            "registration_capture.png",
        )
        if input_error:
            flash(input_error, "danger")
            return render_template("register.html", **_registration_context(form_data, input_error))

        enrollment_result = analyze_image(captured_frame, current_app.config["PROCESSED_FOLDER"])
        current_app.logger.info(
            "registration biometric username=%s success=%s timings=%s",
            username,
            enrollment_result.get("success"),
            enrollment_result.get("timings", {}),
        )
        if not enrollment_result["success"]:
            localized_error, localized_recommendation = _resolve_biometric_feedback(
                preferred_language,
                enrollment_result.get("error_code"),
            )
            flash("message_biometric_enrollment_failed", "danger")
            return render_template(
                "register.html",
                **_registration_context(form_data, error_message=f"{localized_error} {localized_recommendation}"),
            )

        try:
            user = User(username=username, email=email, preferred_language=preferred_language, role="user")
            user.set_password(password)
            user.biometric_template = dump_feature_summary(enrollment_result["biometric_template"])
            user.biometric_reference_path = to_public_path(biometric_path)
            db.session.add(user)
            db.session.flush()
            profile = BiometricProfile(
                user_id=user.id,
                enrollment_image_path=to_public_path(biometric_path),
                processed_image_path=to_public_path(enrollment_result["processed_path"]),
                biometric_template=dump_feature_summary(enrollment_result["biometric_template"]),
                feature_summary=dump_feature_summary(
                    {
                        "features": enrollment_result["features"],
                        "stage_images": {key: to_public_path(value) for key, value in enrollment_result.get("stage_images", {}).items()},
                    }
                ),
            )
            db.session.add(profile)
            log_event("register", f"New user {username} registered.", user.id)
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            flash("message_registration_failed", "danger")
            return render_template("register.html", **_registration_context(form_data, "message_registration_failed"))

        flash("message_registered", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html", **_registration_context())


@auth_bp.route("/logout")
@login_required
def logout():
    log_event("logout", f"User {current_user.username} signed out.", current_user.id)
    db.session.commit()
    logout_user()
    flash("message_logout", "info")
    return redirect(url_for("main.home"))

from pathlib import Path
import time
import uuid

import cv2
import numpy as np


FACE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
EYE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
MAX_PROCESSING_DIMENSION = 1280
MAX_DETECTION_DIMENSION = 960


def _stage_status_template():
    return {
        "frame_received": "pending",
        "frame_decoded": "pending",
        "face_detection": "pending",
        "eye_detection": "pending",
        "eye_selection": "pending",
        "pupil_segmentation": "pending",
        "iris_segmentation": "pending",
        "feature_extraction": "pending",
        "comparison": "pending",
        "final_result": "pending",
    }


def _timing_template():
    return {
        "frame_received": 0.0,
        "frame_decoded": 0.0,
        "preprocessing": 0.0,
        "face_detected": 0.0,
        "eyes_detected": 0.0,
        "best_eye_selected": 0.0,
        "segmentation": 0.0,
        "feature_extraction": 0.0,
        "database_comparison": 0.0,
        "final_decision": 0.0,
        "total": 0.0,
    }


def _build_failure(
    error_code,
    details,
    processing_time,
    *,
    processed_path=None,
    stage_images=None,
    stages=None,
    timings=None,
):
    return {
        "success": False,
        "is_valid_image": False,
        "error_code": error_code,
        "error_message": None,
        "details": details,
        "processing_time": processing_time,
        "processed_path": str(processed_path) if processed_path else None,
        "stage_images": stage_images or {},
        "stages": stages or _stage_status_template(),
        "timings": timings or _timing_template(),
        "status": "failed",
    }


def compare_templates(candidate_template, enrolled_template, threshold=82):
    comparable_keys = [
        ("darkness_index", 160.0, 0.16),
        ("contrast_index", 70.0, 0.16),
        ("edge_density", 1.0, 0.12),
        ("pupil_ratio", 0.25, 0.18),
        ("iris_ratio", 0.45, 0.18),
        ("symmetry_score", 1.0, 0.10),
        ("iris_code", 255.0, 0.10),
    ]

    distance = 0.0
    component_scores = {}
    for key, scale, weight in comparable_keys:
        candidate_value = float(candidate_template.get(key, 0.0))
        enrolled_value = float(enrolled_template.get(key, 0.0))
        component_distance = min(abs(candidate_value - enrolled_value) / scale, 1.0)
        component_scores[key] = round((1.0 - component_distance) * 100, 2)
        distance += component_distance * weight

    similarity = round(max(0.0, (1.0 - distance) * 100), 2)
    return {
        "similarity_score": similarity,
        "decision": "matched" if similarity >= threshold else "not_matched",
        "threshold": threshold,
        "component_scores": component_scores,
    }


def _ensure_processed_dir(processed_dir):
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def _stage_output_path(processed_dir, source_name, suffix):
    source_name = Path(str(source_name)).stem
    return processed_dir / f"{source_name}_{suffix}.png"


def _save_stage_image(image, processed_dir, source_name, suffix):
    processed_dir = _ensure_processed_dir(processed_dir)
    stage_path = _stage_output_path(processed_dir, source_name, suffix)
    if not cv2.imwrite(str(stage_path), image):
        raise RuntimeError(f"failed_to_write_stage_image:{suffix}")
    return str(stage_path)


def _save_invalid_preview(frame, processed_dir, source_name, label, region=None):
    preview = frame.copy()
    if region is not None:
        x, y, w, h = region
        cv2.rectangle(preview, (x, y), (x + w, y + h), (80, 160, 255), 2)
    cv2.putText(
        preview,
        label,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (80, 160, 255),
        2,
    )
    return _save_stage_image(preview, processed_dir, source_name, "processed")


def _resize_if_needed(frame, max_dimension):
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return frame, 1.0

    scale = max_dimension / float(longest_side)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _scale_region(region, scale_factor):
    x, y, w, h = region
    return (
        int(round(x / scale_factor)),
        int(round(y / scale_factor)),
        int(round(w / scale_factor)),
        int(round(h / scale_factor)),
    )


def _normalize_region(region, frame_shape):
    x, y, w, h = region
    max_height, max_width = frame_shape[:2]
    x = max(0, min(x, max_width - 1))
    y = max(0, min(y, max_height - 1))
    w = max(1, min(w, max_width - x))
    h = max(1, min(h, max_height - y))
    return (x, y, w, h)


def _eye_candidate_metrics(gray_frame, region):
    x, y, w, h = region
    eye_gray = gray_frame[y : y + h, x : x + w]
    area = w * h
    blur_score = float(cv2.Laplacian(eye_gray, cv2.CV_64F).var())
    glare_ratio = float(np.count_nonzero(eye_gray >= 245) / eye_gray.size)
    contrast = float(np.std(eye_gray))
    score = (
        min(area / 12000, 1.0) * 0.4
        + min(blur_score / 180, 1.0) * 0.25
        + min(contrast / 65, 1.0) * 0.2
        + max(0.0, 1.0 - (glare_ratio / 0.06)) * 0.15
    )
    return {
        "region": region,
        "area": area,
        "blur_score": blur_score,
        "glare_ratio": glare_ratio,
        "contrast": contrast,
        "score": score,
    }


def _eye_side(face_region, eye_region):
    fx, _, fw, _ = face_region
    x, _, w, _ = eye_region
    eye_center = x + (w / 2.0)
    face_center = fx + (fw / 2.0)
    return "left" if eye_center < face_center else "right"


def _detect_best_eye_region(gray, *, require_both_eyes=False):
    detection_gray, detection_scale = _resize_if_needed(gray, MAX_DETECTION_DIMENSION)

    faces_small = FACE_CASCADE.detectMultiScale(
        detection_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
    )
    face_regions = [_normalize_region(_scale_region(tuple(face), detection_scale), gray.shape) for face in faces_small]
    face_candidates = []
    ranked_eyes = []
    small_eye_count = 0

    for face_region in face_regions:
        fx, fy, fw, fh = face_region
        upper_face = gray[fy : fy + int(fh * 0.6), fx : fx + fw]
        if upper_face.size == 0:
            continue

        face_detection_gray, face_scale = _resize_if_needed(upper_face, 420)
        detected_eyes = EYE_CASCADE.detectMultiScale(
            face_detection_gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(max(22, face_detection_gray.shape[1] // 10), max(16, face_detection_gray.shape[0] // 12)),
        )

        eye_regions = []
        small_eye_regions = []
        for ex, ey, ew, eh in detected_eyes:
            scaled_region = (
                int(round(ex / face_scale)) + fx,
                int(round(ey / face_scale)) + fy,
                int(round(ew / face_scale)),
                int(round(eh / face_scale)),
            )
            global_region = _normalize_region(scaled_region, gray.shape)
            gx, gy, gw, gh = global_region
            if gw * gh < 3000 or gw < 48 or gh < 24:
                small_eye_regions.append(global_region)
                small_eye_count += 1
            else:
                eye_regions.append(global_region)
                metrics = _eye_candidate_metrics(gray, global_region)
                metrics["source"] = "face"
                metrics["face_region"] = face_region
                metrics["eye_side"] = _eye_side(face_region, global_region)
                ranked_eyes.append(metrics)

        face_candidates.append(
            {
                "face_region": face_region,
                "eye_regions": eye_regions,
                "small_eye_regions": small_eye_regions,
            }
        )

    if require_both_eyes and face_candidates:
        max_detected_eyes = max((len(item["eye_regions"]) for item in face_candidates), default=0)
        if max_detected_eyes < 2:
            error_code = "both_eyes_not_visible"
            if small_eye_count > 0:
                error_code = "face_detected_eyes_too_small"
            return None, {
                "face_detected": True,
                "detected_faces": len(face_regions),
                "detected_eyes": max_detected_eyes,
                "small_eye_regions": small_eye_count,
                "require_both_eyes": True,
            }, error_code, face_regions

    if ranked_eyes:
        ranked_eyes.sort(key=lambda item: (item["score"], item["area"], item["blur_score"], item["contrast"]), reverse=True)
        best_eye = ranked_eyes[0]
        return best_eye["region"], {
            "selection_source": "face_region",
            "face_detected": True,
            "detected_faces": len(face_regions),
            "detected_eyes": len(ranked_eyes),
            "reliable_eye_count": len(ranked_eyes),
            "selected_eye_score": round(best_eye["score"], 4),
            "selected_eye_side": best_eye.get("eye_side", "unknown"),
            "selected_eye_region": tuple(int(value) for value in best_eye["region"]),
            "used_full_frame": True,
            "require_both_eyes": require_both_eyes,
        }, None, face_regions

    direct_detection_gray, direct_scale = _resize_if_needed(gray, MAX_DETECTION_DIMENSION)
    direct_eyes_small = EYE_CASCADE.detectMultiScale(
        direct_detection_gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(48, 36),
    )

    if len(direct_eyes_small) == 0:
        error_code = "no_face_detected" if len(face_regions) == 0 else "no_reliable_eye_detected"
        return None, {
            "face_detected": False,
            "detected_faces": len(face_regions),
            "detected_eyes": 0,
            "require_both_eyes": require_both_eyes,
        }, error_code, face_regions

    direct_regions = [_normalize_region(_scale_region(tuple(eye), direct_scale), gray.shape) for eye in direct_eyes_small]
    ranked_direct_eyes = [_eye_candidate_metrics(gray, region) for region in direct_regions]
    ranked_direct_eyes.sort(key=lambda item: (item["score"], item["area"], item["blur_score"], item["contrast"]), reverse=True)
    best_eye = ranked_direct_eyes[0]
    return best_eye["region"], {
        "selection_source": "full_image",
        "face_detected": False,
        "detected_faces": 0,
        "detected_eyes": len(ranked_direct_eyes),
        "reliable_eye_count": len(ranked_direct_eyes),
        "selected_eye_score": round(best_eye["score"], 4),
        "selected_eye_side": "unknown",
        "selected_eye_region": tuple(int(value) for value in best_eye["region"]),
        "used_full_frame": True,
        "require_both_eyes": require_both_eyes,
    }, None, face_regions


def _build_detection_stage(frame, faces, selected_eye):
    detection_image = frame.copy()
    for fx, fy, fw, fh in faces:
        cv2.rectangle(detection_image, (fx, fy), (fx + fw, fy + fh), (80, 140, 255), 2)
    x, y, w, h = selected_eye
    cv2.rectangle(detection_image, (x, y), (x + w, y + h), (28, 230, 255), 2)
    cv2.putText(
        detection_image,
        "Eye Detection",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (28, 230, 255),
        2,
    )
    return detection_image


def _segment_eye(eye_gray):
    blurred = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    _, pupil_thresh = cv2.threshold(blurred, 42, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return pupil_thresh


def _estimate_iris_radius(gray_eye, pupil_center, pupil_radius, eye_width, eye_height):
    circles = cv2.HoughCircles(
        gray_eye,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(eye_width // 4, 20),
        param1=80,
        param2=18,
        minRadius=max(int(pupil_radius * 1.6), 12),
        maxRadius=max(int(min(eye_width, eye_height) * 0.48), 18),
    )
    if circles is not None:
        rounded = np.round(circles[0, :]).astype("int")
        px, py = pupil_center
        best = min(rounded, key=lambda circle: abs(circle[0] - px) + abs(circle[1] - py))
        return int(best[2])
    return min(int(pupil_radius * 2.4), max(12, int(min(eye_width, eye_height) * 0.45)))


def _load_frame(image_input):
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return None
        return image_input.copy()

    frame = cv2.imread(str(image_input))
    if frame is None or frame.size == 0:
        return None
    return frame


def analyze_image(image_input, processed_dir, require_both_eyes=False):
    total_start = time.perf_counter()
    processed_dir = _ensure_processed_dir(processed_dir)
    source_name = image_input if not isinstance(image_input, np.ndarray) else f"captured_frame_{uuid.uuid4().hex[:12]}"
    stage_images = {}
    stage_statuses = _stage_status_template()
    timings = _timing_template()

    frame_load_start = time.perf_counter()
    frame = _load_frame(image_input)
    timings["frame_received"] = round(time.perf_counter() - frame_load_start, 4)

    if frame is None:
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "invalid_frame",
            {"technical_note": "unable_to_decode_frame"},
            timings["total"],
            stages=stage_statuses,
            timings=timings,
        )

    stage_statuses["frame_received"] = "completed"
    stage_statuses["frame_decoded"] = "completed"

    resize_start = time.perf_counter()
    frame, processing_scale = _resize_if_needed(frame, MAX_PROCESSING_DIMENSION)
    timings["frame_decoded"] = round(time.perf_counter() - resize_start, 4)
    stage_images["original"] = _save_stage_image(frame, processed_dir, source_name, "original")

    preprocess_start = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    timings["preprocessing"] = round(time.perf_counter() - preprocess_start, 4)
    stage_images["preprocessing"] = _save_stage_image(gray, processed_dir, source_name, "preprocessing")

    detection_start = time.perf_counter()
    selected_eye, selection_details, selection_error, faces = _detect_best_eye_region(gray, require_both_eyes=require_both_eyes)
    detection_duration = round(time.perf_counter() - detection_start, 4)
    timings["face_detected"] = detection_duration
    timings["eyes_detected"] = detection_duration
    timings["best_eye_selected"] = detection_duration
    stage_statuses["face_detection"] = "completed" if selection_details.get("face_detected") else "failed"

    if selection_error:
        label = "Validation: no reliable eye"
        if selection_error == "face_detected_eyes_too_small":
            label = "Validation: eyes too small"
        elif selection_error == "both_eyes_not_visible":
            label = "Validation: both eyes required"
        elif selection_error == "no_face_detected":
            label = "Validation: no face detected"
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, label)
        stage_images["final"] = processed_path
        stage_statuses["eye_detection"] = "failed"
        stage_statuses["eye_selection"] = "failed"
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            selection_error,
            {
                **selection_details,
                "processing_scale": round(processing_scale, 4),
            },
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    stage_statuses["face_detection"] = "completed"
    stage_statuses["eye_detection"] = "completed"
    stage_statuses["eye_selection"] = "completed"
    stage_images["eye_detection"] = _save_stage_image(
        _build_detection_stage(frame, faces, selected_eye),
        processed_dir,
        source_name,
        "eye_detection",
    )

    x, y, w, h = selected_eye
    eye_gray = gray[y : y + h, x : x + w]
    eye_area = w * h

    if eye_gray.size == 0:
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, "Validation: invalid eye region", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["eye_selection"] = "failed"
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "no_reliable_eye_detected",
            {**selection_details, "technical_note": "empty_eye_region"},
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    if eye_area < 9000 or w < 95 or h < 55:
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, "Validation: eye region too small", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "eye_region_too_small",
            {
                **selection_details,
                "detected_eye_width": int(w),
                "detected_eye_height": int(h),
                "eye_area": int(eye_area),
            },
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    blur_score = float(cv2.Laplacian(eye_gray, cv2.CV_64F).var())
    if blur_score < 70:
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, "Validation: image too blurry", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "image_too_blurry",
            {**selection_details, "blur_score": round(blur_score, 2), "minimum_required": 70},
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    glare_ratio = float(np.count_nonzero(eye_gray >= 245) / eye_gray.size)
    if glare_ratio > 0.035:
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, "Validation: strong glare detected", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "strong_glare",
            {**selection_details, "glare_ratio": round(glare_ratio, 4), "maximum_allowed": 0.035},
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    contrast = float(np.std(eye_gray))
    if contrast < 26:
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, "Validation: low contrast", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "poor_contrast",
            {**selection_details, "contrast_index": round(contrast, 2), "minimum_required": 26},
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    segmentation_start = time.perf_counter()
    pupil_thresh = _segment_eye(eye_gray)
    stage_statuses["pupil_segmentation"] = "completed"
    stage_images["pupil_segmentation"] = _save_stage_image(pupil_thresh, processed_dir, source_name, "pupil_segmentation")

    contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if 30 < cv2.contourArea(cnt) < (w * h * 0.35)]
    if not valid_contours:
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, "Validation: segmentation failed", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["iris_segmentation"] = "failed"
        timings["segmentation"] = round(time.perf_counter() - segmentation_start, 4)
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "segmentation_failed",
            {**selection_details, "candidate_contours": 0, "technical_note": "pupil_segmentation_failed"},
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    pupil = max(valid_contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(pupil)
    cx, cy, radius = int(cx), int(cy), max(int(radius), 5)
    _, pupil_y, _, pupil_h = cv2.boundingRect(pupil)

    if pupil_y <= int(h * 0.06) or (pupil_y + pupil_h) >= int(h * 0.94) or radius < 6:
        processed_path = _save_invalid_preview(frame, processed_dir, source_name, "Validation: eyelid blockage", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["iris_segmentation"] = "failed"
        timings["segmentation"] = round(time.perf_counter() - segmentation_start, 4)
        timings["total"] = round(time.perf_counter() - total_start, 4)
        return _build_failure(
            "segmentation_failed",
            {
                **selection_details,
                "estimated_pupil_radius": int(radius),
                "pupil_top_margin": int(pupil_y),
                "pupil_bottom_margin": int(h - (pupil_y + pupil_h)),
            },
            timings["total"],
            processed_path=processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
            timings=timings,
        )

    iris_radius = _estimate_iris_radius(eye_gray, (cx, cy), radius, w, h)
    iris_overlay = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(iris_overlay, (cx, cy), iris_radius, (25, 240, 193), 2)
    cv2.circle(iris_overlay, (cx, cy), radius, (255, 102, 0), 2)
    stage_statuses["iris_segmentation"] = "completed"
    stage_images["iris_segmentation"] = _save_stage_image(iris_overlay, processed_dir, source_name, "iris_segmentation")
    timings["segmentation"] = round(time.perf_counter() - segmentation_start, 4)

    feature_start = time.perf_counter()
    pupil_mask = np.zeros_like(eye_gray)
    cv2.circle(pupil_mask, (cx, cy), radius, 255, -1)
    iris_mask = np.zeros_like(eye_gray)
    cv2.circle(iris_mask, (cx, cy), iris_radius, 255, -1)
    cv2.circle(iris_mask, (cx, cy), radius, 0, -1)

    masked_pixels = eye_gray[pupil_mask == 255]
    iris_pixels = eye_gray[iris_mask == 255]
    mean_darkness = float(255 - np.mean(masked_pixels)) if masked_pixels.size else 0.0
    iris_texture = float(np.std(iris_pixels)) if iris_pixels.size else 0.0
    edges = cv2.Canny(eye_gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    pupil_ratio = float((np.pi * (radius ** 2)) / (w * h))
    iris_ratio = float((np.pi * (iris_radius ** 2)) / (w * h))
    left_half = eye_gray[:, : eye_gray.shape[1] // 2]
    right_half = cv2.flip(eye_gray[:, eye_gray.shape[1] // 2 :], 1)
    min_width = min(left_half.shape[1], right_half.shape[1])
    symmetry = 1.0 - float(np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width])) / 255)

    feature_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        feature_overlay,
        "Feature Extraction",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (25, 240, 193),
        2,
    )
    stage_statuses["feature_extraction"] = "completed"
    stage_images["feature_extraction"] = _save_stage_image(feature_overlay, processed_dir, source_name, "feature_extraction")
    timings["feature_extraction"] = round(time.perf_counter() - feature_start, 4)

    detection_quality = min(1.0, eye_area / 18000)
    normalized_darkness = min(1.0, mean_darkness / 160)
    normalized_contrast = min(1.0, contrast / 70)
    normalized_edges = min(1.0, edge_density * 8)
    normalized_ratio = 1.0 - min(abs(pupil_ratio - 0.11) / 0.11, 1.0)
    normalized_symmetry = max(0.0, min(symmetry, 1.0))

    confidence = (
        0.24 * detection_quality
        + 0.18 * normalized_darkness
        + 0.16 * normalized_contrast
        + 0.12 * normalized_edges
        + 0.10 * normalized_ratio
        + 0.10 * normalized_symmetry
        + 0.10 * min(iris_texture / 55, 1.0)
    ) * 100
    confidence = round(float(max(0.0, min(confidence, 99.2))), 2)
    status = "authenticated" if confidence >= 72 else "rejected"

    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (28, 230, 255), 2)
    cv2.circle(annotated, (x + cx, y + cy), radius, (255, 102, 0), 2)
    cv2.circle(annotated, (x + cx, y + cy), iris_radius, (25, 240, 193), 2)
    cv2.putText(
        annotated,
        f"Quality: {confidence:.2f}%",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (28, 230, 255),
        2,
    )
    cv2.putText(
        annotated,
        "Biometric Template Ready",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (25, 240, 193),
        2,
    )

    processed_path = _save_stage_image(annotated, processed_dir, source_name, "processed")
    stage_images["final"] = processed_path
    timings["total"] = round(time.perf_counter() - total_start, 4)

    features = {
        "detected_eye_width": int(w),
        "detected_eye_height": int(h),
        "estimated_pupil_radius": int(radius),
        "estimated_iris_radius": int(iris_radius),
        "darkness_index": round(mean_darkness, 2),
        "contrast_index": round(contrast, 2),
        "edge_density": round(edge_density, 4),
        "pupil_ratio": round(pupil_ratio, 4),
        "iris_ratio": round(iris_ratio, 4),
        "iris_texture_index": round(iris_texture, 2),
        "symmetry_score": round(normalized_symmetry, 4),
        "selection_source": selection_details["selection_source"],
    }
    biometric_template = {
        "darkness_index": round(mean_darkness, 2),
        "contrast_index": round(contrast, 2),
        "edge_density": round(edge_density, 4),
        "pupil_ratio": round(pupil_ratio, 4),
        "iris_ratio": round(iris_ratio, 4),
        "symmetry_score": round(normalized_symmetry, 4),
        "iris_code": int((radius * 31 + iris_radius * 17 + int(mean_darkness) + int(contrast * 10)) % 256),
    }

    return {
        "success": True,
        "is_valid_image": True,
        "error_code": None,
        "error_message": None,
        "details": {
            **selection_details,
            "frame_received": True,
            "best_eye_selected": True,
            "processing_scale": round(processing_scale, 4),
        },
        "status": status,
        "confidence_score": confidence,
        "processing_time": timings["total"],
        "features": features,
        "biometric_template": biometric_template,
        "processed_path": processed_path,
        "stage_images": stage_images,
        "stages": stage_statuses,
        "timings": timings,
    }

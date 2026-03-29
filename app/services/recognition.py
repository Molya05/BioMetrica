from pathlib import Path
import time

import cv2
import numpy as np


def _build_failure(error_code, details, processing_time, processed_path=None, stage_images=None, stages=None):
    return {
        "success": False,
        "is_valid_image": False,
        "error_code": error_code,
        "error_message": None,
        "details": details,
        "processing_time": processing_time,
        "processed_path": str(processed_path) if processed_path else None,
        "stage_images": stage_images or {},
        "stages": stages or {},
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
    source_name = Path(source_name).stem
    return processed_dir / f"{source_name}_{suffix}.png"


def _save_stage_image(image, processed_dir, source_name, suffix):
    processed_dir = _ensure_processed_dir(processed_dir)
    stage_path = _stage_output_path(processed_dir, source_name, suffix)
    cv2.imwrite(str(stage_path), image)
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


def _detect_best_eye_region(gray, require_both_eyes=False):
    face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    face_candidates = []

    for fx, fy, fw, fh in faces:
        upper_face = gray[fy : fy + int(fh * 0.6), fx : fx + fw]
        detected_eyes = eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(max(26, fw // 10), max(18, fh // 12)),
        )

        eye_regions = []
        small_eye_regions = []
        for ex, ey, ew, eh in detected_eyes:
            global_region = (fx + ex, fy + ey, ew, eh)
            if ew * eh < 3000 or ew < 48 or eh < 24:
                small_eye_regions.append(global_region)
            else:
                eye_regions.append(global_region)

        face_candidates.append(
            {
                "face_region": (fx, fy, fw, fh),
                "eye_regions": eye_regions,
                "small_eye_regions": small_eye_regions,
            }
        )

    ranked_eyes = []
    for face_candidate in face_candidates:
        for region in face_candidate["eye_regions"]:
            metrics = _eye_candidate_metrics(gray, region)
            metrics["source"] = "face"
            metrics["face_region"] = face_candidate["face_region"]
            metrics["eye_side"] = _eye_side(face_candidate["face_region"], region)
            ranked_eyes.append(metrics)

    if require_both_eyes and face_candidates:
        max_detected_eyes = max((len(item["eye_regions"]) for item in face_candidates), default=0)
        if max_detected_eyes < 2:
            total_small = sum(len(item["small_eye_regions"]) for item in face_candidates)
            error_code = "both_eyes_not_visible"
            if total_small > 0:
                error_code = "face_detected_eyes_too_small"
            return None, {
                "face_detected": True,
                "detected_faces": len(faces),
                "detected_eyes": max_detected_eyes,
                "small_eye_regions": total_small,
                "require_both_eyes": True,
            }, error_code, faces

    if ranked_eyes:
        ranked_eyes.sort(key=lambda item: (item["score"], item["area"], item["blur_score"], item["contrast"]), reverse=True)
        best_eye = ranked_eyes[0]
        details = {
            "selection_source": "face_region",
            "face_detected": True,
            "detected_faces": len(faces),
            "detected_eyes": len(ranked_eyes),
            "reliable_eye_count": len(ranked_eyes),
            "selected_eye_score": round(best_eye["score"], 4),
            "selected_eye_side": best_eye.get("eye_side", "unknown"),
            "selected_eye_region": tuple(int(value) for value in best_eye["region"]),
            "used_full_frame": True,
            "require_both_eyes": require_both_eyes,
        }
        return best_eye["region"], details, None, faces

    if len(faces) > 0:
        total_small = sum(len(item["small_eye_regions"]) for item in face_candidates)
        if total_small > 0:
            return None, {"face_detected": True, "detected_faces": len(faces), "small_eye_regions": total_small}, "face_detected_eyes_too_small", faces
        return None, {"face_detected": True, "detected_faces": len(faces)}, "no_reliable_eye_detected", faces

    direct_eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
    if len(direct_eyes) == 0:
        return None, {"face_detected": False, "detected_faces": 0, "detected_eyes": 0}, "no_reliable_eye_detected", faces

    direct_regions = [tuple(int(value) for value in eye) for eye in direct_eyes]
    ranked_direct_eyes = [_eye_candidate_metrics(gray, region) for region in direct_regions]
    ranked_direct_eyes.sort(key=lambda item: (item["score"], item["area"], item["blur_score"], item["contrast"]), reverse=True)
    best_eye = ranked_direct_eyes[0]
    details = {
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
    }
    return best_eye["region"], details, None, faces


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
    blurred = cv2.GaussianBlur(eye_gray, (9, 9), 0)
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


def analyze_image(image_path, processed_dir, require_both_eyes=False):
    start = time.perf_counter()
    processed_dir = _ensure_processed_dir(processed_dir)
    frame = cv2.imread(str(image_path))
    stage_images = {}
    stage_statuses = {
        "frame_received": "pending",
        "face_detection": "pending",
        "eye_detection": "pending",
        "eye_selection": "pending",
        "pupil_segmentation": "pending",
        "iris_segmentation": "pending",
        "feature_extraction": "pending",
        "comparison": "pending",
        "final_result": "pending",
    }

    if frame is None:
        elapsed = round(time.perf_counter() - start, 3)
        return _build_failure("image_load_error", {"technical_note": "unable_to_open_image"}, elapsed, stages=stage_statuses)

    stage_statuses["frame_received"] = "completed"
    stage_images["original"] = _save_stage_image(frame, processed_dir, image_path, "original")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    stage_images["preprocessing"] = _save_stage_image(gray, processed_dir, image_path, "preprocessing")

    selected_eye, selection_details, selection_error, faces = _detect_best_eye_region(gray, require_both_eyes=require_both_eyes)
    stage_statuses["face_detection"] = "completed" if selection_details.get("face_detected") else "failed"
    if selection_error:
        elapsed = round(time.perf_counter() - start, 3)
        label = "Validation: no reliable eye"
        if selection_error == "face_detected_eyes_too_small":
            label = "Validation: eyes too small"
        elif selection_error == "both_eyes_not_visible":
            label = "Validation: both eyes required"
        processed_path = _save_invalid_preview(frame, processed_dir, image_path, label)
        stage_images["final"] = processed_path
        stage_statuses["eye_detection"] = "failed"
        stage_statuses["eye_selection"] = "failed"
        return _build_failure(
            selection_error,
            selection_details,
            elapsed,
            processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
        )

    stage_statuses["eye_detection"] = "completed"
    stage_statuses["eye_selection"] = "completed"
    stage_images["eye_detection"] = _save_stage_image(
        _build_detection_stage(frame, faces, selected_eye),
        processed_dir,
        image_path,
        "eye_detection",
    )

    x, y, w, h = selected_eye
    eye_gray = gray[y : y + h, x : x + w]
    eye_color = frame[y : y + h, x : x + w]
    eye_area = w * h

    if eye_area < 9000 or w < 95 or h < 55:
        elapsed = round(time.perf_counter() - start, 3)
        processed_path = _save_invalid_preview(frame, processed_dir, image_path, "Validation: eye region too small", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        return _build_failure(
            "eye_region_too_small",
            {
                **selection_details,
                "detected_eye_width": int(w),
                "detected_eye_height": int(h),
                "eye_area": int(eye_area),
            },
            elapsed,
            processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
        )

    blur_score = float(cv2.Laplacian(eye_gray, cv2.CV_64F).var())
    if blur_score < 70:
        elapsed = round(time.perf_counter() - start, 3)
        processed_path = _save_invalid_preview(frame, processed_dir, image_path, "Validation: image too blurry", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        return _build_failure(
            "image_too_blurry",
            {**selection_details, "blur_score": round(blur_score, 2), "minimum_required": 70},
            elapsed,
            processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
        )

    glare_ratio = float(np.count_nonzero(eye_gray >= 245) / eye_gray.size)
    if glare_ratio > 0.035:
        elapsed = round(time.perf_counter() - start, 3)
        processed_path = _save_invalid_preview(frame, processed_dir, image_path, "Validation: strong glare detected", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        return _build_failure(
            "strong_glare",
            {**selection_details, "glare_ratio": round(glare_ratio, 4), "maximum_allowed": 0.035},
            elapsed,
            processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
        )

    contrast = float(np.std(eye_gray))
    if contrast < 26:
        elapsed = round(time.perf_counter() - start, 3)
        processed_path = _save_invalid_preview(frame, processed_dir, image_path, "Validation: low contrast", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["pupil_segmentation"] = "failed"
        return _build_failure(
            "poor_contrast",
            {**selection_details, "contrast_index": round(contrast, 2), "minimum_required": 26},
            elapsed,
            processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
        )

    pupil_thresh = _segment_eye(eye_gray)
    stage_statuses["pupil_segmentation"] = "completed"
    stage_images["pupil_segmentation"] = _save_stage_image(pupil_thresh, processed_dir, image_path, "pupil_segmentation")

    contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if 30 < cv2.contourArea(cnt) < (w * h * 0.35)]
    if not valid_contours:
        elapsed = round(time.perf_counter() - start, 3)
        processed_path = _save_invalid_preview(frame, processed_dir, image_path, "Validation: pupil blocked", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["iris_segmentation"] = "failed"
        return _build_failure(
            "eye_partially_blocked",
            {**selection_details, "candidate_contours": len(valid_contours), "technical_note": "pupil_segmentation_failed"},
            elapsed,
            processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
        )

    pupil = max(valid_contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(pupil)
    cx, cy, radius = int(cx), int(cy), max(int(radius), 5)
    _, pupil_y, _, pupil_h = cv2.boundingRect(pupil)

    if pupil_y <= int(h * 0.06) or (pupil_y + pupil_h) >= int(h * 0.94) or radius < 6:
        elapsed = round(time.perf_counter() - start, 3)
        processed_path = _save_invalid_preview(frame, processed_dir, image_path, "Validation: eyelid blockage", selected_eye)
        stage_images["final"] = processed_path
        stage_statuses["iris_segmentation"] = "failed"
        return _build_failure(
            "eye_partially_blocked",
            {
                **selection_details,
                "estimated_pupil_radius": int(radius),
                "pupil_top_margin": int(pupil_y),
                "pupil_bottom_margin": int(h - (pupil_y + pupil_h)),
            },
            elapsed,
            processed_path,
            stage_images=stage_images,
            stages=stage_statuses,
        )

    iris_radius = _estimate_iris_radius(eye_gray, (cx, cy), radius, w, h)
    iris_overlay = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(iris_overlay, (cx, cy), iris_radius, (25, 240, 193), 2)
    cv2.circle(iris_overlay, (cx, cy), radius, (255, 102, 0), 2)
    stage_statuses["iris_segmentation"] = "completed"
    stage_images["iris_segmentation"] = _save_stage_image(iris_overlay, processed_dir, image_path, "iris_segmentation")

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
    stage_images["feature_extraction"] = _save_stage_image(feature_overlay, processed_dir, image_path, "feature_extraction")

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

    processed_path = _save_stage_image(annotated, processed_dir, image_path, "processed")
    matching_preview = annotated.copy()
    cv2.putText(
        matching_preview,
        "Matching Prototype",
        (20, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 220, 120),
        2,
    )
    stage_statuses["comparison"] = "completed"
    stage_statuses["final_result"] = "completed"
    stage_images["matching"] = _save_stage_image(matching_preview, processed_dir, image_path, "matching")
    stage_images["final"] = processed_path

    elapsed = round(time.perf_counter() - start, 3)
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
        },
        "status": status,
        "confidence_score": confidence,
        "processing_time": elapsed,
        "features": features,
        "biometric_template": biometric_template,
        "processed_path": processed_path,
        "stage_images": stage_images,
        "stages": stage_statuses,
    }

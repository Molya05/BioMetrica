document.addEventListener("DOMContentLoaded", () => {
    initMenu();
    initRecognitionForm();
    initRegistrationEnrollment();
    initBiometricLogin();

    const payload = window.analyticsPayload;
    if (payload && window.Chart) {
        buildCharts(payload);
    }
});

function isLocalhostHost(hostname) {
    return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
}

function normalizeAppPath(path) {
    if (!path) {
        return "";
    }
    if (/^https?:\/\//i.test(path) || path.startsWith("data:")) {
        return path;
    }
    return path.startsWith("/") ? path : `/${path}`;
}

function initMenu() {
    const menuToggle = document.querySelector("[data-menu-toggle]");
    const nav = document.querySelector(".site-nav");
    const backdrop = document.querySelector(".site-nav-backdrop");
    const body = document.body;
    if (menuToggle && nav) {
        const closeMenu = () => {
            nav.classList.remove("open");
            backdrop?.classList.remove("open");
            menuToggle.setAttribute("aria-expanded", "false");
            body.classList.remove("menu-open");
        };

        const toggleMenu = () => {
            const nextOpen = !nav.classList.contains("open");
            nav.classList.toggle("open", nextOpen);
            backdrop?.classList.toggle("open", nextOpen);
            menuToggle.setAttribute("aria-expanded", nextOpen ? "true" : "false");
            body.classList.toggle("menu-open", nextOpen);
        };

        menuToggle.addEventListener("click", toggleMenu);
        backdrop?.addEventListener("click", closeMenu);
        nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
        window.addEventListener("resize", () => {
            if (window.innerWidth > 760) {
                closeMenu();
            }
        });
    }
}

function createCameraController(config) {
    const {
        input,
        preview,
        hiddenInput,
        openButton,
        captureButton,
        closeButton,
        panel,
        video,
        canvas,
        status,
        readyMessage,
        deniedMessage,
        unavailableMessage,
        insecureContextMessage,
        captureMaxDimension,
        outputMimeType,
        outputQuality,
    } = config;

    let stream = null;

    const setStatus = (message) => {
        if (status) {
            status.textContent = message || "";
        }
    };

    const setPreview = (src) => {
        if (preview) {
            preview.innerHTML = `<img src="${src}" alt="Biometric eye preview">`;
        }
    };

    const restorePreview = () => {
        if (preview) {
            preview.innerHTML = `<span>${preview.dataset.placeholder || "Preview"}</span>`;
        }
    };

    const clearCaptured = () => {
        if (hiddenInput) {
            hiddenInput.value = "";
        }
    };

    const stop = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }
        if (video) {
            video.srcObject = null;
        }
        if (panel) {
            panel.hidden = true;
        }
        if (captureButton) {
            captureButton.hidden = true;
        }
        if (closeButton) {
            closeButton.hidden = true;
        }
        if (openButton) {
            openButton.hidden = false;
        }
    };

    const open = async () => {
        clearCaptured();
        if (input) {
            input.value = "";
        }
        restorePreview();

        if (!window.isSecureContext && !isLocalhostHost(window.location.hostname)) {
            setStatus(insecureContextMessage || unavailableMessage);
            return false;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setStatus(unavailableMessage);
            return false;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user" },
                audio: false,
            });
            video.srcObject = stream;
            panel.hidden = false;
            captureButton.hidden = false;
            closeButton.hidden = false;
            openButton.hidden = true;
            setStatus(readyMessage);
            return true;
        } catch (error) {
            const denied = error && (error.name === "NotAllowedError" || error.name === "SecurityError");
            setStatus(denied ? deniedMessage : unavailableMessage);
            stop();
            return false;
        }
    };

    const capture = () => {
        if (!stream || !video || !canvas) {
            setStatus(unavailableMessage);
            return null;
        }

        const width = video.videoWidth;
        const height = video.videoHeight;
        if (!width || !height) {
            setStatus(unavailableMessage);
            return null;
        }

        let targetWidth = width;
        let targetHeight = height;
        if (captureMaxDimension && Math.max(width, height) > captureMaxDimension) {
            const scale = captureMaxDimension / Math.max(width, height);
            targetWidth = Math.max(1, Math.round(width * scale));
            targetHeight = Math.max(1, Math.round(height * scale));
        }

        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const context = canvas.getContext("2d");
        if (!context) {
            setStatus(unavailableMessage);
            return null;
        }

        context.drawImage(video, 0, 0, targetWidth, targetHeight);
        const dataUrl = canvas.toDataURL(outputMimeType || "image/png", outputQuality);
        if (hiddenInput) {
            hiddenInput.value = dataUrl;
        }
        setPreview(dataUrl);
        stop();
        setStatus("");
        return dataUrl;
    };

    if (preview) {
        preview.dataset.placeholder = preview.textContent.trim();
    }

    if (input) {
        input.addEventListener("change", (event) => {
            clearCaptured();
            stop();
            setStatus("");

            const file = event.target.files[0];
            if (!file) {
                restorePreview();
                return;
            }

            const reader = new FileReader();
            reader.onload = (readerEvent) => setPreview(readerEvent.target.result);
            reader.readAsDataURL(file);
        });
    }

    if (openButton) {
        openButton.addEventListener("click", open);
    }

    if (captureButton) {
        captureButton.addEventListener("click", capture);
    }

    if (closeButton) {
        closeButton.addEventListener("click", () => {
            stop();
            setStatus("");
            if (!hasSource()) {
                restorePreview();
            }
        });
    }

    const hasSource = () => Boolean((input && input.files && input.files.length > 0) || (hiddenInput && hiddenInput.value));

    window.addEventListener("beforeunload", stop);
    window.addEventListener("pageshow", stop);

    return {
        stop,
        open,
        capture,
        clearCaptured,
        setStatus,
        setPreview,
        restorePreview,
        hasSource,
    };
}

function initRecognitionForm() {
    const form = document.getElementById("analysis-form");
    const overlay = document.getElementById("loading-overlay");
    const button = document.getElementById("analyze-button");
    if (!form || !overlay || !button) {
        return;
    }

    const camera = createCameraController({
        input: document.getElementById("eye-image-input"),
        preview: document.getElementById("image-preview"),
        hiddenInput: document.getElementById("webcam-image-data"),
        openButton: document.getElementById("open-camera-button"),
        captureButton: document.getElementById("capture-button"),
        closeButton: document.getElementById("close-camera-button"),
        panel: document.getElementById("camera-panel"),
        video: document.getElementById("camera-video"),
        canvas: document.getElementById("camera-canvas"),
        status: document.getElementById("camera-status"),
        readyMessage: form.dataset.cameraReady,
        deniedMessage: form.dataset.cameraDenied,
        unavailableMessage: form.dataset.cameraUnavailable,
        insecureContextMessage: form.dataset.cameraHttpsRequired || form.dataset.cameraUnavailable,
    });

    const defaultButtonLabel = button.textContent;
    const resetUi = () => {
        overlay.hidden = true;
        button.disabled = false;
        button.textContent = defaultButtonLabel;
    };

    resetUi();
    window.addEventListener("pageshow", resetUi);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        overlay.hidden = false;
        button.disabled = true;
        button.textContent = form.dataset.processingLabel || defaultButtonLabel;

        try {
            if (!camera.hasSource()) {
                camera.setStatus(form.dataset.validationMessage || "Please select an image file before submitting.");
                return;
            }

            if (!form.reportValidity()) {
                return;
            }

            await new Promise((resolve) => window.requestAnimationFrame(resolve));
            camera.stop();
            form.submit();
        } catch (error) {
            resetUi();
            throw error;
        } finally {
            if (!form.checkValidity() || !camera.hasSource()) {
                resetUi();
            }
        }
    });
}

function initRegistrationEnrollment() {
    const form = document.getElementById("registration-form");
    if (!form) {
        return;
    }

    const submitButton = document.getElementById("register-submit-button");
    const errorBox = document.getElementById("register-form-error");
    const defaultButtonLabel = submitButton ? submitButton.textContent : "";
    let isSubmitting = false;

    const camera = createCameraController({
        input: document.getElementById("register-biometric-image-input"),
        preview: document.getElementById("register-image-preview"),
        hiddenInput: document.getElementById("register-webcam-image-data"),
        openButton: document.getElementById("register-open-camera-button"),
        captureButton: document.getElementById("register-capture-button"),
        closeButton: document.getElementById("register-close-camera-button"),
        panel: document.getElementById("register-camera-panel"),
        video: document.getElementById("register-camera-video"),
        canvas: document.getElementById("register-camera-canvas"),
        status: document.getElementById("register-camera-status"),
        readyMessage: form.dataset.cameraReady,
        deniedMessage: form.dataset.cameraDenied,
        unavailableMessage: form.dataset.cameraUnavailable,
        insecureContextMessage: form.dataset.cameraHttpsRequired || form.dataset.cameraUnavailable,
        captureMaxDimension: 1280,
        outputMimeType: "image/jpeg",
        outputQuality: 0.86,
    });

    const showError = (message) => {
        if (errorBox) {
            errorBox.textContent = message || "";
            errorBox.hidden = !message;
        }
    };

    const resetSubmitState = () => {
        isSubmitting = false;
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = defaultButtonLabel;
        }
    };

    showError("");
    resetSubmitState();
    window.addEventListener("pageshow", resetSubmitState);

    form.addEventListener("submit", async (event) => {
        if (isSubmitting) {
            event.preventDefault();
            return;
        }

        event.preventDefault();
        showError("");
        camera.setStatus("");

        if (!form.reportValidity()) {
            console.warn("Registration form blocked by browser validation.");
            return;
        }

        if (!camera.hasSource()) {
            const message = form.dataset.validationMessage || "Please provide a biometric image before registering.";
            console.warn("Registration blocked: biometric source missing.");
            camera.setStatus(message);
            showError(message);
            return;
        }

        isSubmitting = true;
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = form.dataset.processingLabel || defaultButtonLabel;
        }

        try {
            form.action = form.dataset.submitUrl || form.getAttribute("action") || window.location.pathname;
            console.info("Submitting registration form", {
                action: form.action,
                method: (form.getAttribute("method") || "post").toUpperCase(),
                source: document.getElementById("register-webcam-image-data")?.value ? "camera" : "file",
            });
            camera.stop();
            await new Promise((resolve) => window.requestAnimationFrame(resolve));
            HTMLFormElement.prototype.submit.call(form);
        } catch (error) {
            console.error("Registration submit failed before request was sent.", error);
            const message = form.dataset.cameraUnavailable || "Registration could not be submitted.";
            showError(message);
            camera.setStatus(message);
            resetSubmitState();
        }
    });
}

function initBiometricLogin() {
    const biometricPanel = document.getElementById("biometric-login-form");
    if (!biometricPanel) {
        return;
    }

    const resultCard = document.getElementById("login-biometric-result");
    const resultMessage = document.getElementById("login-biometric-message");
    const resultDetail = document.getElementById("login-biometric-detail");
    const stageGrid = document.getElementById("login-stage-grid");
    const loadingOverlay = document.getElementById("login-loading-overlay");
    const processingStep = document.getElementById("login-processing-step");
    const faceIdShell = document.getElementById("login-faceid-shell");
    const faceIdOrb = document.getElementById("login-faceid-orb");
    const faceIdIcon = document.getElementById("login-faceid-icon");
    const stageCopy = document.getElementById("login-stage-copy");
    const identifierInput = document.getElementById("login-identifier-input");
    const hiddenImageInput = document.getElementById("login-webcam-image-data");
    const captureButton = document.getElementById("login-capture-button");
    let requestInFlight = false;
    let redirectScheduled = false;

    const camera = createCameraController({
        input: null,
        preview: document.getElementById("login-image-preview"),
        hiddenInput: hiddenImageInput,
        openButton: document.getElementById("login-open-camera-button"),
        captureButton: document.getElementById("login-capture-button"),
        closeButton: document.getElementById("login-close-camera-button"),
        panel: document.getElementById("login-camera-panel"),
        video: document.getElementById("login-camera-video"),
        canvas: document.getElementById("login-camera-canvas"),
        status: document.getElementById("login-camera-status"),
        readyMessage: biometricPanel.dataset.cameraReady,
        deniedMessage: biometricPanel.dataset.cameraDenied,
        unavailableMessage: biometricPanel.dataset.cameraUnavailable,
        insecureContextMessage: biometricPanel.dataset.cameraHttpsRequired,
    });

    const stageElements = stageGrid ? Array.from(stageGrid.querySelectorAll("[data-stage]")) : [];
    const diagnostics = {
        userFound: document.getElementById("diag-user-found"),
        cameraOpened: document.getElementById("diag-camera-opened"),
        frameCaptured: document.getElementById("diag-frame-captured"),
        faceDetected: document.getElementById("diag-face-detected"),
        eyeDetected: document.getElementById("diag-eye-detected"),
        bestEyeSelected: document.getElementById("diag-best-eye-selected"),
        pupilDetected: document.getElementById("diag-pupil-detected"),
        irisSegmented: document.getElementById("diag-iris-segmented"),
        featureExtracted: document.getElementById("diag-feature-extracted"),
        templateFound: document.getElementById("diag-template-found"),
        comparisonStarted: document.getElementById("diag-comparison-started"),
        similarityScore: document.getElementById("diag-similarity-score"),
        finalDecision: document.getElementById("diag-final-decision"),
        loginDecision: document.getElementById("diag-login-decision"),
    };

    const stepLabels = {
        waiting: biometricPanel.dataset.waiting,
        capturing_image: biometricPanel.dataset.processingLabel || biometricPanel.dataset.waiting,
        credentials_invalid: biometricPanel.dataset.enterCredentials,
        user_lookup_failed: biometricPanel.dataset.userNotFound || biometricPanel.dataset.enterCredentials,
        frame_received: biometricPanel.dataset.frameReceived,
        face_detection: biometricPanel.dataset.faceDetection,
        detecting_eye: biometricPanel.dataset.detectingEye,
        eye_selection: biometricPanel.dataset.selectingEye,
        segmenting_pupil: biometricPanel.dataset.scanning,
        segmenting_iris: biometricPanel.dataset.scanning,
        extracting_features: biometricPanel.dataset.extractingFeatures,
        comparing_database: biometricPanel.dataset.comparingDatabase,
        template_lookup_failed: biometricPanel.dataset.matchNotFound,
        match_not_found: biometricPanel.dataset.matchNotFound,
        login_successful: biometricPanel.dataset.success,
        login_failed: biometricPanel.dataset.matchNotFound,
    };

    const setDiagnostic = (element, text, state = "") => {
        if (!element) {
            return;
        }
        element.textContent = text;
        element.className = state;
    };

    const setScannerState = (state, text) => {
        if (faceIdShell) {
            faceIdShell.classList.remove("scanner-idle", "scanner-processing", "scanner-success", "scanner-error");
            faceIdShell.classList.add(`scanner-${state}`);
        }
        if (faceIdOrb) {
            faceIdOrb.classList.remove("scanner-idle", "scanner-processing", "scanner-success", "scanner-error");
            faceIdOrb.classList.add(`scanner-${state}`);
        }
        if (stageCopy) {
            stageCopy.textContent = text || biometricPanel.dataset.waiting;
        }
        if (faceIdIcon) {
            if (state === "success") {
                faceIdIcon.textContent = "✓";
            } else if (state === "error") {
                faceIdIcon.textContent = "!";
            } else {
                faceIdIcon.textContent = "◎";
            }
        }
    };

    const setProcessingState = (visible, text = "") => {
        if (loadingOverlay) {
            loadingOverlay.hidden = !visible;
        }
        if (processingStep) {
            processingStep.textContent = text || biometricPanel.dataset.comparingDatabase;
        }
    };

    const resolveStepMessage = (result, fallback = "") => {
        if (result?.error_message) {
            return result.error_message;
        }
        if (result?.current_step && stepLabels[result.current_step]) {
            return stepLabels[result.current_step];
        }
        return fallback || biometricPanel.dataset.waiting;
    };

    const updateDiagnosticsFromResult = (result) => {
        const userFoundState = result.user_found ? "authenticated" : "failed";
        const userFoundText = result.user_found ? biometricPanel.dataset.diagUserFound : biometricPanel.dataset.diagUserNotFound;
        setDiagnostic(diagnostics.userFound, userFoundText, userFoundState);
        setDiagnostic(diagnostics.cameraOpened, result.camera_opened ? biometricPanel.dataset.diagCompleted : biometricPanel.dataset.diagPending, result.camera_opened ? "authenticated" : "");
        setDiagnostic(diagnostics.frameCaptured, result.frame_received ? biometricPanel.dataset.diagCaptured : biometricPanel.dataset.diagFailed, result.frame_received ? "authenticated" : "failed");
        setDiagnostic(diagnostics.faceDetected, result.face_detected ? biometricPanel.dataset.diagYes : biometricPanel.dataset.diagNo, result.face_detected ? "authenticated" : "failed");
        setDiagnostic(diagnostics.eyeDetected, result.eye_detected ? `${biometricPanel.dataset.diagDetected} (${result.detected_eyes || 0})` : biometricPanel.dataset.diagNotDetected, result.eye_detected ? "authenticated" : "failed");
        setDiagnostic(diagnostics.bestEyeSelected, result.best_eye_selected ? `${biometricPanel.dataset.diagYes}${result.selected_eye_side ? ` (${result.selected_eye_side})` : ""}` : biometricPanel.dataset.diagNo, result.best_eye_selected ? "authenticated" : "failed");
        setDiagnostic(diagnostics.pupilDetected, result.pupil_detected ? biometricPanel.dataset.diagDetected : biometricPanel.dataset.diagFailed, result.pupil_detected ? "authenticated" : "failed");
        setDiagnostic(diagnostics.irisSegmented, result.iris_segmented ? biometricPanel.dataset.diagSegmented : biometricPanel.dataset.diagFailed, result.iris_segmented ? "authenticated" : "failed");
        setDiagnostic(diagnostics.featureExtracted, result.features_extracted ? biometricPanel.dataset.diagCompleted : biometricPanel.dataset.diagFailed, result.features_extracted ? "authenticated" : "failed");
        setDiagnostic(diagnostics.templateFound, result.template_found ? biometricPanel.dataset.diagFound : biometricPanel.dataset.diagNotFound, result.template_found ? "authenticated" : "failed");
        setDiagnostic(diagnostics.comparisonStarted, result.comparison_started ? biometricPanel.dataset.diagStarted : biometricPanel.dataset.diagPending, result.comparison_started ? "authenticated" : "");
        setDiagnostic(diagnostics.similarityScore, result.similarity_score ? `${result.similarity_score}%` : "-", result.similarity_score ? "authenticated" : "");
        setDiagnostic(diagnostics.finalDecision, result.match_passed ? biometricPanel.dataset.diagMatchFound : biometricPanel.dataset.diagNoMatch, result.match_passed ? "authenticated" : "failed");
        setDiagnostic(diagnostics.loginDecision, result.login_allowed ? biometricPanel.dataset.diagAllowed : biometricPanel.dataset.diagDenied, result.login_allowed ? "authenticated" : "failed");
    };

    const paintStages = (statuses) => {
        stageElements.forEach((element) => {
            element.classList.remove("pending", "completed", "failed");
            const state = statuses[element.dataset.stage] || "pending";
            element.classList.add(state);
        });
    };

    const showResult = (message, detail, isError) => {
        if (!resultCard || !resultMessage || !resultDetail) {
            return;
        }
        resultCard.classList.toggle("result-success", !isError);
        resultCard.classList.toggle("result-error", Boolean(isError));
        resultMessage.textContent = message || "";
        resultDetail.textContent = detail || "";
    };

    const setCaptureBusy = (nextValue) => {
        requestInFlight = nextValue;
        if (captureButton) {
            captureButton.disabled = nextValue;
        }
    };

    const startProcessingSequence = () => {
        const sequence = [
            {
                delay: 0,
                text: biometricPanel.dataset.findingUser,
                update: () => {
                    setScannerState("processing", biometricPanel.dataset.findingUser);
                    setDiagnostic(diagnostics.userFound, biometricPanel.dataset.diagInProgress);
                    setDiagnostic(diagnostics.templateFound, biometricPanel.dataset.diagPending);
                    setDiagnostic(diagnostics.comparisonStarted, biometricPanel.dataset.diagPending);
                },
            },
            {
                delay: 500,
                text: biometricPanel.dataset.frameReceived,
                update: () => {
                    setScannerState("processing", biometricPanel.dataset.frameReceived);
                    setDiagnostic(diagnostics.frameCaptured, biometricPanel.dataset.diagCaptured, "authenticated");
                    setDiagnostic(diagnostics.faceDetected, biometricPanel.dataset.diagInProgress);
                },
            },
            {
                delay: 1000,
                text: biometricPanel.dataset.faceDetection,
                update: () => {
                    setScannerState("processing", biometricPanel.dataset.faceDetection);
                    setDiagnostic(diagnostics.eyeDetected, biometricPanel.dataset.diagInProgress);
                    setDiagnostic(diagnostics.bestEyeSelected, biometricPanel.dataset.diagPending);
                },
            },
            {
                delay: 1500,
                text: biometricPanel.dataset.detectingEye,
                update: () => {
                    setScannerState("processing", biometricPanel.dataset.detectingEye);
                    setDiagnostic(diagnostics.eyeDetected, biometricPanel.dataset.diagInProgress);
                    setDiagnostic(diagnostics.bestEyeSelected, biometricPanel.dataset.diagInProgress);
                    setDiagnostic(diagnostics.pupilDetected, biometricPanel.dataset.diagInProgress);
                    setDiagnostic(diagnostics.irisSegmented, biometricPanel.dataset.diagPending);
                    setDiagnostic(diagnostics.featureExtracted, biometricPanel.dataset.diagPending);
                    setDiagnostic(diagnostics.templateFound, biometricPanel.dataset.diagPending);
                    setDiagnostic(diagnostics.comparisonStarted, biometricPanel.dataset.diagPending);
                },
            },
            {
                delay: 2200,
                text: biometricPanel.dataset.scanning,
                update: () => {
                    setScannerState("processing", biometricPanel.dataset.scanning);
                    setDiagnostic(diagnostics.irisSegmented, biometricPanel.dataset.diagInProgress);
                    setDiagnostic(diagnostics.featureExtracted, biometricPanel.dataset.diagInProgress);
                },
            },
            {
                delay: 2900,
                text: biometricPanel.dataset.comparingDatabase,
                update: () => {
                    setScannerState("processing", biometricPanel.dataset.comparingDatabase);
                    setDiagnostic(diagnostics.templateFound, biometricPanel.dataset.diagInProgress);
                    setDiagnostic(diagnostics.comparisonStarted, biometricPanel.dataset.diagInProgress);
                },
            },
        ];

        const timers = sequence.map((step) => window.setTimeout(() => {
            setProcessingState(true, step.text);
            step.update();
        }, step.delay));

        return () => timers.forEach((timerId) => window.clearTimeout(timerId));
    };

    const resetVerification = () => {
        setCaptureBusy(false);
        redirectScheduled = false;
        paintStages({
            frame_received: "pending",
            face_detection: "pending",
            eye_detection: "pending",
            eye_selection: "pending",
            pupil_segmentation: "pending",
            iris_segmentation: "pending",
            feature_extraction: "pending",
            comparison: "pending",
            final_result: "pending",
        });
        setDiagnostic(diagnostics.userFound, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.cameraOpened, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.frameCaptured, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.faceDetected, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.eyeDetected, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.bestEyeSelected, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.pupilDetected, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.irisSegmented, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.featureExtracted, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.templateFound, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.comparisonStarted, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.similarityScore, "-");
        setDiagnostic(diagnostics.finalDecision, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.loginDecision, biometricPanel.dataset.diagPending);
        setProcessingState(false, biometricPanel.dataset.comparingDatabase);
        setScannerState("idle", biometricPanel.dataset.cameraWaiting || biometricPanel.dataset.waiting);
        showResult(biometricPanel.dataset.waiting, biometricPanel.dataset.required, false);
    };

    resetVerification();

    [identifierInput].forEach((input) => {
        input?.addEventListener("input", resetVerification);
    });

    captureButton?.addEventListener("click", async () => {
        if (requestInFlight) {
            return;
        }

        if (!identifierInput?.value.trim()) {
            setDiagnostic(diagnostics.userFound, biometricPanel.dataset.diagUserNotFound, "failed");
            setScannerState("error", biometricPanel.dataset.enterCredentials);
            showResult(biometricPanel.dataset.required, biometricPanel.dataset.enterCredentials, true);
            return;
        }

        const capturedImage = camera.capture();
        if (!capturedImage) {
            setDiagnostic(diagnostics.frameCaptured, biometricPanel.dataset.diagFailed, "failed");
            setScannerState("error", biometricPanel.dataset.cameraUnavailable);
            return;
        }

        paintStages({
            frame_received: "pending",
            face_detection: "pending",
            eye_detection: "pending",
            eye_selection: "pending",
            pupil_segmentation: "pending",
            iris_segmentation: "pending",
            feature_extraction: "pending",
            comparison: "pending",
            final_result: "pending",
        });
        setDiagnostic(diagnostics.userFound, biometricPanel.dataset.diagInProgress);
        setDiagnostic(diagnostics.cameraOpened, biometricPanel.dataset.diagCompleted, "authenticated");
        setDiagnostic(diagnostics.frameCaptured, biometricPanel.dataset.diagCaptured, "authenticated");
        setDiagnostic(diagnostics.faceDetected, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.eyeDetected, biometricPanel.dataset.diagPending);
        setDiagnostic(diagnostics.bestEyeSelected, biometricPanel.dataset.diagPending);
        setScannerState("processing", biometricPanel.dataset.findingUser);
        showResult(biometricPanel.dataset.findingUser, biometricPanel.dataset.verifying, false);

        const payload = new FormData();
        payload.append("identifier", identifierInput?.value.trim() || "");
        payload.append("preferred_language", document.documentElement.lang || "en");
        payload.append("webcam_image_data", hiddenImageInput.value);
        setCaptureBusy(true);
        redirectScheduled = false;
        const stopProcessingSequence = startProcessingSequence();
        try {
            const response = await fetch(biometricPanel.dataset.endpoint, {
                method: "POST",
                body: payload,
                credentials: "same-origin",
            });
            let result = null;
            try {
                result = await response.json();
            } catch (error) {
                result = {
                    success: false,
                    biometric_verified: false,
                    error_message: biometricPanel.dataset.cameraUnavailable,
                    recommendation_message: "",
                    stages: {
                        frame_received: "failed",
                        face_detection: "failed",
                        eye_detection: "failed",
                        eye_selection: "failed",
                        pupil_segmentation: "failed",
                        iris_segmentation: "failed",
                        feature_extraction: "failed",
                        comparison: "failed",
                        final_result: "failed",
                    },
                };
            }

            stopProcessingSequence();
            setProcessingState(false, biometricPanel.dataset.comparingDatabase);
            paintStages(result.stages || {});
            updateDiagnosticsFromResult(result);
            if (result.stage_images && result.stage_images.final) {
                camera.setPreview(normalizeAppPath(result.stage_images.final));
            } else if (result.annotated_path) {
                camera.setPreview(normalizeAppPath(result.annotated_path));
            }

            if (!response.ok || !result.success || !result.biometric_verified) {
                const detail = result.recommendation_message || (result.threshold ? `${biometricPanel.dataset.thresholdLabel}: ${result.threshold}%` : "");
                const fallbackMessage = result.match_passed === false ? biometricPanel.dataset.matchThresholdFailed : biometricPanel.dataset.matchNotFound;
                setScannerState("error", resolveStepMessage(result, fallbackMessage));
                showResult(result.error_message || fallbackMessage, detail, true);
                return;
            }

            const detail = result.similarity_score
                ? `${result.matched_user} - ${result.similarity_score}% (${biometricPanel.dataset.thresholdLabel}: ${result.threshold}%)`
                : result.matched_user;
            setProcessingState(true, biometricPanel.dataset.finalizingLogin);
            setScannerState("success", result.message || biometricPanel.dataset.success);
            if (faceIdIcon) {
                faceIdIcon.textContent = "✓";
            }
            showResult(result.message || biometricPanel.dataset.success, detail || "", false);
            if (result.redirect_url) {
                redirectScheduled = true;
                window.setTimeout(() => {
                    window.location.href = normalizeAppPath(result.redirect_url);
                }, 500);
            }
        } catch (error) {
            stopProcessingSequence();
            setProcessingState(false, biometricPanel.dataset.comparingDatabase);
            paintStages({
                frame_received: "failed",
                face_detection: "failed",
                eye_detection: "failed",
                eye_selection: "failed",
                pupil_segmentation: "failed",
                iris_segmentation: "failed",
                feature_extraction: "failed",
                comparison: "failed",
                final_result: "failed",
            });
            setDiagnostic(diagnostics.loginDecision, biometricPanel.dataset.diagDenied, "failed");
            setScannerState("error", biometricPanel.dataset.cameraUnavailable);
            showResult(biometricPanel.dataset.cameraUnavailable, "", true);
        } finally {
            if (!redirectScheduled) {
                setProcessingState(false, biometricPanel.dataset.comparingDatabase);
            }
            setCaptureBusy(false);
        }
    });

    document.getElementById("login-open-camera-button")?.addEventListener("click", () => {
        setScannerState("idle", biometricPanel.dataset.cameraReady);
        showResult(biometricPanel.dataset.waiting, biometricPanel.dataset.verifying, false);
    });

    document.getElementById("login-close-camera-button")?.addEventListener("click", () => {
        if (!requestInFlight) {
            setScannerState("idle", biometricPanel.dataset.cameraWaiting || biometricPanel.dataset.waiting);
        }
    });
}

function buildCharts(payload) {
    const sharedGrid = {
        color: "rgba(155, 184, 202, 0.18)",
    };
    const sharedTicks = {
        color: "#9bb8ca",
    };

    const dayChartEl = document.getElementById("dayChart");
    if (dayChartEl) {
        new Chart(dayChartEl, {
            type: "bar",
            data: {
                labels: payload.days,
                datasets: [{
                    label: payload.analysis_dataset_label || "Analyses",
                    data: payload.counts,
                    backgroundColor: "rgba(53, 217, 255, 0.55)",
                    borderColor: "#35d9ff",
                    borderWidth: 1,
                    borderRadius: 10,
                }],
            },
            options: {
                plugins: {
                    legend: { labels: { color: "#e6fbff" } },
                },
                scales: {
                    x: { grid: sharedGrid, ticks: sharedTicks },
                    y: { grid: sharedGrid, ticks: sharedTicks, beginAtZero: true },
                },
            },
        });
    }

    const statusChartEl = document.getElementById("statusChart");
    if (statusChartEl) {
        new Chart(statusChartEl, {
            type: "doughnut",
            data: {
                labels: payload.status_labels || ["Authenticated", "Rejected", "Failed"],
                datasets: [{
                    data: payload.status_values,
                    backgroundColor: ["#19f0c1", "#35d9ff", "#ffb84d"],
                    borderWidth: 0,
                }],
            },
            options: {
                plugins: {
                    legend: { labels: { color: "#e6fbff" } },
                },
            },
        });
    }

    const confidenceChartEl = document.getElementById("confidenceChart");
    if (confidenceChartEl) {
        new Chart(confidenceChartEl, {
            type: "line",
            data: {
                labels: payload.confidence_labels,
                datasets: [{
                    label: payload.confidence_dataset_label || "Confidence",
                    data: payload.confidence_values,
                    borderColor: "#35d9ff",
                    backgroundColor: "rgba(53, 217, 255, 0.16)",
                    fill: true,
                    tension: 0.35,
                }],
            },
            options: {
                plugins: {
                    legend: { labels: { color: "#e6fbff" } },
                },
                scales: {
                    x: { grid: sharedGrid, ticks: sharedTicks },
                    y: {
                        grid: sharedGrid,
                        ticks: sharedTicks,
                        beginAtZero: true,
                        suggestedMax: 100,
                    },
                },
            },
        });
    }
}

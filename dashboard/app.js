/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  Skeleton ID Dashboard — Client-Side JavaScript
 *  Handles: WebSocket streaming, camera, enrollment, API calls, UI updates
 * ═══════════════════════════════════════════════════════════════════════════
 */

const API_BASE = window.location.origin;
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${WS_PROTOCOL}//${window.location.host}/ws/stream`;

// ── State ────────────────────────────────────────────────────────────────────
const state = {
    ws: null,
    cameraStream: null,
    isStreaming: false,
    frameLoopTimer: null,
    isEnrolling: false,
    enrollUserId: null,
    enrollFrameCount: 0,
    fps: 0,
    frameCount: 0,
    lastFpsTime: Date.now(),
    usePhoneCamera: false,
    phoneCameraUrl: "",
};

// ── DOM References ───────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Initialize ───────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initLiveFeed();
    initEnrollment();
    initTraining();
    initReportDownload();
    loadUsers();
    loadStats();
    checkHealth();
});

// ═══════════════════════════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════

function initTabs() {
    $$(".nav-item").forEach((btn) => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.tab;

            // Update nav
            $$(".nav-item").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            // Update panels
            $$(".tab-panel").forEach((p) => p.classList.remove("active"));
            $(`#tab-${tab}`).classList.add("active");

            // Load data for certain tabs
            if (tab === "users") loadUsers();
            if (tab === "stats") loadStats();
            if (tab === "training") loadModelStatus();
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  LIVE FEED
// ═══════════════════════════════════════════════════════════════════════════════

function initLiveFeed() {
    $("#btn-start-camera").addEventListener("click", startCamera);
    $("#btn-stop-camera").addEventListener("click", stopCamera);
}

async function startCamera() {
    try {
        state.cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false,
        });

        const video = $("#webcam-video");
        video.srcObject = state.cameraStream;
        await video.play();

        // Set canvas size
        const canvas = $("#skeleton-canvas");
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        // UI updates
        $("#video-overlay").classList.add("hidden");
        $("#btn-start-camera").classList.add("hidden");
        $("#btn-stop-camera").classList.remove("hidden");

        // Connect WebSocket
        connectWebSocket();

        // Start sending frames
        state.isStreaming = true;
        scheduleFrameLoop();

        toast("Camera started", "success");
    } catch (err) {
        toast(`Camera error: ${err.message}`, "error");
    }
}

function stopCamera() {
    state.isStreaming = false;

    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach((t) => t.stop());
        state.cameraStream = null;
    }

    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }

    if (state.frameLoopTimer) {
        clearTimeout(state.frameLoopTimer);
        state.frameLoopTimer = null;
    }

    const video = $("#webcam-video");
    video.srcObject = null;

    // Clear canvas
    const canvas = $("#skeleton-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // UI
    $("#video-overlay").classList.remove("hidden");
    $("#btn-start-camera").classList.remove("hidden");
    $("#btn-stop-camera").classList.add("hidden");
    $("#id-badge").classList.add("hidden");

    toast("Camera stopped", "info");
}

function connectWebSocket() {
    if (state.ws) state.ws.close();

    state.ws = new WebSocket(WS_URL);

    state.ws.onopen = () => {
        updateStatus(true);
        toast("Connected to server pipeline ✅", "success");

        if (state.isStreaming) {
            scheduleFrameLoop();
        }
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleStreamResult(data);
    };

    state.ws.onclose = () => {
        updateStatus(false);
        toast("Pipeline disconnected ❌", "error");
    };

    state.ws.onerror = (err) => {
        updateStatus(false);
        console.error("WS error", err);
        toast("Connection error — is the server running?", "error");
    };
}

function sendFrameLoop() {
    state.frameLoopTimer = null;

    if (!state.isStreaming || !state.ws) {
        return;
    }

    if (state.ws.readyState !== WebSocket.OPEN) {
        scheduleFrameLoop(100);
        return;
    }

    // Use the correct video element based on current mode
    let source;
    if (state.isEnrolling && state.usePhoneCamera) {
        if (!state.phoneImage || !state.phoneImage.complete) return;
        source = state.phoneImage;
    } else {
        source = state.isEnrolling ? $("#enroll-video") : $("#webcam-video");
    }

    if (!source || (source.tagName === "VIDEO" && source.readyState < 2)) {
        // Source not ready yet, try again shortly
        scheduleFrameLoop(100);
        return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    
    try {
        ctx.drawImage(source, 0, 0, 640, 480);
        
        // Get frame as base64
        const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
        const base64 = dataUrl.split(",")[1];

        const msg = {
            frame: base64,
            mode: state.isEnrolling ? "enroll" : "identify",
            user_id: state.isEnrolling ? state.enrollUserId : null
        };

        state.ws.send(JSON.stringify(msg));

        // If it's a phone camera "shot" URL, we need to refresh the source for the NEXT loop
        if (state.isEnrolling && state.usePhoneCamera && state.phoneImage) {
            // Add cache buster to URL to force fresh frame
            const baseUrl = state.phoneCameraUrl.split("?")[0];
            state.phoneImage.src = `${baseUrl}?t=${Date.now()}`;
        }
    } catch (e) {
        console.warn("Frame capture error (likely CORS):", e);
        // Don't let the loop die, it might recover or work for webcam
    }

    // FPS counter
    state.frameCount++;
    const now = Date.now();
    if (now - state.lastFpsTime >= 1000) {
        state.fps = state.frameCount;
        state.frameCount = 0;
        state.lastFpsTime = now;
        $("#fps-badge").textContent = `${state.fps} FPS`;
    }

    // Next frame (~15 FPS to balance load)
    scheduleFrameLoop(66);
}

function scheduleFrameLoop(delay = 0) {
    if (!state.isStreaming || state.frameLoopTimer) {
        return;
    }

    state.frameLoopTimer = setTimeout(() => {
        state.frameLoopTimer = null;
        sendFrameLoop();
    }, delay);
}

function handleStreamResult(data) {
    if (!data.detected) {
        $("#id-name").textContent = "No person detected";
        $("#id-method").textContent = "—";
        $("#id-avatar").textContent = "?";
        $("#id-badge").classList.add("hidden");
        clearCanvas(state.isEnrolling ? "#enroll-canvas" : "#skeleton-canvas");
        
        // Update overlay status
        if (state.isEnrolling) {
            $("#enroll-overlay").classList.remove("hidden");
            $("#enroll-overlay p").textContent = data.status_msg || "No person detected";
        }
        return;
    }

    // Draw skeleton on the correct canvas (enrollment or live feed)
    if (data.keypoints) {
        if (state.isEnrolling) {
            drawSkeleton(data.keypoints, "#enroll-canvas");
        } else {
            drawSkeleton(data.keypoints, "#skeleton-canvas");
        }
    }

    // Display status message in overlay
    if (state.isEnrolling) {
        if (data.status_msg) {
            $("#enroll-overlay").classList.remove("hidden");
            $("#enroll-overlay p").textContent = data.status_msg;
        }
        
        // Hide overlay if everything is okay
        if (data.features_ok) {
            $("#enroll-overlay").classList.add("hidden");
        }
    }

    // Handle enrollment sample collection
    if (state.isEnrolling && data.mode === "enroll" && data.features_ok) {
        enrollFrame(data);
    }

    // Update pipeline stats
    $("#stat-latency").textContent = `${data.latency_ms} ms`;
    $("#stat-features").textContent = data.num_features || "--";
    $("#stat-gait").textContent = `${data.gait_buffer || 0} / 30`;

    // Exit early for ID processing if features aren't perfect
    if (!data.features_ok) return;

    const id = data.identification || {};
    const user = id.user || "unknown";
    const conf = id.confidence || 0;
    const isKnown = id.is_known || false;
    const method = id.method || "none";

    // Update identification display
    $("#id-name").textContent = isKnown ? user : "Unknown Person";
    $("#id-method").textContent = method === "none" ? "No model loaded" : `Method: ${method}`;
    $("#id-avatar").textContent = isKnown ? user.charAt(0).toUpperCase() : "?";
    $("#stat-method").textContent = method;

    // Confidence bar
    const confPct = Math.round(conf * 100);
    const bar = $("#confidence-bar");
    bar.style.width = `${confPct}%`;
    bar.className = "confidence-bar";
    if (confPct >= 75) bar.classList.add("high");
    else if (confPct >= 50) bar.classList.add("low");
    else bar.classList.add("very-low");
    $("#confidence-label").textContent = `Confidence: ${confPct}%`;

    // ID badge on video
    const badge = $("#id-badge");
    if (isKnown) {
        badge.classList.remove("hidden");
        $("#id-badge-name").textContent = user;
        $("#id-badge-conf").textContent = `${confPct}%`;
    } else {
        badge.classList.add("hidden");
    }

    // Top candidates: show only the current identified person
    const list = $("#candidates-list");
    const displayName = isKnown ? user : "Unknown Person";
    list.innerHTML = `
        <div class="candidate-row">
            <span class="candidate-name">${displayName}</span>
            <span class="candidate-score">${confPct}%</span>
        </div>
    `;

    // Handle enrollment
    if (state.isEnrolling && data.mode === "enroll" && data.features_ok) {
        enrollFrame(data);
    }

    // Display status message in overlay
    if (state.isEnrolling && data.status_msg) {
        $("#enroll-overlay").classList.remove("hidden");
        $("#enroll-overlay p").textContent = data.status_msg;
    } else if (state.isEnrolling && data.features_ok) {
        $("#enroll-overlay").classList.add("hidden");
    }
}

// ── Skeleton Drawing ─────────────────────────────────────────────────────────

const SKELETON_CONNECTIONS = [
    [11, 13], [13, 15], [12, 14], [14, 16], // Arms
    [11, 12], [23, 24],                       // Shoulders, Hips
    [11, 23], [12, 24],                       // Torso
    [23, 25], [25, 27], [24, 26], [26, 28],  // Legs
    [0, 11], [0, 12],                         // Head to shoulders
];

function drawSkeleton(keypoints, canvasSelector = "#skeleton-canvas") {
    const canvas = $(canvasSelector);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw connections
    ctx.strokeStyle = "rgba(0, 200, 255, 0.7)";
    ctx.lineWidth = 2;
    SKELETON_CONNECTIONS.forEach(([i, j]) => {
        const a = keypoints[i];
        const b = keypoints[j];
        if (a && b && a.visibility > 0.2 && b.visibility > 0.2) {
            ctx.beginPath();
            ctx.moveTo(a.x * w, a.y * h);
            ctx.lineTo(b.x * w, b.y * h);
            ctx.stroke();
        }
    });

    // Draw keypoints
    keypoints.forEach((kp, idx) => {
        if (kp.visibility > 0.2) {
            ctx.beginPath();
            ctx.arc(kp.x * w, kp.y * h, 4, 0, Math.PI * 2);
            ctx.fillStyle =
                idx >= 11 ? "rgba(99, 255, 132, 0.9)" : "rgba(99, 200, 255, 0.7)";
            ctx.fill();
        }
    });
}

function clearCanvas(canvasSelector = "#skeleton-canvas") {
    const canvas = $(canvasSelector);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ENROLLMENT
// ═══════════════════════════════════════════════════════════════════════════════

function initEnrollment() {
    $("#btn-create-user").addEventListener("click", createUser);
    $("#btn-start-enrollment").addEventListener("click", startEnrollment);
    $("#btn-stop-enrollment").addEventListener("click", stopEnrollment);
    $("#enroll-select").addEventListener("change", (e) => {
        $("#btn-start-enrollment").disabled = !e.target.value;
    });

    // Camera source toggling (Webcam vs Phone)
    const btnWebcam = $("#btn-cam-webcam");
    const btnPhone = $("#btn-cam-phone");
    const phoneConfig = $("#phone-camera-config");

    if (btnWebcam && btnPhone) {
        btnWebcam.addEventListener("click", () => {
            state.usePhoneCamera = false;
            btnWebcam.classList.add("active");
            btnWebcam.classList.remove("btn-outline");
            btnWebcam.classList.add("btn-secondary");
            btnPhone.classList.remove("active");
            btnPhone.classList.add("btn-outline");
            btnPhone.classList.remove("btn-secondary");
            phoneConfig.style.display = "none";
        });

        btnPhone.addEventListener("click", () => {
            state.usePhoneCamera = true;
            btnPhone.classList.add("active");
            btnPhone.classList.remove("btn-outline");
            btnPhone.classList.add("btn-secondary");
            btnWebcam.classList.remove("active");
            btnWebcam.classList.add("btn-outline");
            btnWebcam.classList.remove("btn-secondary");
            phoneConfig.style.display = "block";
        });
    }
}

async function createUser() {
    const name = $("#enroll-name").value.trim();
    const email = $("#enroll-email").value.trim();

    if (!name) {
        toast("Please enter a name", "error");
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/users/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, email: email || null }),
        });

        if (!res.ok) {
            const err = await res.json();
            toast(err.detail || "Failed to create user", "error");
            return;
        }

        const user = await res.json();
        toast(`User "${user.name}" created!`, "success");

        // Clear form
        $("#enroll-name").value = "";
        $("#enroll-email").value = "";

        // Refresh user list in dropdown
        await loadEnrollDropdown();
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    }
}

async function loadEnrollDropdown() {
    try {
        const res = await fetch(`${API_BASE}/api/users/`);
        const users = await res.json();

        const select = $("#enroll-select");
        select.innerHTML = '<option value="">-- Select User --</option>';
        users.forEach((u) => {
            const opt = document.createElement("option");
            opt.value = u.user_id;
            opt.textContent = `${u.name} (${u.enrollment_status})`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to load users for dropdown", err);
    }
}

async function startEnrollment() {
    const userId = $("#enroll-select").value;
    if (!userId) return;

    state.isEnrolling = true;
    state.enrollUserId = userId;
    state.enrollFrameCount = 0;

    const enrollVideo = $("#enroll-video");
    const enrollCanvas = $("#enroll-canvas");
    
    // Stop any existing camera stream first
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach((t) => t.stop());
        state.cameraStream = null;
    }

    try {
        if (state.usePhoneCamera) {
            // PHONE CAMERA MODE
            const url = $("#phone-camera-url").value;
            if (!url) {
                toast("Please enter your Phone Camera URL", "error");
                $("#enroll-overlay").classList.remove("hidden");
                $("#enroll-overlay p").textContent = "Enter your phone camera URL before starting enrollment";
                state.isEnrolling = false;
                return;
            }
            state.phoneCameraUrl = url;
            
            // Show a visual indicator that we're connecting
            $("#enroll-overlay p").textContent = "Connecting to phone camera...";
            
            // Create an image object to poll the IP camera (works better for MJPEG than <video>)
            state.phoneImage = new Image();
            state.phoneImage.crossOrigin = "anonymous";
            state.phoneImage.src = url;
            
            // Start the send loop once the WebSocket connects
        } else {
            // WEBCAM MODE
            state.cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" },
                audio: false,
            });

            enrollVideo.srcObject = state.cameraStream;
            await enrollVideo.play();
        }

        // Set the enrollment canvas size
        enrollCanvas.width = 640;
        enrollCanvas.height = 480;

        // Hide the overlay text
        $("#enroll-overlay").classList.add("hidden");

        // Connect WebSocket for processing
        connectWebSocket();
        state.isStreaming = true;
        scheduleFrameLoop();
        
        toast(`Enrollment started using ${state.usePhoneCamera ? "Phone" : "Webcam"}`, "info");
    } catch (err) {
        toast(`Camera error: ${err.message}`, "error");
        state.isEnrolling = false;
        $("#enroll-overlay").classList.remove("hidden");
        $("#enroll-overlay p").textContent = "Select a user and start enrollment";
        return;
    }

    // UI
    $("#btn-start-enrollment").classList.add("hidden");
    $("#btn-stop-enrollment").classList.remove("hidden");
}

function stopEnrollment() {
    state.isEnrolling = false;
    state.enrollUserId = null;
    state.isStreaming = false;
    if (state.frameLoopTimer) {
        clearTimeout(state.frameLoopTimer);
        state.frameLoopTimer = null;
    }

    // Stop the camera stream
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach((t) => t.stop());
        state.cameraStream = null;
    }

    // Close WebSocket
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }

    // Reset enrollment video
    const enrollVideo = $("#enroll-video");
    enrollVideo.srcObject = null;
    $("#enroll-overlay").classList.remove("hidden");

    // Clear the enrollment canvas
    clearCanvas("#enroll-canvas");

    // Reset UI
    $("#btn-start-enrollment").classList.remove("hidden");
    $("#btn-stop-enrollment").classList.add("hidden");

    toast(
        `Enrollment stopped. ${state.enrollFrameCount} frames collected.`,
        "success"
    );
}

async function enrollFrame(data) {
    if (!state.enrollUserId) return;

    // Throttle: don't send faster than every 250ms and don't send while previous is pending
    if (state._enrollPending) return;
    const now = Date.now();
    if (state._lastEnrollTime && (now - state._lastEnrollTime) < 250) return;

    // Use REAL features from the WebSocket pipeline (not placeholder zeros!)
    const staticFeatures = data.static_features;
    if (!staticFeatures || staticFeatures.length === 0) {
        return; // No features extracted this frame, skip
    }

    state._enrollPending = true;
    state._lastEnrollTime = now;

    try {
        const res = await fetch(`${API_BASE}/api/enroll/frame`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: state.enrollUserId,
                static_features: staticFeatures,
                gait_features: null,
            }),
        });

        if (res.ok) {
            const result = await res.json();
            state.enrollFrameCount = result.frames_collected;

            // Update progress — use server-returned progress (based on min_enrollment_frames config)
            const pct = Math.min(result.progress || 0, 100);
            $("#enroll-progress-bar").style.width = `${pct}%`;
            $("#enroll-progress-text").textContent = `${result.frames_collected} frames (${Math.round(pct)}%)`;

            if (result.status === "completed") {
                toast("Enrollment complete! ✅", "success");
                stopEnrollment();
            }
        }
    } catch (err) {
        // Silent — don't spam errors during enrollment
    } finally {
        state._enrollPending = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  USERS
// ═══════════════════════════════════════════════════════════════════════════════

async function loadUsers() {
    try {
        const res = await fetch(`${API_BASE}/api/users/`);
        const users = await res.json();

        const grid = $("#users-grid");

        if (users.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                        <circle cx="9" cy="7" r="4"/>
                    </svg>
                    <p>No users enrolled yet. Go to Enroll tab to add users.</p>
                </div>`;
            return;
        }

        grid.innerHTML = users
            .map(
                (u) => `
            <div class="user-card glass-card">
                <div class="user-card-header">
                    <div class="user-avatar">${u.name.charAt(0).toUpperCase()}</div>
                    <div>
                        <div class="user-name">${u.name}</div>
                        <div class="user-id">${u.user_id.substring(0, 8)}...</div>
                    </div>
                </div>
                <span class="user-status-badge ${u.enrollment_status}">${u.enrollment_status}</span>
                <div class="user-card-meta">
                    <span class="user-frames">${u.enrollment_frames_count} frames</span>
                    <button class="btn-delete-user" onclick="deleteUser('${u.user_id}')">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            </div>`
            )
            .join("");

        // Also update enrollment dropdown
        await loadEnrollDropdown();
    } catch (err) {
        toast(`Failed to load users: ${err.message}`, "error");
    }
}

async function deleteUser(userId) {
    if (!confirm("Delete this user and all their data?")) return;

    try {
        const res = await fetch(`${API_BASE}/api/users/${userId}`, {
            method: "DELETE",
        });
        if (res.ok) {
            toast("User deleted", "success");
            loadUsers();
        } else {
            toast("Failed to delete user", "error");
        }
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    }
}

// Make deleteUser accessible from inline onclick
window.deleteUser = deleteUser;

// ═══════════════════════════════════════════════════════════════════════════════
//  TRAINING
// ═══════════════════════════════════════════════════════════════════════════════

function initTraining() {
    $("#btn-train").addEventListener("click", trainModel);
    loadModelStatus();
}

async function trainModel() {
    const type = $("#train-type").value;
    const epochs = parseInt($("#train-epochs").value) || 100;

    $("#training-status").classList.remove("hidden");
    $("#btn-train").disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_type: type,
                epochs,
                batch_size: 32,
            }),
        });

        const result = await res.json();

        if (result.success) {
            toast("Training complete! ✅", "success");

            // Show results
            const card = $("#training-results-card");
            card.style.display = "block";
            $("#training-results").textContent = JSON.stringify(result, null, 2);
        } else {
            toast(`Training failed: ${result.detail || "Unknown error"}`, "error");
        }
    } catch (err) {
        toast(`Training error: ${err.message}`, "error");
    } finally {
        $("#training-status").classList.add("hidden");
        $("#btn-train").disabled = false;
        loadModelStatus();
    }
}

async function loadModelStatus() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const health = await res.json();

        const models = health.models || {};

        // SVM
        if (models.svm) {
            $("#svm-state").textContent = "Trained ✅";
            $("#svm-state").style.color = "var(--success)";
        } else {
            $("#svm-state").textContent = "Not Trained";
            $("#svm-state").style.color = "var(--text-muted)";
        }

        // LSTM
        if (models.lstm) {
            $("#lstm-state").textContent = "Trained ✅";
            $("#lstm-state").style.color = "var(--success)";
        } else {
            $("#lstm-state").textContent = "Not Trained";
            $("#lstm-state").style.color = "var(--text-muted)";
        }
    } catch (err) {
        // Server might not be running
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/api/stats`);
        const data = await res.json();

        $("#stat-total-users").textContent = data.total_users || 0;

        const stats = data.identification_stats || {};
        $("#stat-total-ids").textContent = stats.total_identifications || 0;
        $("#stat-avg-conf").textContent = `${Math.round(
            (stats.avg_confidence || 0) * 100
        )}%`;
        $("#stat-avg-latency").textContent = `${Math.round(
            stats.avg_latency_ms || 0
        )}ms`;

        // Recent identifications
        const recent = data.recent_identifications || [];
        const tbody = $("#recent-tbody");

        if (recent.length === 0) {
            tbody.innerHTML =
                '<tr><td colspan="5" class="text-center text-muted">No data yet</td></tr>';
        } else {
            tbody.innerHTML = recent
                .map(
                    (r) => `
                <tr>
                    <td>${new Date(r.timestamp).toLocaleTimeString()}</td>
                    <td>${r.predicted_user_id || "Unknown"}</td>
                    <td>${Math.round((r.confidence || 0) * 100)}%</td>
                    <td>${Math.round(r.latency_ms || 0)}ms</td>
                    <td>${r.model_version || "—"}</td>
                </tr>`
                )
                .join("");
        }
    } catch (err) {
        // Stats endpoint may not be available
    }
}

// Refresh button
document.addEventListener("DOMContentLoaded", () => {
    const btnRefreshUsers = $("#btn-refresh-users");
    if (btnRefreshUsers) btnRefreshUsers.addEventListener("click", loadUsers);

    const btnRefreshStats = $("#btn-refresh-stats");
    if (btnRefreshStats) btnRefreshStats.addEventListener("click", loadStats);
});

function initReportDownload() {
    const btnDownloadReport = $("#btn-download-report");
    if (btnDownloadReport) {
        btnDownloadReport.addEventListener("click", downloadReportPdf);
    }
}

async function downloadReportPdf() {
    const button = $("#btn-download-report");
    if (button) button.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/report/pdf`);
        if (!res.ok) {
            throw new Error(`Report request failed (${res.status})`);
        }

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `identification-report-${new Date().toISOString().slice(0, 10)}.pdf`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        toast("PDF report downloaded", "success");
    } catch (err) {
        toast(`Failed to download report: ${err.message}`, "error");
    } finally {
        if (button) button.disabled = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  HEALTH CHECK & UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        updateStatus(data.status === "healthy");
    } catch {
        updateStatus(false);
    }
}

function updateStatus(online) {
    const status = $("#system-status");
    const dot = status.querySelector(".status-dot");
    const text = status.querySelector("span");

    if (online) {
        dot.className = "status-dot online";
        text.textContent = "Connected";
    } else {
        dot.className = "status-dot offline";
        text.textContent = "Disconnected";
    }
}

function toast(message, type = "info") {
    const container = $("#toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

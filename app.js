const VISION_VERSION = "0.10.21";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/wasm`;
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const EYE_CLOSED_THRESHOLD = 0.20;
const EYE_CLOSED_DURATION_MS = 1500;
const ALERT_COOLDOWN_MS = 8000;
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const EYE_CONTOURS = [LEFT_EYE, RIGHT_EYE];

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const context = canvas.getContext("2d");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const cameraMessage = document.querySelector("#camera-message");
const errorMessage = document.querySelector("#error-message");
let stream = null;
let landmarker = null;
let faceMeshConnections = [];
let faceOvalConnections = [];
let faceLandmarkerPromise = null;
let running = false;
let animationFrame = 0;
let lastInferenceAt = 0;
let closedEyesSince = null;
let lastAlertAt = 0;
let lastVideoTime = -1;
let voiceReady = "speechSynthesis" in window;

function updateVoiceStatus(text) {
  document.querySelector("#voice-status").textContent = text;
}

function announce(message) {
  if (!voiceReady) {
    updateVoiceStatus("Speech unavailable in this browser");
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.92;
  utterance.onstart = () => updateVoiceStatus("Speaking alert…");
  utterance.onend = () => updateVoiceStatus("Ready to speak");
  utterance.onerror = () => updateVoiceStatus("Voice alert could not play");
  window.speechSynthesis.speak(utterance);
}

async function loadFaceLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await import(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/vision_bundle.mjs`
  );
  const files = await FilesetResolver.forVisionTasks(WASM_ROOT);
  const options = {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  };
  let model;
  try {
    model = await FaceLandmarker.createFromOptions(files, options);
  } catch {
    model = await FaceLandmarker.createFromOptions(files, {
      ...options,
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "CPU" },
    });
  }
  faceMeshConnections = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
  faceOvalConnections = FaceLandmarker.FACE_LANDMARKS_FACE_OVAL;
  return model;
}

function setAttention(level, title, description, amount) {
  const panel = document.querySelector("#attention-panel");
  panel.className = `attention ${level}`;
  document.querySelector("#attention-title").textContent = title;
  document.querySelector("#attention-copy").textContent = description;
  document.querySelector("#attention-meter").style.width = `${amount}%`;
}

function setEye(value, state, stateColor = "") {
  document.querySelector("#eye-value").textContent = value;
  const label = document.querySelector("#eye-state");
  label.textContent = state;
  label.style.color = stateColor;
}

function pointDistance(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function eyeAspectRatio(landmarks, indices) {
  const points = indices.map((index) => landmarks[index]);
  const horizontal = pointDistance(points[0], points[3]);
  if (!horizontal) return 1;
  return (pointDistance(points[1], points[5]) + pointDistance(points[2], points[4]))
    / (2 * horizontal);
}

function drawConnections(landmarks, connections) {
  context.beginPath();
  for (const { start, end } of connections) {
    context.moveTo(landmarks[start].x * canvas.width, landmarks[start].y * canvas.height);
    context.lineTo(landmarks[end].x * canvas.width, landmarks[end].y * canvas.height);
  }
  context.stroke();
}

function drawFace(landmarks, now) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  const pulse = 0.5 + Math.sin(now / 650) * 0.5;
  const faceGradient = context.createLinearGradient(
    canvas.width * 0.22,
    canvas.height * 0.12,
    canvas.width * 0.78,
    canvas.height * 0.92,
  );
  faceGradient.addColorStop(0, "#43f5e8");
  faceGradient.addColorStop(0.48, "#83a8ff");
  faceGradient.addColorStop(1, "#d78bff");

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.globalAlpha = 0.2 + pulse * 0.12;
  context.strokeStyle = faceGradient;
  context.lineWidth = Math.max(0.7, canvas.width / 1150);
  drawConnections(landmarks, faceMeshConnections);

  context.globalAlpha = 0.34 + pulse * 0.12;
  context.strokeStyle = "#63f3ec";
  context.lineWidth = Math.max(4, canvas.width / 190);
  context.shadowBlur = 18 + pulse * 8;
  context.shadowColor = "#56e8ff";
  drawConnections(landmarks, faceOvalConnections);

  context.globalAlpha = 0.92;
  context.strokeStyle = faceGradient;
  context.lineWidth = Math.max(1.8, canvas.width / 480);
  context.shadowBlur = 5 + pulse * 5;
  context.shadowColor = "#77dfff";
  drawConnections(landmarks, faceOvalConnections);

  context.shadowBlur = 8;
  context.shadowColor = "#d58bff";
  context.strokeStyle = "#c49bff";
  context.fillStyle = "rgba(75, 221, 255, 0.10)";
  context.lineWidth = Math.max(1.5, canvas.width / 600);
  for (const contour of EYE_CONTOURS) {
    context.beginPath();
    contour.forEach((index, offset) => {
      const point = landmarks[index];
      const x = point.x * canvas.width;
      const y = point.y * canvas.height;
      if (offset === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.closePath();
    context.fill();
    context.stroke();
  }

  context.shadowBlur = 0;
  const accentRadius = Math.max(1.5, canvas.width / 430);
  context.globalAlpha = 0.62 + pulse * 0.3;
  for (const index of [10, 1, 33, 263, 61, 291, 152]) {
    const point = landmarks[index];
    context.beginPath();
    context.arc(
      point.x * canvas.width,
      point.y * canvas.height,
      accentRadius * (0.85 + pulse * 0.2),
      0,
      Math.PI * 2,
    );
    context.fillStyle = index === 1 ? "#ffffff" : faceGradient;
    context.fill();
  }
  context.restore();
}

function updateFaceStatus(landmarks, now) {
  const leftClosed = eyeAspectRatio(landmarks, LEFT_EYE) < EYE_CLOSED_THRESHOLD;
  const rightClosed = eyeAspectRatio(landmarks, RIGHT_EYE) < EYE_CLOSED_THRESHOLD;
  const eyesClosed = leftClosed && rightClosed;
  document.querySelector("#face-value").textContent = "Tracking";
  document.querySelector("#tracking-state").textContent = "ACTIVE";
  document.querySelector("#tracking-state").style.color = "var(--lime)";
  document.querySelector("#face-badge").classList.add("active");
  cameraMessage.textContent = "Face tracked on-device. Look ahead and drive safely.";

  if (!eyesClosed) {
    closedEyesSince = null;
    setEye("Open", "FOCUSED", "var(--lime)");
    setAttention("focused", "Looking good", "Your eyes are open. Keep your attention on the road.", 100);
    return;
  }

  closedEyesSince ??= now;
  const closedDuration = now - closedEyesSince;
  if (closedDuration >= EYE_CLOSED_DURATION_MS) {
    setEye("Closed — alert", "CHECK IN", "var(--amber)");
    setAttention("warning", "Eyes closed", "Please open your eyes and focus on the road.", 88);
    if (now - lastAlertAt >= ALERT_COOLDOWN_MS) {
      announce("Please open your eyes and focus on the road.");
      lastAlertAt = now;
    }
  } else {
    setEye("Closed", "DETECTED", "var(--amber)");
    setAttention("warning", "Stay focused", "Eyes closed. Alerting if they stay closed.", 50);
  }
}

function drawNoFace() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  closedEyesSince = null;
  document.querySelector("#face-value").textContent = "Not detected";
  document.querySelector("#tracking-state").textContent = "SEARCHING";
  document.querySelector("#tracking-state").style.color = "";
  document.querySelector("#face-badge").classList.remove("active");
  cameraMessage.textContent = "Move your face into the camera frame.";
  setEye("Waiting for face", "WAITING");
  setAttention("missing", "Face not in view", "Move into the camera frame to resume monitoring.", 12);
}

function syncCanvasSize() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return false;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return true;
}

function cameraLoop(now) {
  if (!running) return;
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && syncCanvasSize()) {
    if (video.currentTime !== lastVideoTime && now - lastInferenceAt >= 80) {
      lastVideoTime = video.currentTime;
      lastInferenceAt = now;
      try {
        const result = landmarker.detectForVideo(video, now);
        const landmarks = result.faceLandmarks?.[0];
        if (landmarks) {
          drawFace(landmarks, now);
          updateFaceStatus(landmarks, now);
        } else {
          drawNoFace();
        }
      } catch (error) {
        stopMonitoring();
        showError(`Face tracking stopped: ${error.message || error}`);
      }
    }
  }
  animationFrame = requestAnimationFrame(cameraLoop);
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

async function startMonitoring() {
  errorMessage.hidden = true;
  startButton.disabled = true;
  startButton.textContent = "Loading face tracking…";
  cameraMessage.textContent = "Requesting camera access and loading on-device face tracking…";
  try {
    faceLandmarkerPromise ??= loadFaceLandmarker();
    const model = await faceLandmarkerPromise;
    const cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    landmarker = model;
    stream = cameraStream;
    video.srcObject = stream;
    await video.play();
    running = true;
    closedEyesSince = null;
    lastVideoTime = -1;
    document.querySelector("#camera-placeholder").hidden = true;
    document.querySelector("#live-state").classList.add("active");
    document.querySelector("#live-state").innerHTML = "<i></i> LIVE";
    document.querySelector("#session-status").classList.add("active");
    document.querySelector("#session-status span").textContent = "MONITORING";
    startButton.hidden = true;
    stopButton.hidden = false;
    document.querySelector("#face-value").textContent = "Searching";
    document.querySelector("#tracking-state").textContent = "SEARCHING";
    setAttention("missing", "Searching for face", "Move into the camera frame to begin.", 10);
    animationFrame = requestAnimationFrame(cameraLoop);
  } catch (error) {
    stopMonitoring();
    showError(`Could not start monitoring: ${error.message || error}. Check camera permission and internet access, then try again.`);
    cameraMessage.textContent = "Check camera access and try again.";
  }
}

function stopMonitoring() {
  running = false;
  cancelAnimationFrame(animationFrame);
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  context.clearRect(0, 0, canvas.width, canvas.height);
  canvas.width = 0;
  canvas.height = 0;
  document.querySelector("#camera-placeholder").hidden = false;
  document.querySelector("#live-state").classList.remove("active");
  document.querySelector("#live-state").innerHTML = "<i></i> OFFLINE";
  document.querySelector("#session-status").classList.remove("active");
  document.querySelector("#session-status span").textContent = "SYSTEM STANDBY";
  document.querySelector("#face-badge").classList.remove("active");
  document.querySelector("#face-value").textContent = "Inactive";
  document.querySelector("#tracking-state").textContent = "OFF";
  setEye("—", "WAITING");
  setAttention("", "Ready when you are", "Start monitoring to see your live status.", 0);
  cameraMessage.textContent = "Position your face in the frame to begin.";
  startButton.hidden = false;
  startButton.disabled = false;
  startButton.textContent = "▶  Start monitoring  ↗";
  stopButton.hidden = true;
  closedEyesSince = null;
}

startButton.addEventListener("click", startMonitoring);
stopButton.addEventListener("click", stopMonitoring);
document.querySelector("#test-button").addEventListener("click", () => {
  announce("This is a Roadwatch voice alert test.");
});
window.addEventListener("pagehide", () => {
  stopMonitoring();
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
});

if (!navigator.mediaDevices?.getUserMedia) {
  showError("Camera access requires a secure HTTPS page or localhost in a modern browser.");
}

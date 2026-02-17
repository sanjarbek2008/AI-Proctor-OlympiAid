import os
import logging
import io
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import scipy.io.wavfile as wav
import librosa
import torch
import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Model Initialization
# ============================================================

# 1. Initialize Silero VAD
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    logger.info("Silero VAD model loaded successfully.")
except Exception as e:
    logger.error(f"Silero VAD init failed: {e}")
    vad_model = None

# 2. Initialize YOLO
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    logger.error(f"YOLO init failed: {e}")
    model = None

# 3. Initialize MediaPipe Face Landmarker
try:
    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=2)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    logger.error(f"MediaPipe init failed: {e}")
    face_landmarker = None

# ============================================================
# Configuration (all configurable via env vars)
# ============================================================

# Suspicious YOLO objects
default_objects = "cell phone,remote,laptop,tablet"
sus_objects_env = os.environ.get("SUSPICIOUS_OBJECTS", default_objects)
SUSPICIOUS_OBJECTS = [obj.strip() for obj in sus_objects_env.split(",")]

YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.5"))

# Temporal smoothing config
FRAME_BUFFER_SIZE = int(os.environ.get("FRAME_BUFFER_SIZE", "30"))       # ~2 sec at 15fps
SUSPICION_TRIGGER_RATIO = float(os.environ.get("SUSPICION_TRIGGER_RATIO", "0.50"))  # 50% of frames
SESSION_TIMEOUT_SECONDS = int(os.environ.get("SESSION_TIMEOUT_SECONDS", "300"))     # 5 min cleanup

# Suspicion score thresholds
SUSPICION_SCORE_THRESHOLD = float(os.environ.get("SUSPICION_SCORE_THRESHOLD", "0.40"))

# Gaze thresholds
GAZE_LEFT_THRESHOLD = float(os.environ.get("GAZE_LEFT_THRESHOLD", "0.35"))
GAZE_RIGHT_THRESHOLD = float(os.environ.get("GAZE_RIGHT_THRESHOLD", "0.65"))

# Head pose thresholds
YAW_THRESHOLD = float(os.environ.get("YAW_THRESHOLD", "0.25"))
PITCH_UP_THRESHOLD = float(os.environ.get("PITCH_UP_THRESHOLD", "0.18"))

# ============================================================
# MediaPipe FaceMesh Landmark Indices
# ============================================================

# Iris centers
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Eye corners (for gaze ratio)
# Left eye: outer=33, inner=133
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
# Right eye: inner=362, outer=263
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Improved yaw landmarks (cheeks + eyes instead of ears)
LEFT_CHEEK = 93
RIGHT_CHEEK = 323
LEFT_EYE_CENTER_TOP = 159
LEFT_EYE_CENTER_BOTTOM = 145
RIGHT_EYE_CENTER_TOP = 386
RIGHT_EYE_CENTER_BOTTOM = 374

NOSE_TIP = 1

# Kept for backward compatibility / fallback
LEFT_EAR = 234
RIGHT_EAR = 454


# ============================================================
# Session Tracker (Temporal Smoothing)
# ============================================================

@dataclass
class FrameRecord:
    """A single frame's analysis results."""
    timestamp: float
    suspicion_score: float
    signals: Dict[str, float] = field(default_factory=dict)


class SessionTracker:
    """Maintains per-session frame history for temporal smoothing.
    
    Only fires violation when suspicious frames exceed threshold
    over a rolling window, reducing false positives from blinks,
    micro-saccades, and natural movement.
    """

    def __init__(self, buffer_size: int = FRAME_BUFFER_SIZE,
                 trigger_ratio: float = SUSPICION_TRIGGER_RATIO,
                 score_threshold: float = SUSPICION_SCORE_THRESHOLD):
        self.buffer_size = buffer_size
        self.trigger_ratio = trigger_ratio
        self.score_threshold = score_threshold
        self.frames: deque = deque(maxlen=buffer_size)
        self.last_active: float = time.time()

    def add_frame(self, score: float, signals: Dict[str, float]) -> Optional[str]:
        """Add a frame record and check if violation threshold is reached.
        
        Returns violation reason string if threshold exceeded, else None.
        """
        self.last_active = time.time()
        self.frames.append(FrameRecord(
            timestamp=time.time(),
            suspicion_score=score,
            signals=signals,
        ))

        # Need at least half the buffer before making decisions
        if len(self.frames) < self.buffer_size // 2:
            return None

        # Count frames exceeding suspicion threshold
        suspicious_count = sum(
            1 for f in self.frames if f.suspicion_score >= self.score_threshold
        )
        ratio = suspicious_count / len(self.frames)

        if ratio >= self.trigger_ratio:
            # Determine the dominant violation type from recent signals
            violation = self._determine_violation()
            # Clear buffer after firing to avoid repeated triggers
            self.frames.clear()
            return violation

        return None

    def _determine_violation(self) -> str:
        """Determine the most prominent violation from recent frames."""
        signal_counts: Dict[str, int] = {}
        for frame in self.frames:
            if frame.suspicion_score >= self.score_threshold:
                for signal_name, value in frame.signals.items():
                    if value > 0:
                        signal_counts[signal_name] = signal_counts.get(signal_name, 0) + 1

        if not signal_counts:
            return "sustained_suspicious_behavior"

        # Return the most frequent signal
        dominant = max(signal_counts, key=signal_counts.get)
        return dominant

    def is_expired(self, timeout: int = SESSION_TIMEOUT_SECONDS) -> bool:
        """Check if this session has been inactive too long."""
        return (time.time() - self.last_active) > timeout


# Module-level session registry
_sessions: Dict[str, SessionTracker] = {}


def _get_tracker(session_id: str) -> SessionTracker:
    """Get or create a SessionTracker for the given session."""
    # Periodic cleanup of expired sessions
    expired = [sid for sid, t in _sessions.items() if t.is_expired()]
    for sid in expired:
        del _sessions[sid]
        logger.info(f"Cleaned up expired session tracker: {sid}")

    if session_id not in _sessions:
        _sessions[session_id] = SessionTracker()
    return _sessions[session_id]


# ============================================================
# Eye Gaze Estimation
# ============================================================

def _compute_gaze_ratio(landmarks, img_w: int, img_h: int) -> Tuple[float, float]:
    """Compute horizontal gaze ratio for both eyes using iris + corner landmarks.
    
    Returns (left_eye_ratio, right_eye_ratio) each in [0, 1].
      < 0.35 → looking left
      > 0.65 → looking right
      0.35–0.65 → centered
      
    The ratio is scale-invariant: iris_x position relative to eye width,
    so it works regardless of camera distance or resolution.
    """
    def get_x(idx):
        return landmarks[idx].x * img_w

    # LEFT EYE: outer corner (33) to inner corner (133)
    left_outer_x = get_x(LEFT_EYE_OUTER)
    left_inner_x = get_x(LEFT_EYE_INNER)
    left_iris_x = get_x(LEFT_IRIS_CENTER)

    left_eye_width = abs(left_inner_x - left_outer_x)
    if left_eye_width > 0:
        left_ratio = (left_iris_x - left_outer_x) / left_eye_width
    else:
        left_ratio = 0.5  # default centered

    # RIGHT EYE: inner corner (362) to outer corner (263)
    right_inner_x = get_x(RIGHT_EYE_INNER)
    right_outer_x = get_x(RIGHT_EYE_OUTER)
    right_iris_x = get_x(RIGHT_IRIS_CENTER)

    right_eye_width = abs(right_outer_x - right_inner_x)
    if right_eye_width > 0:
        right_ratio = (right_iris_x - right_inner_x) / right_eye_width
    else:
        right_ratio = 0.5  # default centered

    return (left_ratio, right_ratio)


# ============================================================
# Head Pose Estimation (Improved)
# ============================================================

def _compute_head_pose(landmarks, img_w: int, img_h: int) -> Dict[str, float]:
    """Compute head yaw and pitch using cheek + eye landmarks (more stable than ears).
    
    Returns dict with:
      'yaw_offset': normalized nose offset from face center (0 = centered)
      'pitch_offset': normalized vertical offset (positive = looking up)
      'face_width': the computed face width in pixels
    """
    def get_point(idx):
        lm = landmarks[idx]
        return (lm.x * img_w, lm.y * img_h)

    nose = get_point(NOSE_TIP)

    # Use cheek landmarks for face width (more stable than ears)
    left_cheek = get_point(LEFT_CHEEK)
    right_cheek = get_point(RIGHT_CHEEK)
    face_width = abs(right_cheek[0] - left_cheek[0])

    if face_width <= 0:
        return {'yaw_offset': 0.0, 'pitch_offset': 0.0, 'face_width': 0.0}

    # Yaw: nose offset from midpoint of cheeks
    face_center_x = (left_cheek[0] + right_cheek[0]) / 2
    yaw_offset = abs(nose[0] - face_center_x) / face_width

    # Pitch: use eye centers for vertical reference (more stable than ears)
    left_eye_mid_y = (get_point(LEFT_EYE_CENTER_TOP)[1] + get_point(LEFT_EYE_CENTER_BOTTOM)[1]) / 2
    right_eye_mid_y = (get_point(RIGHT_EYE_CENTER_TOP)[1] + get_point(RIGHT_EYE_CENTER_BOTTOM)[1]) / 2
    avg_eye_y = (left_eye_mid_y + right_eye_mid_y) / 2

    # Positive = nose above eyes = looking up
    # Negative = nose below eyes = looking down
    pitch_offset = (avg_eye_y - nose[1]) / face_width

    return {
        'yaw_offset': yaw_offset,
        'pitch_offset': pitch_offset,
        'face_width': face_width,
    }


# ============================================================
# Main Image Analysis
# ============================================================

def analyze_image(image_bytes: bytes, session_id: str = "default") -> Optional[str]:
    """Analyzes an image for proctoring violations.
    
    Uses a multi-signal suspicion scoring system with temporal smoothing.
    YOLO detections (multiple people, forbidden objects) still return immediately.
    Face/gaze analysis feeds into a per-session rolling buffer.
    
    Returns a reason string if suspicious, else None.
    """
    try:
        # Convert bytes to OpenCV Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Failed to decode image")
            return "image_decode_error"

        # ===== PHASE 1: YOLO Detection (Immediate Returns) =====
        results = model(img, verbose=False)
        detections = results[0].boxes

        person_count = 0
        suspicious_objects = []

        for box in detections:
            conf = float(box.conf[0])
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == 'person':
                person_count += 1
            elif label in SUSPICIOUS_OBJECTS:
                suspicious_objects.append(label)

        if person_count > 1:
            logger.info(f"Multiple people detected: {person_count}")
            return "multiple_people_detected"

        if suspicious_objects:
            logger.info(f"Suspicious object detected: {suspicious_objects[0]}")
            return f"forbidden_object_{suspicious_objects[0].replace(' ', '_')}"

        # ===== PHASE 2: MediaPipe Face Analysis (Scored + Smoothed) =====
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        face_results = face_landmarker.detect(mp_image)

        if not face_results.face_landmarks:
            # MediaPipe lost the face. Check YOLO results.
            if person_count >= 1:
                # YOLO sees a person, but MediaPipe doesn't.
                # In a math exam context, this 95% means they are looking down writing.
                logger.info("Person detected but face hidden. Assuming writing posture.")
                return None  # SAFE — they're just looking down
            else:
                # YOLO sees no one, MediaPipe sees no one.
                return "user_absent_from_chair"

        num_faces = len(face_results.face_landmarks)
        if num_faces > 1:
            logger.info(f"Multiple faces detected: {num_faces}")
            return "multiple_faces_detected"

        # --- Single face analysis with suspicion scoring ---
        landmarks = face_results.face_landmarks[0]
        img_h, img_w = img.shape[:2]

        suspicion_score = 0.0
        signals: Dict[str, float] = {}

        # 1. Head Pose (improved with cheek/eye landmarks)
        try:
            pose = _compute_head_pose(landmarks, img_w, img_h)
            
            # Determine if they are looking down based on Pitch
            is_looking_down = pose['pitch_offset'] < -0.05

            # Yaw check (head turned sideways)
            if pose['yaw_offset'] > YAW_THRESHOLD:
                # When looking down, be lenient on Yaw
                # Head rotation calculations are less accurate when looking down
                if is_looking_down:
                    # Only flag if it's EXTREME yaw when looking down
                    if pose['yaw_offset'] > (YAW_THRESHOLD * 1.5):
                        suspicion_score += 0.4
                        signals['head_turned_sideways_while_down'] = pose['yaw_offset']
                else:
                    # Normal look-ahead yaw check
                    suspicion_score += 0.4
                    signals['head_turned_sideways'] = pose['yaw_offset']
                    logger.debug(f"Yaw signal: offset={pose['yaw_offset']:.2f}")

            # Pitch check (head up — not allowed)
            if pose['pitch_offset'] > PITCH_UP_THRESHOLD:
                suspicion_score += 0.6
                signals['head_turned_up'] = pose['pitch_offset']
                logger.debug(f"Pitch up signal: offset={pose['pitch_offset']:.2f}")

        except (IndexError, KeyError):
            is_looking_down = False

        # 2. Eye Gaze Estimation
        # Skip gaze check if looking down (iris landmarks are unreliable)
        if not is_looking_down:
            try:
                left_ratio, right_ratio = _compute_gaze_ratio(landmarks, img_w, img_h)
                avg_gaze = (left_ratio + right_ratio) / 2

                if avg_gaze < GAZE_LEFT_THRESHOLD:
                    suspicion_score += 0.5
                    signals['gaze_left'] = avg_gaze
                    logger.debug(f"Gaze signal: L={left_ratio:.2f} R={right_ratio:.2f} avg={avg_gaze:.2f}")
                elif avg_gaze > GAZE_RIGHT_THRESHOLD:
                    suspicion_score += 0.5
                    signals['gaze_right'] = avg_gaze
                    logger.debug(f"Gaze signal: L={left_ratio:.2f} R={right_ratio:.2f} avg={avg_gaze:.2f}")

            except (IndexError, KeyError):
                pass
        else:
            logger.debug("Skipping gaze check (user looking down)")

        # New Logic: Immediate Logging
        if suspicion_score >= SUSPICION_SCORE_THRESHOLD:
            # Get the most suspicious signal name to use as the reason
            reason = max(signals, key=signals.get) if signals else "general_suspicion"
            logger.info(f"IMMEDIATE VIOLATION: {reason} (Score: {suspicion_score})")
            return reason

        # Fallback to tracker for long-term patterns if score is low but non-zero
        tracker = _get_tracker(session_id)
        return tracker.add_frame(suspicion_score, signals)

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return None


# ============================================================
# Audio Analysis (unchanged)
# ============================================================

def analyze_audio(audio_bytes: bytes) -> Optional[str]:
    """Analyzes audio using Silero VAD to differentiate:
    - "speech_detected": Human speech (high probability)
    - "whisper_suspected": Possible speech/whisper (medium probability)
    - "loud_noise_detected": High energy but filtered by VAD (ignored as noise)
    - "suspicious_silence": Audio is completely dead
    """
    try:
        # 1. Load audio with librosa (resample to 16kHz for Silero VAD)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        # 2. Check for Silence / Mic Mute (Basic RMS check)
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = np.mean(rms)

        if mean_rms < 0.0001:
            logger.info(f"Suspicious silence: RMS={mean_rms}")
            return "suspicious_silence"

        # 3. Prepare audio for Silero VAD
        audio_tensor = torch.from_numpy(y)

        # 4. Get Speech Probability
        chunk_size = 512
        speech_probs = []

        for i in range(0, len(audio_tensor), chunk_size):
            chunk = audio_tensor[i:i+chunk_size]

            if len(chunk) < chunk_size:
                pad_size = chunk_size - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

            prob = vad_model(chunk, sr).item()
            speech_probs.append(prob)

        avg_speech_prob = sum(speech_probs) / len(speech_probs) if speech_probs else 0.0

        logger.info(f"Silero VAD Avg Prob: {avg_speech_prob:.4f}, RMS: {mean_rms:.4f}")

        # 5. Decision Logic
        if avg_speech_prob > 0.4:
            return "speech_detected"

        elif 0.15 < avg_speech_prob <= 0.4:
            if mean_rms > 0.002:
                return "whisper_suspected"

        elif mean_rms > 0.05:
            logger.info(f"Loud noise detected (not speech): RMS={mean_rms:.4f}, AvgProb={avg_speech_prob:.4f}")
            return "loud_noise_detected"

        return None

    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return None
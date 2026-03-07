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
import joblib

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

# 4. Initialize Simple-Head-Pose SVR Model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'best_model_svr_23_01_24_17.joblib')
    head_pose_model = joblib.load(model_path)
    logger.info("Simple-Head-Pose SVR model loaded successfully.")
except Exception as e:
    logger.error(f"Simple-Head-Pose SVR model init failed: {e}")
    head_pose_model = None

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

# Simple-Head-Pose specific 7 landmarks
SHP_NOSE = 1
SHP_FOREHEAD = 10
SHP_LEFT_EYE = 33
SHP_MOUTH_LEFT = 61
SHP_CHIN = 199
SHP_RIGHT_EYE = 263
SHP_MOUTH_RIGHT = 291

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

def _compute_head_pose(landmarks_list, img_w: int, img_h: int) -> Dict[str, float]:
    """Compute head yaw and pitch using the Simple-Head-Pose trained SVR model.
    
    Returns dict with:
      'yaw_offset': yaw angle in normalized units mapped to backward compatibility range
      'pitch_offset': pitch angle mapped to backward compatibility range (positive = looking up)
      'face_width': default fallback value 
    """
    if head_pose_model is None:
        return {'yaw_offset': 0.0, 'pitch_offset': 0.0, 'face_width': 0.0}

    # 1. Extract the 7 specific landmarks required by the model
    # Order matters: NOSE, FOREHEAD, LEFT_EYE, MOUTH_LEFT, CHIN, RIGHT_EYE, MOUTH_RIGHT
    target_idx = [SHP_NOSE, SHP_FOREHEAD, SHP_LEFT_EYE, SHP_MOUTH_LEFT, SHP_CHIN, SHP_RIGHT_EYE, SHP_MOUTH_RIGHT]
    
    try:
        # Get x,y landmarks correctly structured
        raw_lms = []
        for idx in target_idx:
            # Not scaling to img_w/img_h yet because the SVR model uses its own relative scaling
            lm = landmarks_list[idx]
            raw_lms.append(np.array([lm.x, lm.y]))
            
        # 2. Replicate Scaling logic exactly as in Simple-Head-Pose
        nose_point = raw_lms[0]
        # Translate to nose=origin
        translated_lms = [lm - nose_point for lm in raw_lms]
        
        # Scale by distance between forehead(index 1) and chin(index 4)
        forehead_point = translated_lms[1]
        chin_point = translated_lms[4]
        reference_length = np.linalg.norm(forehead_point - chin_point)
        
        if reference_length <= 0:
            return {'yaw_offset': 0.0, 'pitch_offset': 0.0, 'face_width': 0.0}
            
        scaled_lms = [lm / reference_length for lm in translated_lms]
        
        # Flatten into 1D array of length 14
        flattened_lms = [item for tuple_item in scaled_lms for item in tuple_item]
        
        # 3. Predict angles (yaw, pitch, roll)
        predictions = head_pose_model.predict([flattened_lms])[0]
        yaw = float(predictions[0])
        pitch = float(predictions[1])
        # roll = predictions[2]
        
        # Normalize SVR output so that forward-looking is ~0
        # Values based on SVR model characteristic output for a centered mock face
        # We also apply a 1.2x multiplier to align sensitivity with previous math-based thresholds
        normalized_yaw = (yaw - 0.13) * 1.2
        normalized_pitch = (pitch + 0.58) * 1.2
        
        return {
            'yaw_offset': abs(normalized_yaw), # Existing code expects absolute yaw
            'pitch_offset': normalized_pitch,
            'face_width': 100.0, # dummy value for compatibility
            'raw_yaw': yaw,
            'raw_pitch': pitch,
            'raw_roll': float(predictions[2]) if len(predictions) > 2 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error computing head pose with SVR: {e}")
        return {'yaw_offset': 0.0, 'pitch_offset': 0.0, 'face_width': 0.0}

def _draw_head_pose_axes(img, yaw, pitch, roll, tdx, tdy, size=50):
    """Draw 3D axes at (tdx, tdy) based on yaw, pitch, roll."""
    try:
        # Replicate Rodrigues-based axis rotation from Simple-Head-Pose drawer.py
        # Simple-Head-Pose uses: yaw = -yaw for flipped axis logic
        rotation_matrix = cv2.Rodrigues(np.array([pitch, -yaw, roll], dtype=np.float64))[0]
        
        # Identity matrix for X, Y, Z axes
        axes_points = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float64)
        
        # Rotate axes
        axes_points = rotation_matrix @ axes_points
        # Scale
        axes_points = (axes_points[:2, :] * size).astype(int)
        
        # Translate to nose point
        axes_points[0, :] += int(tdx)
        axes_points[1, :] += int(tdy)
        
        # Draw lines: X (Blue), Y (Green), Z (Red)
        # Point 3 is the origin (last col of identity matrix was 0,0,0)
        origin = (axes_points[0, 3], axes_points[1, 3])
        cv2.line(img, origin, (axes_points[0, 0], axes_points[1, 0]), (255, 0, 0), 3) # X - Blue
        cv2.line(img, origin, (axes_points[0, 1], axes_points[1, 1]), (0, 255, 0), 3) # Y - Green
        cv2.line(img, origin, (axes_points[0, 2], axes_points[1, 2]), (0, 0, 255), 3) # Z - Red
    except Exception as e:
        logger.debug(f"Axis drawing failed: {e}")


# ============================================================
# Main Image Analysis
# ============================================================

def analyze_image(image_bytes: bytes, session_id: str = "default") -> Tuple[Optional[str], bytes]:
    """Analyzes an image for proctoring violations and returns (reason, annotated_image_bytes).
    
    Uses a multi-signal suspicion scoring system with temporal smoothing.
    YOLO detections (multiple people, forbidden objects) still return immediately.
    Face/gaze analysis feeds into a per-session rolling buffer.
    """
    def get_encoded(image):
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes()

    try:
        # Convert bytes to OpenCV Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Failed to decode image")
            return "image_decode_error", image_bytes


        # ===== PHASE 1: YOLO Detection =====
        results = model(img, verbose=False)
        detections = results[0].boxes

        person_count = 0
        suspicious_objects = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Draw YOLO boxes for transparency
            color = (0, 255, 0) # Green default
            if label == 'person':
                person_count += 1
            elif label in SUSPICIOUS_OBJECTS:
                suspicious_objects.append(label)
                color = (0, 0, 255) # Red for suspicious
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # ===== PHASE 2: MediaPipe Face Analysis (Scored + Smoothed) =====
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        face_results = face_landmarker.detect(mp_image)
        img_h, img_w = img.shape[:2]

        face_reason = None
        face_detected = False
        is_looking_down = False

        if face_results.face_landmarks:
            face_detected = True
            num_faces = len(face_results.face_landmarks)
            if num_faces > 1:
                logger.info(f"Multiple faces detected: {num_faces}")
                cv2.putText(img, "MULTIPLE FACES", (50, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                face_reason = "multiple_faces_detected"
            
            # Use the first face for detailed pose analysis
            landmarks = face_results.face_landmarks[0]
            
            # --- Single face analysis with SVR Head Pose + Gaze ---
            suspicion_score = 0.0
            signals: Dict[str, float] = {}

            # 1. Head Pose (SVR-based robust estimation)
            try:
                # Draw the 7 SVR keypoints for debugging
                target_idx = [SHP_NOSE, SHP_FOREHEAD, SHP_LEFT_EYE, SHP_MOUTH_LEFT, SHP_CHIN, SHP_RIGHT_EYE, SHP_MOUTH_RIGHT]
                for idx in target_idx:
                    lm = landmarks[idx]
                    cv2.circle(img, (int(lm.x * img_w), int(lm.y * img_h)), 3, (255, 0, 255), -1)

                pose = _compute_head_pose(landmarks, img_w, img_h)
                yaw = pose['yaw_offset']
                pitch = pose['pitch_offset']
                
                # Draw 3D axes at the nose for debugging
                nose_lm = landmarks[SHP_NOSE]
                _draw_head_pose_axes(img, pose['raw_yaw'], pose['raw_pitch'], pose['raw_roll'], 
                                     nose_lm.x * img_w, nose_lm.y * img_h, size=60)

                # Determine if they are looking down based on Pitch
                is_looking_down = pitch < -0.05

                # Draw pose info on image for proctoring transparency
                pose_color = (0, 255, 255) # Yellow/Cyan
                cv2.putText(img, f"Yaw: {yaw:.2f} Pitch: {pitch:.2f}", (img_w - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
                if is_looking_down:
                    cv2.putText(img, "STATUS: LOOKING DOWN", (img_w - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Yaw check (head turned sideways)
                if yaw > YAW_THRESHOLD:
                    if is_looking_down:
                        if yaw > (YAW_THRESHOLD * 1.5):
                            suspicion_score += 0.4
                            signals['head_turned_sideways_while_down'] = yaw
                    else:
                        suspicion_score += 0.4
                        signals['head_turned_sideways'] = yaw

                # Pitch check (head turned up — not allowed)
                if pitch > PITCH_UP_THRESHOLD:
                    suspicion_score += 0.6
                    signals['head_turned_up'] = pitch

                # 2. Eye Gaze Estimation (skipped if looking down)
                if not is_looking_down:
                    left_ratio, right_ratio = _compute_gaze_ratio(landmarks, img_w, img_h)
                    avg_gaze = (left_ratio + right_ratio) / 2

                    if avg_gaze < GAZE_LEFT_THRESHOLD:
                        suspicion_score += 0.5
                        signals['gaze_left'] = avg_gaze
                        cv2.putText(img, f"GAZE LEFT: {avg_gaze:.2f}", (10, img_h - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    elif avg_gaze > GAZE_RIGHT_THRESHOLD:
                        suspicion_score += 0.5
                        signals['gaze_right'] = avg_gaze
                        cv2.putText(img, f"GAZE RIGHT: {avg_gaze:.2f}", (10, img_h - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Final suspect check for this frame
                if suspicion_score >= SUSPICION_SCORE_THRESHOLD:
                    face_reason = max(signals, key=signals.get) if signals else "general_suspicion"
                else:
                    tracker = _get_tracker(session_id)
                    face_reason = tracker.add_frame(suspicion_score, signals)

            except Exception as e:
                logger.error(f"Face processing failed: {e}")
        
        else:
            # No face detected by MediaPipe
            if person_count >= 1:
                logger.info("Person detected but face is invisible. Warning issued.")
                cv2.putText(img, "FACE NOT VISIBLE", (50, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                face_reason = "face_not_visible"
            else:
                face_reason = "user_absent_from_chair"

        # ===== PHASE 3: Prioritize and Return =====
        final_reason = None
        if person_count > 1:
            final_reason = "multiple_people_detected"
            cv2.putText(img, "ALERT: MULTIPLE PEOPLE", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif suspicious_objects:
            final_reason = f"forbidden_object_{suspicious_objects[0].replace(' ', '_')}"
            cv2.putText(img, f"ALERT: {suspicious_objects[0].upper()}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Use the face-based reason if YOLO is clean
            final_reason = face_reason

        if final_reason:
            cv2.putText(img, f"FINAL: {final_reason}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return final_reason, get_encoded(img)

        return None, get_encoded(img)

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return None, image_bytes



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
    if not audio_bytes or len(audio_bytes) < 100:
        return None

    try:
        # 1. Load audio with librosa (resample to 16kHz for Silero VAD)
        # Using soundfile backend directly via BytesIO can be flaky if headers are missing
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        except Exception as e:
            # Fallback/Log specifically for format errors
            logger.debug(f"Audio decoding failed: {e}. Attempting raw interpretation.")
            return None

        if len(y) == 0:
            return None

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

            prob = vad_model(chunk, 16000).item()
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
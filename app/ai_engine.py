import os
import logging
import io
from typing import Optional, Dict, Tuple

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

# Head pose thresholds
YAW_THRESHOLD = float(os.environ.get("YAW_THRESHOLD", "0.15"))
PITCH_UP_THRESHOLD = float(os.environ.get("PITCH_UP_THRESHOLD", "0.05"))

# ============================================================
# MediaPipe FaceMesh Landmark Indices
# ============================================================

# Simple-Head-Pose specific 7 landmarks
SHP_NOSE = 1
SHP_FOREHEAD = 10
SHP_LEFT_EYE = 33
SHP_MOUTH_LEFT = 61
SHP_CHIN = 199
SHP_RIGHT_EYE = 263
SHP_MOUTH_RIGHT = 291


# ============================================================
# Head Pose Estimation
# ============================================================

def _compute_head_pose(landmarks_list, img_w: int, img_h: int) -> Dict[str, float]:
    """Compute head yaw and pitch using the Simple-Head-Pose trained SVR model.
    
    Returns dict with:
      'yaw_offset': absolute yaw angle (normalized)
      'pitch_offset': pitch angle (positive = looking up, negative = looking down)
    """
    if head_pose_model is None:
        return {'yaw_offset': 0.0, 'pitch_offset': 0.0, 'face_width': 0.0}

    # Extract the 7 specific landmarks required by the model
    # Order: NOSE, FOREHEAD, LEFT_EYE, MOUTH_LEFT, CHIN, RIGHT_EYE, MOUTH_RIGHT
    target_idx = [SHP_NOSE, SHP_FOREHEAD, SHP_LEFT_EYE, SHP_MOUTH_LEFT, SHP_CHIN, SHP_RIGHT_EYE, SHP_MOUTH_RIGHT]
    
    try:
        raw_lms = []
        for idx in target_idx:
            lm = landmarks_list[idx]
            raw_lms.append(np.array([lm.x, lm.y]))
            
        # Translate so nose is the origin
        nose_point = raw_lms[0]
        translated_lms = [lm - nose_point for lm in raw_lms]
        
        # Scale by forehead-to-chin distance
        forehead_point = translated_lms[1]
        chin_point = translated_lms[4]
        reference_length = np.linalg.norm(forehead_point - chin_point)
        
        if reference_length <= 0:
            return {'yaw_offset': 0.0, 'pitch_offset': 0.0, 'face_width': 0.0}
            
        scaled_lms = [lm / reference_length for lm in translated_lms]
        flattened_lms = [item for tuple_item in scaled_lms for item in tuple_item]
        
        # Predict angles (yaw, pitch, roll)
        predictions = head_pose_model.predict([flattened_lms])[0]
        yaw = float(predictions[0])
        pitch = float(predictions[1])
        
        # Normalize SVR output so that forward-looking is ~0
        normalized_yaw = (yaw - 0.13) * 1.2
        normalized_pitch = (pitch + 0.58) * 1.2
        
        return {
            'yaw_offset': abs(normalized_yaw),
            'pitch_offset': normalized_pitch,
            'face_width': 100.0,
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
        rotation_matrix = cv2.Rodrigues(np.array([pitch, -yaw, roll], dtype=np.float64))[0]
        
        axes_points = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float64)
        
        axes_points = rotation_matrix @ axes_points
        axes_points = (axes_points[:2, :] * size).astype(int)
        
        axes_points[0, :] += int(tdx)
        axes_points[1, :] += int(tdy)
        
        origin = (axes_points[0, 3], axes_points[1, 3])
        cv2.line(img, origin, (axes_points[0, 0], axes_points[1, 0]), (255, 0, 0), 3)  # X - Blue
        cv2.line(img, origin, (axes_points[0, 1], axes_points[1, 1]), (0, 255, 0), 3)  # Y - Green
        cv2.line(img, origin, (axes_points[0, 2], axes_points[1, 2]), (0, 0, 255), 3)  # Z - Red
    except Exception as e:
        logger.debug(f"Axis drawing failed: {e}")


# ============================================================
# Main Image Analysis
# ============================================================

def analyze_image(image_bytes: bytes, session_id: str = "default") -> Tuple[Optional[str], bytes]:
    """Analyzes a single image independently for proctoring violations.

    Returns (violation_reason, annotated_image_bytes).
    Any detected violation is flagged and returned immediately — no scoring,
    no buffering, no session state.
    """
    def get_encoded(image):
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes()

    try:
        # Decode image
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

            color = (0, 255, 0)  # Green default
            if label == 'person':
                person_count += 1
            elif label in SUSPICIOUS_OBJECTS:
                suspicious_objects.append(label)
                color = (0, 0, 255)  # Red for suspicious

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ===== PHASE 2: YOLO-based immediate flags =====
        if person_count > 1:
            reason = "multiple_people_detected"
            cv2.putText(img, "ALERT: MULTIPLE PEOPLE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(img, f"FINAL: {reason}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return reason, get_encoded(img)

        if suspicious_objects:
            reason = f"forbidden_object_{suspicious_objects[0].replace(' ', '_')}"
            cv2.putText(img, f"ALERT: {suspicious_objects[0].upper()}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(img, f"FINAL: {reason}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return reason, get_encoded(img)

        # ===== PHASE 3: MediaPipe Face Analysis =====
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        face_results = face_landmarker.detect(mp_image)
        img_h, img_w = img.shape[:2]

        if face_results.face_landmarks:
            num_faces = len(face_results.face_landmarks)

            # Multiple faces
            if num_faces > 1:
                reason = "multiple_faces_detected"
                cv2.putText(img, "MULTIPLE FACES", (50, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(img, f"FINAL: {reason}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return reason, get_encoded(img)

            # Analyze the single face
            landmarks = face_results.face_landmarks[0]

            try:
                # Draw the 7 SVR keypoints
                target_idx = [SHP_NOSE, SHP_FOREHEAD, SHP_LEFT_EYE, SHP_MOUTH_LEFT,
                               SHP_CHIN, SHP_RIGHT_EYE, SHP_MOUTH_RIGHT]
                for idx in target_idx:
                    lm = landmarks[idx]
                    cv2.circle(img, (int(lm.x * img_w), int(lm.y * img_h)), 3, (255, 0, 255), -1)

                pose = _compute_head_pose(landmarks, img_w, img_h)
                yaw = pose['yaw_offset']
                pitch = pose['pitch_offset']

                # Draw 3D axes
                nose_lm = landmarks[SHP_NOSE]
                _draw_head_pose_axes(img, pose['raw_yaw'], pose['raw_pitch'], pose['raw_roll'],
                                     nose_lm.x * img_w, nose_lm.y * img_h, size=60)

                is_looking_down = pitch < -0.05

                # Draw pose info
                pose_color = (0, 255, 255)
                cv2.putText(img, f"Yaw: {yaw:.2f} Pitch: {pitch:.2f}", (img_w - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
                if is_looking_down:
                    cv2.putText(img, "STATUS: LOOKING DOWN", (img_w - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # --- Immediate violation checks (no scoring, no buffering) ---

                # Head turned sideways
                if yaw > YAW_THRESHOLD:
                    reason = "head_turned_sideways"
                    logger.info(f"Flag: {reason} (yaw={yaw:.2f})")
                    cv2.putText(img, f"FINAL: {reason}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return reason, get_encoded(img)

                # Head looking up (not allowed)
                if pitch > PITCH_UP_THRESHOLD:
                    reason = "head_turned_up"
                    logger.info(f"Flag: {reason} (pitch={pitch:.2f})")
                    cv2.putText(img, f"FINAL: {reason}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return reason, get_encoded(img)

                # Not looking down (neutral/forward is also not allowed)
                if not is_looking_down:
                    reason = "not_looking_down"
                    logger.info(f"Flag: {reason} (pitch={pitch:.2f})")
                    cv2.putText(img, f"FINAL: {reason}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return reason, get_encoded(img)

            except Exception as e:
                logger.error(f"Face processing failed: {e}")

        else:
            # No face detected
            if person_count >= 1:
                reason = "face_not_visible"
                logger.info("Person detected but face is invisible.")
                cv2.putText(img, "FACE NOT VISIBLE", (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(img, f"FINAL: {reason}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return reason, get_encoded(img)
            else:
                reason = "user_absent_from_chair"
                cv2.putText(img, f"FINAL: {reason}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return reason, get_encoded(img)

        # All checks passed — clean frame
        return None, get_encoded(img)

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return None, image_bytes


# ============================================================
# Audio Analysis
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
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        except Exception as e:
            logger.debug(f"Audio decoding failed: {e}.")
            return None

        if len(y) == 0:
            return None

        # Check for Silence / Mic Mute
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = np.mean(rms)

        if mean_rms < 0.0001:
            logger.info(f"Suspicious silence: RMS={mean_rms}")
            return "suspicious_silence"

        # Prepare audio for Silero VAD
        audio_tensor = torch.from_numpy(y)
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
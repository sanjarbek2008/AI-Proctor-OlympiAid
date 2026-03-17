import os
import logging
import io
import urllib.request
from typing import Optional, Dict, Tuple, List

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

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"AI Engine is using device: {device.upper()}")

# 1. Initialize Silero VAD
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    vad_model = vad_model.to(device)
    logger.info(f"Silero VAD model loaded successfully on {device.upper()}.")
except Exception as e:
    logger.error(f"Silero VAD init failed: {e}")
    vad_model = None

# 2. Initialize YOLO
try:
    model = YOLO("yolov8n.pt")
    model.to(device)
    logger.info(f"YOLO model loaded successfully on {device.upper()}.")
except Exception as e:
    logger.error(f"YOLO init failed: {e}")
    model = None

# 3. Initialize MediaPipe Face Landmarker (for face crop when available)
try:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    face_landmarker_path = os.path.join(project_root, 'face_landmarker.task')
    if not os.path.exists(face_landmarker_path):
        face_landmarker_path = 'face_landmarker.task'
    
    def create_landmarker(use_gpu=False):
        delegate = mp_python.BaseOptions.Delegate.GPU if use_gpu else mp_python.BaseOptions.Delegate.CPU
        base_options = mp_python.BaseOptions(model_asset_path=face_landmarker_path, delegate=delegate)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=2
        )
        return vision.FaceLandmarker.create_from_options(options)

    try:
        if device == "cuda":
            face_landmarker = create_landmarker(use_gpu=True)
            logger.info("MediaPipe Face Landmarker loaded successfully on GPU.")
        else:
            face_landmarker = create_landmarker(use_gpu=False)
            logger.info("MediaPipe Face Landmarker loaded successfully on CPU.")
    except Exception as gpu_e:
        if device == "cuda":
            logger.warning(f"MediaPipe Face Landmarker GPU init failed (possibly build flags): {gpu_e}. Falling back to CPU.")
            face_landmarker = create_landmarker(use_gpu=False)
            logger.info("MediaPipe Face Landmarker loaded successfully on CPU (fallback).")
        else:
            raise gpu_e
except Exception as e:
    logger.error(f"MediaPipe Face Landmarker init failed: {e}")
    face_landmarker = None

# 4. Initialize MediaPipe Face Detector (permissive - for face presence check)
face_detector = None
try:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    face_detector_path = os.path.join(project_root, 'blaze_face_short_range.tflite')
    if not os.path.exists(face_detector_path):
        # Download the model if not present
        face_detector_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        try:
            urllib.request.urlretrieve(face_detector_url, face_detector_path)
            logger.info("Downloaded MediaPipe Face Detector model.")
        except Exception as dl_e:
            logger.warning(f"Could not download face detector: {dl_e}. Face presence check may fail.")
    if os.path.exists(face_detector_path):
        def create_detector(use_gpu=False):
            delegate = mp_python.BaseOptions.Delegate.GPU if use_gpu else mp_python.BaseOptions.Delegate.CPU
            fd_options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_detector_path, delegate=delegate),
                min_detection_confidence=0.3,
                min_suppression_threshold=0.5
            )
            return vision.FaceDetector.create_from_options(fd_options)

        try:
            if device == "cuda":
                face_detector = create_detector(use_gpu=True)
                logger.info("MediaPipe Face Detector loaded successfully on GPU.")
            else:
                face_detector = create_detector(use_gpu=False)
                logger.info("MediaPipe Face Detector loaded successfully on CPU.")
        except Exception as gpu_e:
            if device == "cuda":
                logger.warning(f"MediaPipe Face Detector GPU init failed: {gpu_e}. Falling back to CPU.")
                face_detector = create_detector(use_gpu=False)
                logger.info("MediaPipe Face Detector loaded successfully on CPU (fallback).")
            else:
                raise gpu_e
except Exception as e:
    logger.error(f"MediaPipe Face Detector init failed: {e}")
    face_detector = None

# 5. Initialize 6DRepNet for head pose (SixDRepNet_Detector handles model + preprocessing)
head_pose_model = None
try:
    from sixdrepnet.regressor import SixDRepNet_Detector
    # gpu_id: 0 for GPU, -1 for CPU
    gpu_id = 0 if device == "cuda" else -1
    head_pose_model = SixDRepNet_Detector(gpu_id=gpu_id)
    logger.info(f"6DRepNet head pose model loaded successfully on {'GPU' if gpu_id == 0 else 'CPU'}.")
except Exception as e:
    logger.error(f"6DRepNet init failed: {e}")
    head_pose_model = None

# ============================================================
# Configuration (all configurable via env vars)
# ============================================================

# Suspicious YOLO objects
default_objects = "cell phone,remote,laptop,tablet"
sus_objects_env = os.environ.get("SUSPICIOUS_OBJECTS", default_objects)
SUSPICIOUS_OBJECTS = [obj.strip() for obj in sus_objects_env.split(",")]

YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.5"))

# Head pose thresholds (in degrees) - only looking down is allowed
YAW_DEG_THRESHOLD = float(os.environ.get("YAW_DEG_THRESHOLD", "30"))
PITCH_UP_DEG_THRESHOLD = float(os.environ.get("PITCH_UP_DEG_THRESHOLD", "15"))
PITCH_DOWN_MIN_DEG = float(os.environ.get("PITCH_DOWN_MIN_DEG", "-20"))


# ============================================================
# Head Pose Estimation (6DRepNet)
# ============================================================

def _get_face_bbox_from_detections(detections, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Extract the largest face bbox from MediaPipe Face Detector results.
    Returns (x_min, y_min, x_max, y_max) in pixel coords or None.
    Handles both normalized (0-1) and pixel coordinates.
    """
    if not detections or len(detections) == 0:
        return None
    best = None
    best_area = 0
    for det in detections:
        bbox = det.bounding_box
        ox, oy = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height
        # MediaPipe Tasks may return normalized (0-1) or pixel coords
        if 0 <= ox <= 1 and 0 <= oy <= 1 and 0 < w <= 1 and 0 < h <= 1:
            x_min = int(ox * img_w)
            y_min = int(oy * img_h)
            x_max = int((ox + w) * img_w)
            y_max = int((oy + h) * img_h)
        else:
            x_min = int(ox)
            y_min = int(oy)
            x_max = int(ox + w)
            y_max = int(oy + h)
        x_min = max(0, min(x_min, img_w - 1))
        y_min = max(0, min(y_min, img_h - 1))
        x_max = max(x_min + 1, min(x_max, img_w))
        y_max = max(y_min + 1, min(y_max, img_h))
        area = (x_max - x_min) * (y_max - y_min)
        if area > best_area:
            best_area = area
            best = (x_min, y_min, x_max, y_max)
    return best


def _get_face_bbox_from_landmarks(landmarks_list, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Extract face bbox from MediaPipe landmarks (min/max of all points)."""
    if not landmarks_list or len(landmarks_list) < 10:
        return None
    xs = [lm.x * img_w for lm in landmarks_list]
    ys = [lm.y * img_h for lm in landmarks_list]
    x_min = max(0, int(min(xs) - 20))
    y_min = max(0, int(min(ys) - 20))
    x_max = min(img_w, int(max(xs) + 20))
    y_max = min(img_h, int(max(ys) + 20))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def _compute_head_pose_6drepnet(img_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int],
                                img_h: int, img_w: int) -> Optional[Dict[str, float]]:
    """Compute head pose using 6DRepNet on a face crop.
    Returns dict with yaw_deg, pitch_deg, roll_deg, or None on failure.
    """
    if head_pose_model is None:
        return None
    x_min, y_min, x_max, y_max = face_bbox
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    if bbox_w < 10 or bbox_h < 10:
        return None
    # Add 20% padding
    pad_w = int(0.2 * bbox_w)
    pad_h = int(0.2 * bbox_h)
    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(img_w, x_max + pad_w)
    y_max = min(img_h, y_max + pad_h)
    crop = img_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    try:
        pitch_arr, yaw_arr, roll_arr = head_pose_model.predict(crop)
        pitch_deg = float(pitch_arr[0])
        yaw_deg = float(yaw_arr[0])
        roll_deg = float(roll_arr[0])
        return {
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
            'roll_deg': roll_deg,
            'center_x': (x_min + x_max) // 2,
            'center_y': (y_min + y_max) // 2,
        }
    except Exception as e:
        logger.debug(f"6DRepNet inference failed: {e}")
        return None


def _draw_head_pose_axes(img, yaw_deg: float, pitch_deg: float, roll_deg: float,
                        tdx: float, tdy: float, size=50):
    """Draw 3D axes at (tdx, tdy) based on yaw, pitch, roll in degrees."""
    try:
        yaw_rad = np.deg2rad(yaw_deg)
        pitch_rad = np.deg2rad(pitch_deg)
        roll_rad = np.deg2rad(roll_deg)
        rotation_matrix = cv2.Rodrigues(np.array([pitch_rad, -yaw_rad, roll_rad], dtype=np.float64))[0]
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


def _draw_all_annotations(img, img_h: int, img_w: int,
                          yolo_drawn: bool,
                          face_bbox: Optional[Tuple[int, int, int, int]],
                          pose: Optional[Dict[str, float]],
                          violation_reason: Optional[str],
                          alert_text: Optional[str] = None):
    """Draw all available annotations on the image for debugging."""
    # YOLO is already drawn by caller before this
    # Draw face bbox if available
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, "FACE", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # Draw pose axes and info if available
    if pose is not None:
        cx = pose.get('center_x', img_w // 2)
        cy = pose.get('center_y', img_h // 2)
        _draw_head_pose_axes(img, pose['yaw_deg'], pose['pitch_deg'], pose['roll_deg'], cx, cy, size=60)
        pose_color = (0, 255, 255)
        cv2.putText(img, f"Yaw: {pose['yaw_deg']:.1f} Pitch: {pose['pitch_deg']:.1f}", (img_w - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        if pose['pitch_deg'] < PITCH_DOWN_MIN_DEG:
            cv2.putText(img, "STATUS: LOOKING DOWN", (img_w - 250, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Draw violation overlay
    if violation_reason:
        cv2.putText(img, f"FINAL: {violation_reason}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if alert_text:
        cv2.putText(img, alert_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


# ============================================================
# Main Image Analysis
# ============================================================

def analyze_image(image_bytes: bytes, session_id: str = "default") -> Tuple[Optional[str], bytes]:
    """Analyzes a single image independently for proctoring violations.

    Returns (violation_reason, annotated_image_bytes).
    Any detected violation is flagged. The returned image includes ALL annotations
    (YOLO, face bbox, head pose axes, violation text) for debugging.
    """
    def get_encoded(image):
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes()

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            return "image_decode_error", image_bytes

        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # ===== PHASE 1: YOLO Detection =====
        if model is None:
            logger.error("YOLO model not loaded")
            return "model_error", get_encoded(img)
        results = model(img, verbose=False)
        detections = results[0].boxes if results and len(results) > 0 else []

        person_count = 0
        suspicious_objects = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            color = (0, 255, 0)
            if label == 'person':
                person_count += 1
            elif label in SUSPICIOUS_OBJECTS:
                suspicious_objects.append(label)
                color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ===== PHASE 2: Face Detector (permissive - for face presence) =====
        face_detections = []
        if face_detector:
            try:
                fd_result = face_detector.detect(mp_image)
                face_detections = fd_result.detections if fd_result and hasattr(fd_result, 'detections') else []
            except Exception as e:
                logger.debug(f"Face detector failed: {e}")

        # Fallback: if Face Detector not loaded, use Face Landmarker for face presence
        if len(face_detections) == 0 and face_landmarker:
            try:
                fl_result = face_landmarker.detect(mp_image)
                if fl_result.face_landmarks and len(fl_result.face_landmarks) > 0:
                    lm_bbox = _get_face_bbox_from_landmarks(fl_result.face_landmarks[0], img_w, img_h)
                    if lm_bbox:
                        x1, y1, x2, y2 = lm_bbox
                        _bbox = type('BBox', (), {'origin_x': x1, 'origin_y': y1, 'width': x2-x1, 'height': y2-y1})()
                        _det = type('Det', (), {'bounding_box': _bbox})()
                        face_detections = [_det]
            except Exception as e:
                logger.debug(f"Face landmarker fallback failed: {e}")

        face_present = len(face_detections) >= 1
        face_bbox = _get_face_bbox_from_detections(face_detections, img_w, img_h) if face_detections else None

        # ===== PHASE 3: YOLO-based immediate flags =====
        if person_count > 1:
            _draw_all_annotations(img, img_h, img_w, True, face_bbox, None,
                                 "multiple_people_detected", "ALERT: MULTIPLE PEOPLE")
            return "multiple_people_detected", get_encoded(img)

        if suspicious_objects:
            reason = f"forbidden_object_{suspicious_objects[0].replace(' ', '_')}"
            _draw_all_annotations(img, img_h, img_w, True, face_bbox, None,
                                 reason, f"ALERT: {suspicious_objects[0].upper()}")
            return reason, get_encoded(img)

        # ===== PHASE 4: Face presence check (permissive: any part of face = OK) =====
        if not face_present:
            if person_count >= 1:
                _draw_all_annotations(img, img_h, img_w, True, None, None,
                                     "face_not_visible", "FACE NOT VISIBLE")
                logger.info("Person detected but no face (Face Detector found zero faces).")
                return "face_not_visible", get_encoded(img)
            else:
                _draw_all_annotations(img, img_h, img_w, True, None, None,
                                     "user_absent_from_chair", None)
                return "user_absent_from_chair", get_encoded(img)

        # ===== PHASE 5: Multiple faces =====
        if len(face_detections) > 1:
            _draw_all_annotations(img, img_h, img_w, True, face_bbox, None,
                                 "multiple_faces_detected", "MULTIPLE FACES")
            return "multiple_faces_detected", get_encoded(img)

        # ===== PHASE 6: Head pose (single face, face present) =====
        # Prefer Face Landmarker bbox for pose (often more accurate), fallback to Face Detector bbox
        pose_bbox = face_bbox
        if face_landmarker:
            try:
                fl_result = face_landmarker.detect(mp_image)
                if fl_result.face_landmarks and len(fl_result.face_landmarks) > 0:
                    lm_bbox = _get_face_bbox_from_landmarks(fl_result.face_landmarks[0], img_w, img_h)
                    if lm_bbox:
                        pose_bbox = lm_bbox
            except Exception as e:
                logger.debug(f"Face landmarker failed: {e}")

        pose = None
        if pose_bbox and head_pose_model:
            pose = _compute_head_pose_6drepnet(img, pose_bbox, img_h, img_w)

        # ===== PHASE 7: Head pose violation checks (only looking down allowed) =====
        if pose is not None:
            yaw_deg = pose['yaw_deg']
            pitch_deg = pose['pitch_deg']

            if abs(yaw_deg) > YAW_DEG_THRESHOLD:
                _draw_all_annotations(img, img_h, img_w, True, face_bbox, pose,
                                     "head_turned_sideways", f"HEAD SIDEWAYS (yaw={yaw_deg:.1f})")
                logger.info(f"Flag: head_turned_sideways (yaw={yaw_deg:.1f})")
                return "head_turned_sideways", get_encoded(img)

            if pitch_deg > PITCH_UP_DEG_THRESHOLD:
                _draw_all_annotations(img, img_h, img_w, True, face_bbox, pose,
                                     "head_turned_up", f"HEAD UP (pitch={pitch_deg:.1f})")
                logger.info(f"Flag: head_turned_up (pitch={pitch_deg:.1f})")
                return "head_turned_up", get_encoded(img)

        else:
            # Pose could not be computed (e.g. partial face, 6DRepNet failed)
            # Face is present (permissive check passed) - allow frame (no pose check)
            pass

        # All checks passed
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

        rms = librosa.feature.rms(y=y)[0]
        mean_rms = np.mean(rms)

        if mean_rms < 0.0001:
            logger.info(f"Suspicious silence: RMS={mean_rms}")
            return "suspicious_silence"

        audio_tensor = torch.from_numpy(y).to(device)
        chunk_size = 512
        speech_probs = []

        for i in range(0, len(audio_tensor), chunk_size):
            chunk = audio_tensor[i:i+chunk_size]
            if len(chunk) < chunk_size:
                pad_size = chunk_size - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            prob = vad_model(chunk.to(device), 16000).item()
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

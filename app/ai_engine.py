import os
import logging
import io
import numpy as np
import scipy.io.wavfile as wav
import librosa
import webrtcvad
import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()

# Configurable suspicious objects list
default_objects = "cell phone,remote,book,laptop,tablet"
sus_objects_env = os.environ.get("SUSPICIOUS_OBJECTS", default_objects)
SUSPICIOUS_OBJECTS = [obj.strip() for obj in sus_objects_env.split(",")]

YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.5"))

# MediaPipe FaceMesh landmark indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
LEFT_EAR = 234
RIGHT_EAR = 454


def analyze_image(image_bytes: bytes) -> str:
    """
    Analyzes an image for proctoring violations.
    Returns a reason string if suspicious, else None.
    """
    try:
        # Convert bytes to OpenCV Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return "image_decode_error"

        # ===== PHASE 1: YOLO Detection =====
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

        # ===== PHASE 2: MediaPipe Face Analysis =====
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        face_results = face_landmarker.detect(mp_image)

        if not face_results.face_landmarks:
            logger.info("No face landmarks detected")
            return "no_face_detected"
        
        num_faces = len(face_results.face_landmarks)
        if num_faces > 1:
            logger.info(f"Multiple faces detected: {num_faces}")
            return "multiple_faces_detected"

        # Analyze single face
        landmarks = face_results.face_landmarks[0]
        img_h, img_w = img.shape[:2]
        
        def get_point(idx):
            lm = landmarks[idx]
            return (lm.x * img_w, lm.y * img_h)

        # Gaze Detection
        try:
            left_iris = get_point(LEFT_IRIS_CENTER)
            left_inner = get_point(LEFT_EYE_INNER)
            left_outer = get_point(LEFT_EYE_OUTER)
            
            right_iris = get_point(RIGHT_IRIS_CENTER)
            right_inner = get_point(RIGHT_EYE_INNER)
            right_outer = get_point(RIGHT_EYE_OUTER)
            
            left_eye_width = abs(left_outer[0] - left_inner[0])
            right_eye_width = abs(right_outer[0] - right_inner[0])
            
            if left_eye_width > 0 and right_eye_width > 0:
                left_gaze_ratio = (left_iris[0] - left_outer[0]) / left_eye_width
                right_gaze_ratio = (right_iris[0] - right_outer[0]) / right_eye_width
                avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
                
                if avg_gaze_ratio < 0.25 or avg_gaze_ratio > 0.75:
                    logger.info(f"Gaze averted: ratio={avg_gaze_ratio:.2f}")
                    return "gaze_averted"
        except (IndexError, KeyError):
            pass

        # Head Pose
        try:
            nose = get_point(NOSE_TIP)
            left_ear = get_point(LEFT_EAR)
            right_ear = get_point(RIGHT_EAR)
            
            face_center_x = (left_ear[0] + right_ear[0]) / 2
            face_width = abs(right_ear[0] - left_ear[0])
            
            if face_width > 0:
                nose_offset = abs(nose[0] - face_center_x) / face_width
                if nose_offset > 0.30:
                    logger.info(f"Head turned: nose_offset={nose_offset:.2f}")
                    return "head_turned"
        except (IndexError, KeyError):
            pass

        return None

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return None


def analyze_audio(audio_bytes: bytes) -> str:
    """
    analyzes audio using VAD (Voice Activity Detection) and spectral features.
    Returns:
    - "speech_detected": Human speech found
    - "loud_noise_detected": High energy but not speech (e.g. slamming door)
    - "suspicious_silence": Audio is completely dead (possible mic mute)
    """
    try:
        # 1. Load audio with librosa (resample to 16kHz for VAD)
        # librosa handles bytes via soundfile if we pass a BytesIO
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # 2. Check for Silence / Mic Mute
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = np.mean(rms)
        
        # Very low threshold for "dead air"
        if mean_rms < 0.0001:
            logger.info(f"Suspicious silence: RMS={mean_rms}")
            return "suspicious_silence"

        # 3. VAD Check (Voice Activity Detection)
        # Convert float32 to int16
        pcm_data = (y * 32767).astype(np.int16)
        
        # VAD requires 10, 20, or 30ms frames
        # 16000Hz * 0.03s = 480 samples
        frame_duration_ms = 30
        frame_size = int(sr * frame_duration_ms / 1000)
        
        num_speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(pcm_data), frame_size):
            frame = pcm_data[i:i+frame_size]
            if len(frame) == frame_size:
                total_frames += 1
                # Convert frame to bytes
                if vad.is_speech(frame.tobytes(), sr):
                    num_speech_frames += 1
        
        speech_ratio = num_speech_frames / total_frames if total_frames > 0 else 0
        
        if speech_ratio > 0.2:  # If >20% of audio contains speech
            logger.info(f"Speech detected: ratio={speech_ratio:.2f}")
            return "speech_detected"

        # 4. Loud Noise Check (if no speech)
        # If mean RMS is high but VAD says no speech, it's likely noise
        if mean_rms > 0.05:
            logger.info(f"Loud noise detected: RMS={mean_rms:.4f}")
            return "loud_noise_detected"

        return None

    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return None
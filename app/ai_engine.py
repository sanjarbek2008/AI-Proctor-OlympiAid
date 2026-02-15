import os
import logging
import io
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

# 1. Initialize Silero VAD
try:
    # Use torch.hub to load the pre-trained Silero VAD model
    # trust_repo=True is often needed for hub loading
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
    # Assuming yolov8n.pt is in the current working directory
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

        # Head Pose Detection (Pitch & Yaw)
        try:
            nose = get_point(NOSE_TIP)
            left_ear = get_point(LEFT_EAR)
            right_ear = get_point(RIGHT_EAR)
            
            face_center_x = (left_ear[0] + right_ear[0]) / 2
            avg_ear_y = (left_ear[1] + right_ear[1]) / 2
            face_width = abs(right_ear[0] - left_ear[0])
            
            if face_width > 0:
                # 1. Yaw (Head Turned Left/Right)
                # Calculate horizontal offset relative to face width
                nose_offset_x = abs(nose[0] - face_center_x) / face_width
                
                if nose_offset_x > 0.30:
                    logger.info(f"Head turned sideways: offset={nose_offset_x:.2f}")
                    return "head_turned_sideways"

                # 2. Pitch (Head Up/Down)
                # Calculate vertical offset relative to face width
                
                # In image coordinates (Y increases downwards):
                # - Looking DOWN: Nose moves DOWN (Y increases) -> (avg_ear_y - nose_y) becomes more negative
                # - Looking UP: Nose moves UP (Y decreases) -> (avg_ear_y - nose_y) becomes positive
                
                vertical_offset = (avg_ear_y - nose[1]) / face_width
                
                # Threshold for Looking UP
                # A positive value means nose is ABOVE the ear line.
                if vertical_offset > 0.20:
                    logger.info(f"Head turned up: offset={vertical_offset:.2f}")
                    return "head_turned_up"
                    
                # We specifically ALLOW looking down (negative vertical_offset), so no check for that.
                
        except (IndexError, KeyError):
            pass

        return None

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return None


def analyze_audio(audio_bytes: bytes) -> str:
    """
    Analyzes audio using Silero VAD to differentiate:
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
        # Silero expects a float32 tensor
        # Librosa loads as float32 numpy array, so just convert to tensor
        audio_tensor = torch.from_numpy(y)
        
        # 4. Get Speech Probability
        # Silero VAD requires chunks of 512 samples for 16kHz
        chunk_size = 512
        speech_probs = []
        
        # Iterate over audio in chunks
        for i in range(0, len(audio_tensor), chunk_size):
            chunk = audio_tensor[i:i+chunk_size]
            
            # If chunk is too small, pad it (or ignore if very small)
            if len(chunk) < chunk_size:
                pad_size = chunk_size - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            
            # Add batch dimension: (1, 512)
            # But vad_model expects (batch, samples) ?? checking docs...
            # Actually vad_model(x, sr) where x is (N,) or (1, N) 
            # The error said "Provided number of samples is ... Supported values: 512"
            # So it expects exactly 512 samples (or multiple of it if it handles it, but error suggests 512)
            
            # Let's ensure it is 1D tensor of 512
            prob = vad_model(chunk, sr).item()
            speech_probs.append(prob)
        
        avg_speech_prob = sum(speech_probs) / len(speech_probs) if speech_probs else 0.0
        
        logger.info(f"Silero VAD Avg Prob: {avg_speech_prob:.4f}, RMS: {mean_rms:.4f}")
        
        # 5. Decision Logic
        # Adjusted thresholds for average probability
        if avg_speech_prob > 0.4:  # 0.4 avg is actually quite high for continuous speech
            return "speech_detected"
        
        elif 0.15 < avg_speech_prob <= 0.4:
            # Check RMS to ensure it's not just quiet noise
            if mean_rms > 0.002: 
                return "whisper_suspected"
            
        # If prob <= 0.15, it's considered noise or silence.
        elif mean_rms > 0.05:
             logger.info(f"Loud noise detected (not speech): RMS={mean_rms:.4f}, AvgProb={avg_speech_prob:.4f}")
             return "loud_noise_detected"

        return None

    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return None
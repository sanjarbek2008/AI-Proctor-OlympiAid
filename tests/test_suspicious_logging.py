
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock YOLO and MediaPipe before importing ai_engine
with patch('ultralytics.YOLO'), \
     patch('mediapipe.tasks.python.vision.FaceLandmarker.create_from_options'):
    from app.ai_engine import analyze_image, SUSPICION_SCORE_THRESHOLD

@pytest.fixture
def mock_img_bytes():
    return b"fake_image_bytes"

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine._compute_head_pose')
@patch('app.ai_engine._compute_gaze_ratio')
@patch('app.ai_engine._get_tracker')
def test_suspicious_frame_returned(mock_get_tracker, mock_gaze, mock_pose, 
                                   mock_mp_image, mock_mp_cvt_color, mock_imdecode, 
                                   mock_mp_landmarker, mock_yolo, mock_img_bytes):
    # Setup mocks
    mock_imdecode.return_value = MagicMock()
    mock_yolo.return_value = [MagicMock(boxes=[])] # No YOLO detections
    
    # MediaPipe results
    mock_face_landmarks = [MagicMock()]
    mock_mp_landmarker.detect.return_value = MagicMock(face_landmarks=mock_face_landmarks)
    
    # Head Pose: Looking up (Pitch 0.3) -> Score 0.6
    mock_pose.return_value = {
        'yaw_offset': 0.0,
        'pitch_offset': 0.3,
        'face_width': 100
    }
    
    # Gaze: centered
    mock_gaze.return_value = (0.5, 0.5)
    
    # Tracker: No sustained violation
    mock_tracker = MagicMock()
    mock_tracker.add_frame.return_value = None
    mock_get_tracker.return_value = mock_tracker
    
    # Run analysis
    result = analyze_image(mock_img_bytes, session_id="test_session")
    
    # Verify suspicious frame is returned immediately
    assert result == "head_turned_up"

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine._get_tracker')
def test_yolo_return_priority(mock_get_tracker, mock_imdecode, mock_mp_landmarker, 
                               mock_yolo, mock_img_bytes):
    # Setup mocks
    mock_imdecode.return_value = MagicMock()
    
    # YOLO detects a phone
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_yolo.return_value = [MagicMock(boxes=[mock_box])]
    mock_yolo.names = {0: 'cell phone'}
    
    # Run analysis
    result = analyze_image(mock_img_bytes, session_id="test_session")
    
    # Verify YOLO flag is returned immediately
    assert result == "forbidden_object_cell_phone"

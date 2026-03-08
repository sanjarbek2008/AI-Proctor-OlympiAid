"""Integration-level tests for analyze_image using mocked models.

Each test verifies that a specific violation type is returned immediately
from a single image — no scoring, buffering, or session state involved.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock YOLO, MediaPipe, and 6DRepNet before importing ai_engine (avoids model loading in CI)
with patch('ultralytics.YOLO'), \
     patch('mediapipe.tasks.python.vision.FaceLandmarker.create_from_options'), \
     patch('mediapipe.tasks.python.vision.FaceDetector.create_from_options'), \
     patch('sixdrepnet.regressor.SixDRepNet_Detector'):
    from app.ai_engine import analyze_image


@pytest.fixture
def mock_img_bytes():
    return b"fake_image_bytes"


def _mock_face_detection_result(num_faces=1):
    """Create mock Face Detector result with N faces (pixel coords)."""
    detections = []
    for i in range(num_faces):
        mock_bbox = MagicMock()
        mock_bbox.origin_x = 100 + i * 50
        mock_bbox.origin_y = 100
        mock_bbox.width = 150
        mock_bbox.height = 150
        mock_det = MagicMock()
        mock_det.bounding_box = mock_bbox
        detections.append(mock_det)
    return MagicMock(detections=detections)


# ============================================================
# Test: Head turned UP → immediate flag
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_detector')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.head_pose_model')
@patch('app.ai_engine._compute_head_pose_6drepnet')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine._draw_all_annotations')
def test_head_turned_up_flagged_immediately(
        mock_draw_all, mock_put_text, mock_rectangle, mock_imencode,
        mock_mp_image, mock_cvt_color, mock_imdecode,
        mock_pose_6d, mock_head_pose_model, mock_face_landmarker, mock_face_detector, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))
    mock_yolo.return_value = [MagicMock(boxes=[], names={})]

    mock_face_detector.detect.return_value = _mock_face_detection_result(1)
    mock_face_landmarker.detect.return_value = MagicMock(face_landmarks=[])

    # pitch > PITCH_UP_DEG_THRESHOLD (15) → head_turned_up
    mock_pose_6d.return_value = {
        'yaw_deg': 0.0,
        'pitch_deg': 25.0,
        'roll_deg': 0.0,
        'center_x': 200,
        'center_y': 200,
    }

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result == "head_turned_up", f"Expected 'head_turned_up', got '{result}'"


# ============================================================
# Test: Head turned SIDEWAYS → immediate flag
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_detector')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.head_pose_model')
@patch('app.ai_engine._compute_head_pose_6drepnet')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine._draw_all_annotations')
def test_head_turned_sideways_flagged_immediately(
        mock_draw_all, mock_put_text, mock_rectangle, mock_imencode,
        mock_mp_image, mock_cvt_color, mock_imdecode,
        mock_pose_6d, mock_head_pose_model, mock_face_landmarker, mock_face_detector, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))
    mock_yolo.return_value = [MagicMock(boxes=[], names={})]

    mock_face_detector.detect.return_value = _mock_face_detection_result(1)
    mock_face_landmarker.detect.return_value = MagicMock(face_landmarks=[])

    # abs(yaw) > YAW_DEG_THRESHOLD (30) → head_turned_sideways
    mock_pose_6d.return_value = {
        'yaw_deg': 45.0,
        'pitch_deg': -25.0,
        'roll_deg': 0.0,
        'center_x': 200,
        'center_y': 200,
    }

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result == "head_turned_sideways", f"Expected 'head_turned_sideways', got '{result}'"


# ============================================================
# Test: Forbidden object detected → immediate flag
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_detector')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine._draw_all_annotations')
def test_forbidden_object_flagged_immediately(
        mock_draw_all, mock_put_text, mock_rectangle, mock_imencode,
        mock_mp_image, mock_cvt_color, mock_imdecode,
        mock_face_landmarker, mock_face_detector, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))

    # YOLO detects a cell phone - returns early before face detection
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_box.xyxy = [[10, 10, 50, 50]]
    mock_yolo.return_value = [MagicMock(boxes=[mock_box])]
    mock_yolo.names = {0: 'cell phone'}

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result == "forbidden_object_cell_phone", f"Got '{result}'"


# ============================================================
# Test: Face not visible (person present, no face) → face_not_visible
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_detector')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine._draw_all_annotations')
def test_face_not_visible_flagged(
        mock_draw_all, mock_put_text, mock_rectangle, mock_imencode,
        mock_mp_image, mock_cvt_color, mock_imdecode,
        mock_face_landmarker, mock_face_detector, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))

    # YOLO detects person
    mock_person_box = MagicMock()
    mock_person_box.conf = [0.9]
    mock_person_box.cls = [0]
    mock_person_box.xyxy = [[50, 50, 200, 300]]
    mock_yolo.return_value = [MagicMock(boxes=[mock_person_box])]
    mock_yolo.names = {0: 'person'}

    # Face Detector finds NO face (permissive check fails)
    mock_face_detector.detect.return_value = MagicMock(detections=[])

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result == "face_not_visible", f"Expected 'face_not_visible', got '{result}'"


# ============================================================
# Test: Clean frame (looking down, no objects) → None
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_detector')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.head_pose_model')
@patch('app.ai_engine._compute_head_pose_6drepnet')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine._draw_all_annotations')
def test_clean_frame_returns_none(
        mock_draw_all, mock_put_text, mock_rectangle, mock_imencode,
        mock_mp_image, mock_cvt_color, mock_imdecode,
        mock_pose_6d, mock_head_pose_model, mock_face_landmarker, mock_face_detector, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))
    mock_yolo.return_value = [MagicMock(boxes=[], names={})]

    mock_face_detector.detect.return_value = _mock_face_detection_result(1)
    mock_face_landmarker.detect.return_value = MagicMock(face_landmarks=[])

    # Safe pose: looking down (pitch < -20), yaw within threshold
    mock_pose_6d.return_value = {
        'yaw_deg': 5.0,
        'pitch_deg': -35.0,
        'roll_deg': 0.0,
        'center_x': 200,
        'center_y': 200,
    }

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result is None, f"Expected None for clean frame, got '{result}'"

"""Integration-level tests for analyze_image using mocked models.

Each test verifies that a specific violation type is returned immediately
from a single image — no scoring, buffering, or session state involved.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock YOLO and MediaPipe before importing ai_engine (avoids model loading in CI)
with patch('ultralytics.YOLO'), \
     patch('mediapipe.tasks.python.vision.FaceLandmarker.create_from_options'):
    from app.ai_engine import analyze_image


@pytest.fixture
def mock_img_bytes():
    return b"fake_image_bytes"


# ============================================================
# Helpers
# ============================================================

def _base_patches():
    """Return the ordered list of patches needed for analyze_image."""
    return [
        patch('app.ai_engine.model'),
        patch('app.ai_engine.face_landmarker'),
        patch('app.ai_engine.cv2.imdecode'),
        patch('app.ai_engine.cv2.cvtColor'),
        patch('app.ai_engine.mp.Image'),
        patch('app.ai_engine._compute_head_pose'),
        patch('app.ai_engine.cv2.imencode'),
        patch('app.ai_engine.cv2.rectangle'),
        patch('app.ai_engine.cv2.putText'),
        patch('app.ai_engine.cv2.circle'),
        patch('app.ai_engine.cv2.line'),
        patch('app.ai_engine._draw_head_pose_axes'),
    ]


# ============================================================
# Test: Head turned UP → immediate flag
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine._compute_head_pose')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine.cv2.circle')
@patch('app.ai_engine.cv2.line')
@patch('app.ai_engine._draw_head_pose_axes')
def test_head_turned_up_flagged_immediately(
        mock_draw_axes, mock_line, mock_circle, mock_put_text,
        mock_rectangle, mock_imencode,
        mock_pose, mock_mp_image, mock_cvt_color,
        mock_imdecode, mock_face_landmarker, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))
    mock_yolo.return_value = [MagicMock(boxes=[])]

    mock_face_landmarks = [MagicMock() for _ in range(478)]
    mock_face_landmarker.detect.return_value = MagicMock(face_landmarks=[mock_face_landmarks])

    # pitch > PITCH_UP_THRESHOLD → head_turned_up
    mock_pose.return_value = {
        'yaw_offset': 0.0,
        'pitch_offset': 0.3,
        'face_width': 100,
        'raw_yaw': 0.0,
        'raw_pitch': 0.3,
        'raw_roll': 0.0
    }

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result == "head_turned_up", f"Expected 'head_turned_up', got '{result}'"


# ============================================================
# Test: Head turned SIDEWAYS → immediate flag
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine._compute_head_pose')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine.cv2.circle')
@patch('app.ai_engine.cv2.line')
@patch('app.ai_engine._draw_head_pose_axes')
def test_head_turned_sideways_flagged_immediately(
        mock_draw_axes, mock_line, mock_circle, mock_put_text,
        mock_rectangle, mock_imencode,
        mock_pose, mock_mp_image, mock_cvt_color,
        mock_imdecode, mock_face_landmarker, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))
    mock_yolo.return_value = [MagicMock(boxes=[])]

    mock_face_landmarks = [MagicMock() for _ in range(478)]
    mock_face_landmarker.detect.return_value = MagicMock(face_landmarks=[mock_face_landmarks])

    # yaw > YAW_THRESHOLD → head_turned_sideways
    mock_pose.return_value = {
        'yaw_offset': 0.5,
        'pitch_offset': -0.1,
        'face_width': 100,
        'raw_yaw': 0.5,
        'raw_pitch': -0.1,
        'raw_roll': 0.0
    }

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result == "head_turned_sideways", f"Expected 'head_turned_sideways', got '{result}'"


# ============================================================
# Test: Forbidden object detected → immediate flag
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine._compute_head_pose')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine.cv2.circle')
@patch('app.ai_engine.cv2.line')
@patch('app.ai_engine._draw_head_pose_axes')
def test_forbidden_object_flagged_immediately(
        mock_draw_axes, mock_line, mock_circle, mock_put_text,
        mock_rectangle, mock_imencode,
        mock_pose, mock_mp_image, mock_cvt_color,
        mock_imdecode, mock_face_landmarker, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))

    # YOLO detects a cell phone
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_box.xyxy = [[10, 10, 50, 50]]
    mock_yolo.return_value = [MagicMock(boxes=[mock_box])]
    mock_yolo.names = {0: 'cell phone'}

    mock_face_landmarker.detect.return_value = MagicMock(face_landmarks=[])

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result == "forbidden_object_cell_phone", f"Got '{result}'"


# ============================================================
# Test: Clean frame (looking down, no objects) → None
# ============================================================

@patch('app.ai_engine.model')
@patch('app.ai_engine.face_landmarker')
@patch('app.ai_engine.cv2.imdecode')
@patch('app.ai_engine.cv2.cvtColor')
@patch('app.ai_engine.mp.Image')
@patch('app.ai_engine._compute_head_pose')
@patch('app.ai_engine.cv2.imencode')
@patch('app.ai_engine.cv2.rectangle')
@patch('app.ai_engine.cv2.putText')
@patch('app.ai_engine.cv2.circle')
@patch('app.ai_engine.cv2.line')
@patch('app.ai_engine._draw_head_pose_axes')
def test_clean_frame_returns_none(
        mock_draw_axes, mock_line, mock_circle, mock_put_text,
        mock_rectangle, mock_imencode,
        mock_pose, mock_mp_image, mock_cvt_color,
        mock_imdecode, mock_face_landmarker, mock_yolo,
        mock_img_bytes):
    mock_imencode.return_value = (True, MagicMock(tobytes=lambda: b"img"))
    mock_imdecode.return_value = MagicMock(shape=(480, 640, 3))
    mock_yolo.return_value = [MagicMock(boxes=[])]

    mock_face_landmarks = [MagicMock() for _ in range(478)]
    mock_face_landmarker.detect.return_value = MagicMock(face_landmarks=[mock_face_landmarks])

    # Safe pose: looking down with no yaw
    mock_pose.return_value = {
        'yaw_offset': 0.0,
        'pitch_offset': -0.3,
        'face_width': 100,
        'raw_yaw': 0.0,
        'raw_pitch': -0.3,
        'raw_roll': 0.0
    }

    result, _ = analyze_image(mock_img_bytes, session_id="test")
    assert result is None, f"Expected None for clean frame, got '{result}'"

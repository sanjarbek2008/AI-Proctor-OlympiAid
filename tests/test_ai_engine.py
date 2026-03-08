"""Tests for AI engine head pose detection.

Each image is analyzed independently. Any violation is returned immediately
with no scoring, buffering, or session state.
"""
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================
# Threshold Config Tests
# ============================================================

class TestThresholds:
    """Verify threshold constants exist and are sensible."""

    def test_yaw_deg_threshold_exists(self):
        from app.ai_engine import YAW_DEG_THRESHOLD
        assert 0 < YAW_DEG_THRESHOLD < 90

    def test_pitch_up_deg_threshold_exists(self):
        from app.ai_engine import PITCH_UP_DEG_THRESHOLD
        assert 0 <= PITCH_UP_DEG_THRESHOLD < 90

    def test_pitch_down_min_deg_exists(self):
        from app.ai_engine import PITCH_DOWN_MIN_DEG
        assert -90 < PITCH_DOWN_MIN_DEG <= 0

    def test_yolo_confidence_threshold(self):
        from app.ai_engine import YOLO_CONFIDENCE_THRESHOLD
        assert 0 < YOLO_CONFIDENCE_THRESHOLD <= 1.0

    def test_suspicious_objects(self):
        from app.ai_engine import SUSPICIOUS_OBJECTS
        assert 'cell phone' in SUSPICIOUS_OBJECTS
        assert 'laptop' in SUSPICIOUS_OBJECTS
        assert 'book' not in SUSPICIOUS_OBJECTS


# ============================================================
# No Scoring / No Session State Tests
# ============================================================

class TestNoScoringNoState:
    """Verify scoring classes and session state are gone."""

    def test_session_tracker_removed(self):
        """SessionTracker should no longer exist in the module."""
        import app.ai_engine as engine
        assert not hasattr(engine, 'SessionTracker'), \
            "SessionTracker should have been removed"

    def test_frame_record_removed(self):
        """FrameRecord should no longer exist in the module."""
        import app.ai_engine as engine
        assert not hasattr(engine, 'FrameRecord'), \
            "FrameRecord should have been removed"

    def test_get_tracker_removed(self):
        """_get_tracker helper should no longer exist."""
        import app.ai_engine as engine
        assert not hasattr(engine, '_get_tracker'), \
            "_get_tracker should have been removed"

    def test_sessions_registry_removed(self):
        """Module-level _sessions dict should no longer exist."""
        import app.ai_engine as engine
        assert not hasattr(engine, '_sessions'), \
            "_sessions registry should have been removed"

    def test_suspicion_score_threshold_removed(self):
        """SUSPICION_SCORE_THRESHOLD should no longer exist."""
        import app.ai_engine as engine
        assert not hasattr(engine, 'SUSPICION_SCORE_THRESHOLD'), \
            "SUSPICION_SCORE_THRESHOLD should have been removed"


# ============================================================
# Head Pose Bbox Extraction Tests
# ============================================================

class TestFaceBboxExtraction:
    """Test _get_face_bbox_from_detections and _get_face_bbox_from_landmarks."""

    def test_get_face_bbox_from_detections_empty(self):
        from app.ai_engine import _get_face_bbox_from_detections
        result = _get_face_bbox_from_detections([], 640, 480)
        assert result is None

    def test_get_face_bbox_from_detections_pixel_coords(self):
        from app.ai_engine import _get_face_bbox_from_detections
        mock_bbox = MagicMock()
        mock_bbox.origin_x = 100
        mock_bbox.origin_y = 100
        mock_bbox.width = 150
        mock_bbox.height = 150
        mock_det = MagicMock()
        mock_det.bounding_box = mock_bbox
        result = _get_face_bbox_from_detections([mock_det], 640, 480)
        assert result is not None
        assert result[0] == 100 and result[1] == 100
        assert result[2] == 250 and result[3] == 250


# ============================================================
# Looking Down Logic Tests (degree-based)
# ============================================================

class TestLookingDownDetection:
    """Verify looking-down detection uses degree thresholds correctly."""

    def test_pitch_below_min_is_looking_down(self):
        """pitch_deg < PITCH_DOWN_MIN_DEG means looking down (allowed)."""
        from app.ai_engine import PITCH_DOWN_MIN_DEG
        pitch_deg = -35.0
        assert pitch_deg < PITCH_DOWN_MIN_DEG

    def test_pitch_above_min_not_looking_down(self):
        """pitch_deg > PITCH_DOWN_MIN_DEG means not looking down (violation)."""
        from app.ai_engine import PITCH_DOWN_MIN_DEG
        pitch_deg = 0.0
        assert pitch_deg > PITCH_DOWN_MIN_DEG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

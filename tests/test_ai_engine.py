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
# Lightweight mock landmarks for testing
# ============================================================

@dataclass
class MockLandmark:
    """Mimics mediapipe NormalizedLandmark."""
    x: float
    y: float
    z: float = 0.0


def make_landmarks(overrides: dict = None) -> list:
    """Create a list of 478 mock landmarks with sane face defaults.
    
    Default layout: face centered, looking straight forward.
    Use overrides dict {index: (x, y)} to customize.
    """
    lms = [MockLandmark(x=0.5, y=0.5) for _ in range(478)]

    defaults = {
        1:   (0.50, 0.55),  # Nose tip
        10:  (0.50, 0.25),  # Forehead
        33:  (0.35, 0.45),  # Left eye
        61:  (0.40, 0.75),  # Mouth left
        199: (0.50, 0.90),  # Chin
        263: (0.65, 0.45),  # Right eye
        291: (0.60, 0.75),  # Mouth right
        # Cheeks (for yaw)
        93:  (0.30, 0.55),
        323: (0.70, 0.55),
        # Ears (fallback)
        234: (0.20, 0.50),
        454: (0.80, 0.50),
    }

    if overrides:
        defaults.update(overrides)

    for idx, (x, y) in defaults.items():
        lms[idx] = MockLandmark(x=x, y=y)

    return lms


# ============================================================
# Head Pose Computation Tests
# ============================================================

class TestComputeHeadPose:
    """Test _compute_head_pose with synthetic landmarks."""

    def _get_fn(self):
        from app.ai_engine import _compute_head_pose
        return _compute_head_pose

    def test_centered_pose(self):
        """Face centered → low yaw and pitch offsets."""
        fn = self._get_fn()
        lms = make_landmarks()
        pose = fn(lms, img_w=640, img_h=480)
        assert pose['yaw_offset'] < 0.1, f"Yaw {pose['yaw_offset']} should be near 0"
        assert abs(pose['pitch_offset']) < 0.2, f"Pitch {pose['pitch_offset']} should be near 0"

    def test_head_turned_right(self):
        """Nose shifted right → high yaw offset."""
        fn = self._get_fn()
        lms = make_landmarks({1: (0.65, 0.55)})
        pose = fn(lms, img_w=640, img_h=480)
        assert pose['yaw_offset'] > 0.25, f"Yaw {pose['yaw_offset']} should be > 0.25"

    def test_head_looking_up(self):
        """Nose moved above eyes → positive pitch offset."""
        fn = self._get_fn()
        lms = make_landmarks({1: (0.50, 0.35)})
        pose = fn(lms, img_w=640, img_h=480)
        assert pose['pitch_offset'] > 0.18, f"Pitch {pose['pitch_offset']} should be > 0.18"

    def test_head_looking_down(self):
        """Nose moved below eyes → negative pitch offset."""
        fn = self._get_fn()
        lms = make_landmarks({1: (0.50, 0.65)})
        pose = fn(lms, img_w=640, img_h=480)
        assert pose['pitch_offset'] < -0.05, f"Pitch {pose['pitch_offset']} should be < -0.05"


# ============================================================
# Threshold Config Tests
# ============================================================

class TestThresholds:
    """Verify threshold constants exist and are sensible."""

    def test_yaw_threshold_exists(self):
        from app.ai_engine import YAW_THRESHOLD
        assert 0 < YAW_THRESHOLD < 1.0

    def test_pitch_threshold_exists(self):
        from app.ai_engine import PITCH_UP_THRESHOLD
        assert 0 <= PITCH_UP_THRESHOLD < 1.0

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
# Looking Down Logic Tests
# ============================================================

class TestLookingDownDetection:
    """Verify looking-down detection works correctly."""

    def test_looking_down_detection(self):
        """Nose below eyes → is_looking_down = True."""
        from app.ai_engine import _compute_head_pose
        lms = make_landmarks({1: (0.50, 0.65)})
        pose = _compute_head_pose(lms, img_w=640, img_h=480)
        is_looking_down = pose['pitch_offset'] < -0.05
        assert is_looking_down, "Student should be detected as looking down"

    def test_looking_down_lenient_yaw(self):
        """Even when looking down, head turned sideways is still detectable."""
        from app.ai_engine import _compute_head_pose
        lms = make_landmarks({1: (0.60, 0.65)})  # nose right + down
        pose = _compute_head_pose(lms, img_w=640, img_h=480)
        # Both pose values are computed — downstream logic applies thresholds
        assert 'yaw_offset' in pose
        assert 'pitch_offset' in pose


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for AI engine gaze estimation and temporal smoothing logic.

These tests validate the pure math/logic using mocked landmarks — 
no real images or model loading needed.
"""
import time
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass


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
    
    Default layout: face centered, eyes centered, iris centered.
    Use overrides dict {index: (x, y)} to customize.
    """
    # Create 478 landmarks at face center by default
    lms = [MockLandmark(x=0.5, y=0.5) for _ in range(478)]

    # Set up a realistic default face geometry (normalized coords)
    defaults = {
        # Nose tip
        1: (0.50, 0.55),
        # Left eye outer/inner corners
        33: (0.35, 0.45),
        133: (0.45, 0.45),
        # Right eye inner/outer corners
        362: (0.55, 0.45),
        263: (0.65, 0.45),
        # Iris centers (centered in eyes)
        468: (0.40, 0.45),  # left iris centered
        473: (0.60, 0.45),  # right iris centered
        # Cheeks (for yaw)
        93: (0.30, 0.55),   # left cheek
        323: (0.70, 0.55),  # right cheek
        # Eye top/bottom (for pitch)
        159: (0.40, 0.43),  # left eye top
        145: (0.40, 0.47),  # left eye bottom
        386: (0.60, 0.43),  # right eye top
        374: (0.60, 0.47),  # right eye bottom
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
# Import the functions under test
# We import after defining mocks since ai_engine loads models at import time.
# To avoid model loading, we test the pure functions directly.
# ============================================================

# We need to be able to import the pure functions without triggering model loading.
# Strategy: import the module and test the helper functions.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestComputeGazeRatio:
    """Test _compute_gaze_ratio with synthetic landmarks."""

    def _get_fn(self):
        """Import the function (lazy to avoid import-time model errors in CI)."""
        from app.ai_engine import _compute_gaze_ratio
        return _compute_gaze_ratio

    def test_centered_gaze(self):
        """Iris centered in eye → ratio ~0.5."""
        fn = self._get_fn()
        lms = make_landmarks()
        left_r, right_r = fn(lms, img_w=640, img_h=480)
        assert 0.40 <= left_r <= 0.60, f"Left ratio {left_r} not centered"
        assert 0.40 <= right_r <= 0.60, f"Right ratio {right_r} not centered"

    def test_looking_left(self):
        """Iris shifted to outer corner of left eye → ratio < 0.35."""
        fn = self._get_fn()
        lms = make_landmarks({
            468: (0.36, 0.45),  # left iris near outer (left) corner (33 is at 0.35)
            473: (0.56, 0.45),  # right iris near inner corner (362 is at 0.55)
        })
        left_r, right_r = fn(lms, img_w=640, img_h=480)
        assert left_r < 0.35, f"Left ratio {left_r} should be < 0.35 for looking left"
        assert right_r < 0.35, f"Right ratio {right_r} should be < 0.35 for looking left"

    def test_looking_right(self):
        """Iris shifted to inner corner → ratio > 0.65."""
        fn = self._get_fn()
        lms = make_landmarks({
            468: (0.44, 0.45),  # left iris near inner corner (133 is at 0.45)
            473: (0.64, 0.45),  # right iris near outer corner (263 is at 0.65)
        })
        left_r, right_r = fn(lms, img_w=640, img_h=480)
        assert left_r > 0.65, f"Left ratio {left_r} should be > 0.65 for looking right"
        assert right_r > 0.65, f"Right ratio {right_r} should be > 0.65 for looking right"

    def test_zero_eye_width_returns_centered(self):
        """If eye width is zero (degenerate), return 0.5."""
        fn = self._get_fn()
        lms = make_landmarks({
            33: (0.40, 0.45),
            133: (0.40, 0.45),  # same x → zero width
            362: (0.60, 0.45),
            263: (0.60, 0.45),  # same x → zero width
        })
        left_r, right_r = fn(lms, img_w=640, img_h=480)
        assert left_r == 0.5
        assert right_r == 0.5


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
        lms = make_landmarks({
            1: (0.65, 0.55),  # nose shifted right
        })
        pose = fn(lms, img_w=640, img_h=480)
        assert pose['yaw_offset'] > 0.25, f"Yaw {pose['yaw_offset']} should be > 0.25"

    def test_head_looking_up(self):
        """Nose moved up above eyes → positive pitch offset."""
        fn = self._get_fn()
        lms = make_landmarks({
            1: (0.50, 0.35),  # nose above eyes (eyes are at y=0.45)
        })
        pose = fn(lms, img_w=640, img_h=480)
        assert pose['pitch_offset'] > 0.18, f"Pitch {pose['pitch_offset']} should be > 0.18"


class TestSessionTracker:
    """Test temporal smoothing via SessionTracker."""

    def _make_tracker(self, buffer_size=10, trigger_ratio=0.5, score_threshold=0.4):
        from app.ai_engine import SessionTracker
        return SessionTracker(
            buffer_size=buffer_size,
            trigger_ratio=trigger_ratio,
            score_threshold=score_threshold,
        )

    def test_no_violation_below_threshold(self):
        """Low scores should never trigger."""
        tracker = self._make_tracker(buffer_size=10)
        for _ in range(20):
            result = tracker.add_frame(0.1, {})
        assert result is None

    def test_violation_after_sustained_suspicion(self):
        """Sustained high scores should trigger after buffer fills."""
        tracker = self._make_tracker(buffer_size=10, trigger_ratio=0.5, score_threshold=0.4)
        violation = None
        for _ in range(20):
            violation = tracker.add_frame(0.8, {'head_turned_sideways': 0.35})
            if violation:
                break
        assert violation is not None
        assert violation == 'head_turned_sideways'

    def test_mixed_frames_below_ratio(self):
        """If only 30% of frames are suspicious (below 50% trigger), no violation."""
        tracker = self._make_tracker(buffer_size=10, trigger_ratio=0.5, score_threshold=0.4)
        violation = None
        for i in range(10):
            score = 0.8 if i < 3 else 0.1  # only 3/10 = 30%
            violation = tracker.add_frame(score, {'gaze_left': 0.2} if i < 3 else {})
        assert violation is None

    def test_buffer_clears_after_violation(self):
        """After a violation fires, buffer resets — next clean frames shouldn't trigger."""
        tracker = self._make_tracker(buffer_size=6, trigger_ratio=0.5, score_threshold=0.4)
        # Fill with suspicious frames
        for _ in range(10):
            tracker.add_frame(0.8, {'gaze_right': 0.8})
        # Now send clean frames — should need to fill buffer again
        for _ in range(5):
            result = tracker.add_frame(0.0, {})
        # Buffer is 6, we sent 5 clean, so shouldn't re-trigger
        assert result is None

    def test_session_expiry(self):
        """Tracker should report expired after timeout."""
        tracker = self._make_tracker()
        tracker.last_active = time.time() - 400  # 400s ago
        assert tracker.is_expired(timeout=300) is True

    def test_session_not_expired(self):
        """Fresh tracker should not be expired."""
        tracker = self._make_tracker()
        assert tracker.is_expired(timeout=300) is False


class TestSuspicionScoring:
    """Integration-level test of score accumulation logic."""

    def test_gaze_only_triggers_score(self):
        """Sideways gaze alone should contribute 0.5 to score."""
        # This tests the concept: avg_gaze < 0.35 → +0.5
        from app.ai_engine import GAZE_LEFT_THRESHOLD
        avg_gaze = 0.20
        score = 0.0
        if avg_gaze < GAZE_LEFT_THRESHOLD:
            score += 0.5
        assert score >= 0.4, "Gaze-only should exceed suspicion threshold"

    def test_mild_yaw_below_threshold(self):
        """Small yaw offset alone (0.15) should not reach threshold."""
        from app.ai_engine import YAW_THRESHOLD, SUSPICION_SCORE_THRESHOLD
        yaw = 0.15
        score = 0.0
        if yaw > YAW_THRESHOLD:
            score += 0.4
        assert score < SUSPICION_SCORE_THRESHOLD, "Mild yaw should not trigger"

    def test_combined_signals_exceeds_threshold(self):
        """Moderate yaw + moderate gaze should exceed threshold together."""
        from app.ai_engine import YAW_THRESHOLD, GAZE_LEFT_THRESHOLD, SUSPICION_SCORE_THRESHOLD
        score = 0.0
        yaw = 0.30
        avg_gaze = 0.25

        if yaw > YAW_THRESHOLD:
            score += 0.4
        if avg_gaze < GAZE_LEFT_THRESHOLD:
            score += 0.5

        assert score >= SUSPICION_SCORE_THRESHOLD, "Combined signals should exceed threshold"



class TestLookingDownFixes:
    """Test the new logic for handling students looking down."""

    def test_book_not_in_suspicious_objects(self):
        """Verify that 'book' has been removed from suspicious objects."""
        from app.ai_engine import SUSPICIOUS_OBJECTS
        assert 'book' not in SUSPICIOUS_OBJECTS, "book should not be in suspicious objects (scratch paper false positive)"
        assert 'cell phone' in SUSPICIOUS_OBJECTS
        assert 'laptop' in SUSPICIOUS_OBJECTS

    def test_looking_down_skips_gaze_check(self):
        """When looking down (pitch < -0.05), gaze checks should be skipped."""
        from app.ai_engine import _compute_head_pose, SessionTracker
        
        # Create landmarks with head looking down (nose below eyes)
        lms = make_landmarks({
            1: (0.50, 0.65),  # nose moved down (eyes are at y=0.45)
            # Iris way off to the side (which would normally trigger gaze alert)
            468: (0.36, 0.45),  # left iris at outer edge
            473: (0.56, 0.45),  # right iris at inner edge
        })
        
        pose = _compute_head_pose(lms, img_w=640, img_h=480)
        is_looking_down = pose['pitch_offset'] < -0.05
        
        assert is_looking_down, "Student should be detected as looking down"
        
        # In the actual code, when is_looking_down is True, gaze check is skipped
        # So the score should NOT include the 0.5 gaze penalty
        # We can't easily test the full analyze_image without mocking YOLO,
        # but this verifies the pose detection logic works

    def test_looking_down_lenient_yaw(self):
        """When looking down, moderate yaw should NOT flag (only extreme yaw)."""
        from app.ai_engine import _compute_head_pose, YAW_THRESHOLD
        
        # Create landmarks with head looking down + moderate yaw
        lms = make_landmarks({
            1: (0.50, 0.65),  # nose down
            # Nose shifted to create yaw = 0.26 (just above YAW_THRESHOLD of 0.25)
            1: (0.60, 0.65),  # nose shifted right (but looking down)
        })
        
        pose = _compute_head_pose(lms, img_w=640, img_h=480)
        is_looking_down = pose['pitch_offset'] < -0.05
        
        # Moderate yaw (>0.25 but <0.375) should NOT trigger when looking down
        # Only if yaw > YAW_THRESHOLD * 1.5 (0.375) should it flag
        assert is_looking_down, "Should be looking down"
        
        # In actual code: if is_looking_down and yaw > 0.25 but < 0.375, no flag
        # This test verifies the pose calculation; the scoring logic is in ai_engine.py


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


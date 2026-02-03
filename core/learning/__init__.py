"""Adaptive learning system for Alfred."""

from .feedback import FeedbackTracker, record_feedback, get_feedback_stats
from .preferences import PreferenceEngine, update_preference, get_preferences
from .patterns import PatternDetector, detect_patterns, get_workflow_suggestions

__all__ = [
    "FeedbackTracker", "record_feedback", "get_feedback_stats",
    "PreferenceEngine", "update_preference", "get_preferences",
    "PatternDetector", "detect_patterns", "get_workflow_suggestions",
]

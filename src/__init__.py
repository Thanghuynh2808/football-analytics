"""
Football Analytics - AI-powered football video analysis system

This package provides tools for:
- Player detection using YOLO
- Team classification based on jersey colors
- Video processing and analysis
"""

__version__ = "1.0.0"
__author__ = "Football Analytics Team"
__email__ = "your.email@example.com"

from .detection.yolo_infer import YOLOInference
from .teams.team_classifier import FootballTeamClassifier
from .teams.team_classifier_runner import TeamClassifierRunner

__all__ = [
    "YOLOInference",
    "FootballTeamClassifier", 
    "TeamClassifierRunner",
]
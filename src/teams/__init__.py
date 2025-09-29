"""
Team classification module for football analytics

Contains functionality for classifying players into teams based on jersey colors.
"""

from .team_classifier import FootballTeamClassifier
from .team_classifier_runner import TeamClassifierRunner

__all__ = ["FootballTeamClassifier", "TeamClassifierRunner"]
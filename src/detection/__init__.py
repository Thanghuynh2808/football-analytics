"""
Detection module for football analytics

Contains YOLO-based player detection functionality.
"""

from .yolo_infer import YOLOInference

__all__ = ["YOLOInference"]
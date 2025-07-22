"""Standardized feature types for TioMagic
Defines all supported video generation features
"""
from enum import Enum
from typing import List

class FeatureType(str, Enum):
    """Standardized feature types for video generation"""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    INTERPOLATE = "interpolate"
    POSE_GUIDANCE = "pose_guidance"

# List of all supported feature types
SUPPORTED_FEATURE_TYPES: List[str] = [
    FeatureType.TEXT_TO_VIDEO,
    FeatureType.IMAGE_TO_VIDEO,
    FeatureType.INTERPOLATE,
    FeatureType.POSE_GUIDANCE,
]

def is_valid_feature_type(feature_type: str) -> bool:
    """Check if a feature type is valid"""
    return feature_type in SUPPORTED_FEATURE_TYPES

def get_feature_types() -> List[str]:
    """Get list of all supported feature types"""
    return SUPPORTED_FEATURE_TYPES.copy()
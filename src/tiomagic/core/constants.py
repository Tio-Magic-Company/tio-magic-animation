"""Standardized feature types for TioMagic
Defines all supported video generation features
"""
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, asdict

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

"""MODAL VIDEO GENERATION TRACKING"""
@dataclass
class Generation:
    call_id: Optional[str] = None
    status: Optional[str] = None
    message: str = ''
    timestamp: Optional[str] = None
    required_args: Optional[dict] = None
    optional_args: Optional[dict] = None
    result_video: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# distributes to appropriate Modal file

# t2v accepts: wan2.1, wan2.1_1.3B, wan2.1_14B
# i2v accepts: wan2.1, wan2.1_14B_480P, wan2.1_14B_720P

from typing import Dict, Optional, Type, List, Any, Set
from ..core import Feature, JobStatus
from .base import AbstractAdapter
from ..registry import registry
from ..providers.modal.wan_2_1 import app as wan21_14b_t2v

@registry.register(
    "wan2.1",
    "wan2.1-t2v-14b",
    description="Wan 2.1 Text-to-Video 14b model hosted on Modal"
)
class Wan21Adapter(AbstractAdapter):
    supported_features: Set[Feature] = {Feature.TEXT_TO_VIDEO, Feature.IMAGE_TO_VIDEO}
    def launch(self, feature: Feature, **optional_args: Any):
        # check if feature is supported
        # check if specified pipeline

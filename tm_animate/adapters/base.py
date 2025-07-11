# The contract that every model must satisfy.
# To add support to a new model, create a new class that inherits from
# base.py and implements all of its abstract methods

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Callable, Set
from ..core import JobStatus, Feature

class AbstractAdapter(ABC):
    """
    Abstract Base Class for all model adapters
    Attributes:
        supported_features: A set of 'Feature' enums that this model can handle
        model_info: Optional parameters that this model supports
    """

    supported_features: Set[Feature] = set()
    def is_supported(self, feature: Feature) -> bool:
        return feature in self.supported_features
    
    # Core Interface: These methods MUST be implemented by every model
    @abstractmethod
    def launch(self, feature: Feature, **optional_args: Any) -> str:
        # submits job to the provider and returns provider's unique job id
        pass
    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        # fetches job's status from the provider and maps it to canonical Job Status
        pass
    @abstractmethod
    def result_url(self, job_id: str) -> Optional[str]:
        # returns public download URL for artifact, if job is complete
        pass
    @abstractmethod
    def download(self, job_id: str, destination_path: str) -> None:
        # download artifact to local file path
        pass

    # optional interface
    def cancel(self, job_id: str) -> bool:
        # cancel a job if provider's API supports it. Default is not supported
        return False
    
    def estimate_cost(self, job_id: str, **optional_args: Any) -> float:
        # estimate job's cost in USD. Default is unknown (0.0)
        return 0.0
import time
from enum import Enum, auto
from typing import Any, Optional, Dict, List, Callable
from pathlib import Path
from .exceptions import *
from .adapters import *
from .providers import *
from .registry import registry

# public classes
class Feature(Enum):
    # type safe way to represent different kinds of generation tasks
    TEXT_TO_VIDEO = auto()
    IMAGE_TO_VIDEO = auto()
    IN_BETWEENS = auto()
    POSE_GUIDANCE = auto()
    LAYOUT_CONTROL = auto()


class JobStatus(Enum):
    # every model must map its provider's statuses to one of these, ensure consistent experience
    QUEUED = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELED = auto()

class Job:
    # delegation class
    def __init__(self, job_id: str, model: "AbstractAdapter", feature: Feature, **kwargs: Any) -> None:
        self.id = job_id
        self._model = model
        self.feature = feature
        self._optional_args = kwargs
        self._start_time = time.monotonic()
    def status(self) -> JobStatus:
        # fetch current status of the job from the provider
        # model.status
        return self._model.status(self.id)
    def result_url(self) -> Optional[str]:
        # get remote URL of generated artifact if available, otherwise returns None
        return self._model.result_url(self.id)
    def download(self, destination_path: str | Path) -> None:
        # download result of completed job to local file
        self._model.download(self.id, str(destination_path))
    def cancel(self) -> bool:
        # attempt to cancel job. Returns True on success. Not all providers support cancellation
        return self._model.cancel(self.id)
    
    # potentially a wait function, blocks execution until job is complete, fails, or timeout is reached
    def elapsed_time(self) -> float:
        # number of seconds that have passed since job was created
        return time.monotonic() - self._start_time
    def estimate_cost(self) -> float:
        # returns estimated cost of a job in USD, if model supports it
        return self._model.estimate_cost(self.id, **self._optional_args)
    def __repr__(self) -> str:
        return f"Job id={self.id}, feature={self.feature.name}, model={self._model.name}"
    
class TioMagic:
    # set provider (local, modal, baseten)
    def __init__(self, provider: str = "modal", **provider_kwargs):
        self.provider = self._create_provider(provider, **provider_kwargs)

    def _create_provider(self, provider_name: str, **kwargs) -> Provider:
        # factory method to create provider instances
        providers = {
            "modal": ModalProvider,
        }

        if provider_name not in providers:
            raise UnknownProviderError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
        
        return providers[provider_name](**kwargs)
    
    # organized by feature
    def text_to_video(self, prompt: str, model: str, **kwargs: Any) -> Job:
        return self._run(Feature.TEXT_TO_VIDEO, model, prompt=prompt, **kwargs)
    # list models, get model info

    def _run(self, feature: Feature, model: str, **optional_arguments: Any) -> Job:
        # find theh model, validate, and launch the job
        model_class = registry.get_model_class(model)
        if not model_class:
            raise UnknownModelError(f"Model '{model}' is not registered")
        mmodel = model_class() # create Model Class, check if authentication works

        # check if feature is supported in model
        if not mmodel.is_supported(feature):
            supported = [m.name for m in mmodel.supported_features]
            raise ValidationError(
                f"Model '{model}' (via {model_class.__name__}) does not support {mmodel.name}. "
                f"It only supports: {supported}"
            )
        
        job_id = mmodel.launch(feature, **optional_arguments)
        return Job(job_id, mmodel, feature, **optional_arguments)
    

tm = TioMagic()

def configure(provider: str = "modal", **provider_kwargs) -> None:
    # configure global client
    global _default_client
    _default_client = tm(provider, **provider_kwargs)
    
    print(f"configure provider: {provider}, provider kwargs: {provider_kwargs}")
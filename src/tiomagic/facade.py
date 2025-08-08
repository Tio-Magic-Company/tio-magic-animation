"""TioMagic - A unified interface for video generation across multiple providers.

This module provides a high-level API for various video generation tasks including
text-to-video, image-to-video, interpolation, and pose guidance. It supports multiple
backend providers (local, modal, baseten) and handles job management, validation,
and status tracking.

Example:
    Basic usage::

        from tiomagic import tm
        
        # Configure provider
        tm.configure(provider="modal", api_key="your-api-key")
        
        # Generate video from text
        job = tm.text_to_video(
            model="cogvideox",
            required_args={"prompt": "A cat playing piano"}
        )
        
        # Check status
        tm.check_generation_status(job.job_id)

Attributes:
    tm: Pre-initialized TioMagic instance for convenience.
"""
from typing import Any, Dict
from uuid import uuid4

from .core.schemas import FEATURE_SCHEMAS
from .core.registry import registry
from .core.config import Configuration
from .core._jobs import Job, JobStatus
from .core._validation import validate_parameters
from .core._utils import create_timestamp
from .core.errors import (
    UnknownModelError, UnknownProviderError, ValidationError,
    JobExecutionError, ResourceNotFoundError,
)

class TioMagic:
    """Main interface for video generation operations.
    
    This class provides methods for various video generation tasks and manages
    provider configuration, job tracking, and error handling. It supports multiple
    backend providers and models through a registry system.
    
    Attributes:
        _config: Configuration instance managing provider settings and API keys.
    """
    def __init__(self):
        """Initialize TioMagic with default configuration."""
        self._config = Configuration()

    def configure(self, provider=None, api_key=None, model_path=None, gpu=None, timeout=None, scaledown_window=None, flow_shift=None):
        """Configure the video generation provider and credentials.
        
        Args:
            provider (str, optional): Provider name ('local', 'modal', 'baseten').
            api_key (str, optional): API key for the specified provider.
            model_path (str, optional): Path to local model files (for local provider).
            gpu (GPUType, optional): GPU type to run on modal.
            timeout (int, optional): Measure of execution time on modal.
            scaledown_window (int, optional): How long container is warm for after completion of a task modal. Inputs will execute more quickly on a warm container compared to a cold one
            flow_shift (float, optional): Set flow shift for models that accept flow shift

        Example:
            >>> tm.configure(provider="modal", api_key="sk-...")
            >>> tm.configure(provider="local", model_path="/path/to/model")
        """
        if provider:
            self._config.set_provider(provider)
        if api_key:
            self._config.set_api_key(provider, api_key)
        if model_path:
            self._config.set_model_path(provider, model_path)
        if provider == 'modal' and (gpu or timeout or scaledown_window):
            self._config.set_modal_options(gpu, timeout, scaledown_window)

    def text_to_video(self, model=None, required_args: Dict[str, Any]= None, **kwargs):
        """Generate video from text prompt.
        
        Creates a video generation job that converts text descriptions into video content
        using the specified model and provider.
        
        Args:
            model (str, optional): Model identifier (e.g., 'cogvideox', 'stable-video').
                If None, uses provider's default model.
            required_args (Dict[str, Any], optional): Required parameters including:
                - prompt (str): Text description of the desired video.
                Additional model-specific parameters may be required.
            **kwargs: Additional optional parameters specific to the model/provider.
            
        Returns:
            Job: Job object for tracking generation progress and retrieving results.
        """
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("text_to_video", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("--> Implementation: ", implementation)

        valid, params, error_msg = validate_parameters("text_to_video", model, required_args, kwargs)
        if not valid:
            # Parse error message to provide more specific validation error
            field = "unknown"
            if "prompt" in error_msg:
                field = "prompt"
            elif "model" in error_msg:
                field = "model"
            
            raise ValidationError(
                field=field,
                message=error_msg,
                value=required_args.get(field) if required_args else None
            )
        
        # if modal provider, add modal_options into kwargs
        print("PROVIDER", provider)
        if provider == 'modal':
            print("PROVIDER MODAL, SET OPTIONS")

            kwargs['modal_options'] = self._config.get_modal_options()

        print("--> Validated parameters: ", params)

        # Start job and return job object for tracking
        job_id = str(uuid4())
        job = Job(
            job_id=job_id, 
            feature="text_to_video", 
            model=model, 
            provider=provider,
        )
        job.save()
        try:

            print(f"--> Text to video create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(required_args, **kwargs))
            print(f"--> Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            raise JobExecutionError(
                job_id=job_id,
                reason=f"Failed to start text-to-video generation: {str(e)}",
                provider_error=str(e)
            )

        return job

    def image_to_video(self, model=None, required_args: Dict[str, Any]= None, **kwargs):
        """Generate video from an input image.
        
        Creates a video generation job that animates or extends a static image into
        video content using the specified model and provider.
        
        Args:
            model (str, optional): Model identifier for image-to-video generation.
                If None, uses provider's default model.
            required_args (Dict[str, Any], optional): Required parameters including:
                - image (str): Path or URL to the input image.
                - prompt (str, optional): Text description to guide video generation.
                Additional model-specific parameters may be required.
            **kwargs: Additional optional parameters specific to the model/provider.
            
        Returns:
            Job: Job object for tracking generation progress and retrieving results.
        """
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("image_to_video", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("--> Implementation: ", implementation)

        valid, params, error_msg = validate_parameters("image_to_video", model, required_args, kwargs)
        if not valid:
            # Parse error message to provide more specific validation error
            field = "unknown"
            if "prompt" in error_msg:
                field = "prompt"
            elif "model" in error_msg:
                field = "model"
            
            raise ValidationError(
                field=field,
                message=error_msg,
                value=required_args.get(field) if required_args else None
            )
        # if modal provider, add modal_options into kwargs
        if provider == 'modal':
            kwargs['modal_options'] = self._config.get_modal_options()

        print("--> Validated parameters: ", params)
        

        job_id = str(uuid4())
        job = Job(
            job_id=job_id,
            feature="image_to_video",
            model=model,
            provider=provider
        )
        job.save()
        try:
            print(f"--> Image to video create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(required_args, **kwargs))
            print(f"--> Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            raise JobExecutionError(
                job_id=job_id,
                reason=f"Failed to start image-to-video generation: {str(e)}",
                provider_error=str(e)
            )

        return job

    def interpolate(self, model=None, required_args: Dict[str, Any] = None, **kwargs):
        """Generate interpolated video between keyframes or images.
        
        Creates a video generation job that interpolates between multiple input frames
        to create smooth transitions or in-between frames.
        
        Args:
            model (str, optional): Model identifier for interpolation.
                If None, uses provider's default interpolation model.
            required_args (Dict[str, Any], optional): Required parameters including:
                - frames (List[str]): List of paths/URLs to keyframe images.
                - prompt (str, optional): Text description to guide interpolation.
                Additional model-specific parameters may be required.
            **kwargs: Additional optional parameters specific to the model/provider.
            
        Returns:
            Job: Job object for tracking generation progress and retrieving results.
        """
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("interpolate", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("--> Implementation: ", implementation)

        valid, params, error_msg = validate_parameters("interpolate", model, required_args, kwargs)
        if not valid:
            # Parse error message to provide more specific validation error
            field = "unknown"
            if "prompt" in error_msg:
                field = "prompt"
            elif "model" in error_msg:
                field = "model"
            
            raise ValidationError(
                field=field,
                message=error_msg,
                value=required_args.get(field) if required_args else None
            )
        # if modal provider, add modal_options into kwargs
        if provider == 'modal':
            kwargs['modal_options'] = self._config.get_modal_options()

        print("--> Validated parameters: ", params)

        job_id = str(uuid4())
        job = Job(
            job_id=job_id,
            feature="interpolate",
            model=model,
            provider=provider
        )
        job.save()

        try:
            print(f"--> Interpolate create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(
                required_args,
                **kwargs))
            print(f"--> Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            raise JobExecutionError(
                job_id=job_id,
                reason=f"Failed to start interpolation generation: {str(e)}",
                provider_error=str(e)
            )

        return job

    def pose_guidance(self, model=None, required_args: Dict[str, Any] = None, **kwargs):
        """Generate video with pose-guided animation.
        
        Creates a video generation job that uses pose information (skeleton, keypoints)
        to guide character or object animation in the generated video.
        
        Args:
            model (str, optional): Model identifier for pose-guided generation.
                If None, uses provider's default pose guidance model.
            required_args (Dict[str, Any], optional): Required parameters including:
                - pose_data: Pose information (format depends on model).
                - prompt (str): Text description of the desired animation.
                - reference_image (str, optional): Reference image for appearance.
                Additional model-specific parameters may be required.
            **kwargs: Additional optional parameters specific to the model/provider.
            
        Returns:
            Job: Job object for tracking generation progress and retrieving results.
        """
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("pose_guidance", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("--> Implementation: ", implementation)

        valid, params, error_msg = validate_parameters("pose_guidance", model, required_args, kwargs)
        if not valid:
           # Parse error message to provide more specific validation error
            field = "unknown"
            if "prompt" in error_msg:
                field = "prompt"
            elif "model" in error_msg:
                field = "model"
            
            raise ValidationError(
                field=field,
                message=error_msg,
                value=required_args.get(field) if required_args else None
            )
        # if modal provider, add modal_options into kwargs
        if provider == 'modal':
            kwargs['modal_options'] = self._config.get_modal_options()

        print("--> Validated parameters: ", params)

        job_id = str(uuid4())
        job = Job(
            job_id=job_id,
            feature="pose_guidance",
            model=model,
            provider=provider
        )
        job.save()

        try:
            print(f"--> Pose guidance create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(
                required_args,
                **kwargs))
            print(f"--> Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            raise JobExecutionError(
                job_id=job_id,
                reason=f"Failed to start pose guidance generation: {str(e)}",
                provider_error=str(e)
            )

        return job
    def check_generation_status(self, job_id, returnJob = False):
        """Check the current status of a video generation job.
        
        Queries the provider for the current status of a running job and updates
        the local job record. This method can be called periodically to monitor
        long-running generation tasks.
        
        Args:
            job_id (str): Unique identifier of the job to check.
            
        Returns:
            None: Updates are made to the job record in place.
        """
        job = Job.get_job(job_id)
        if not job:
            raise ResourceNotFoundError(
                resource_type="Job",
                resource_id=job_id,
                location="job storage"
            )

        # Check current status
        print("--> Check job: ", job.job_id)

        if job.generation and job.generation["call_id"]:
            try:
                # assumption it is modal for now
                impl_class = registry.get_implementation(job.feature, job.model, job.provider)
                implementation = self._create_implementation(impl_class, job.provider)
                job.check_status(lambda: implementation.check_generation_status(job.generation))
                job.update(last_updated= create_timestamp())
                # checks generation status and updates generation_log.json
                if returnJob:
                    return job
                else:
                    return

            except (UnknownModelError, UnknownProviderError):
                raise
            except Exception as e:
                job.update(status=JobStatus.FAILED)
                job.save()
                raise JobExecutionError(
                    job_id=job_id,
                    reason=f"Failed to check generation status: {str(e)}",
                    provider_error=str(e)
                )
        else:
            # Job exists but has no generation info
            raise JobExecutionError(
                job_id=job_id,
                reason="Job exists but has no generation information to check"
            )
    
    def cancel_job(self, job_id):
        """Cancel an active video generation job.
        
        Attempts to cancel a running job with the provider. The success of cancellation
        depends on the provider's capabilities and the current state of the job.
        
        Args:
            job_id (str): Unique identifier of the job to cancel.
            
        Returns:
            None: Job status is updated to reflect cancellation.
        """
        job = Job.get_job(job_id)
        if not job:
            raise ResourceNotFoundError(
                resource_type="Job",
                resource_id=job_id,
                location="job storage"
            )
        print("--> Cancel job: ", job.job_id)
        if job.generation and job.generation["call_id"]:
            try:
                impl_class = registry.get_implementation(job.feature, job.model, job.provider)
                implementation = self._create_implementation(impl_class, job.provider)
                job.cancel_job(lambda: implementation.cancel_job(job.generation))
                job.update(last_updated= create_timestamp())
            except (UnknownModelError, UnknownProviderError):
                raise
            except Exception as e:
                job.update(status=JobStatus.FAILED)
                job.save()
                raise JobExecutionError(
                    job_id=job_id,
                    reason=f"Failed to cancel job: {str(e)}",
                    provider_error=str(e)
                )
        else:
            raise JobExecutionError(
                job_id=job_id,
                reason="Job has no active generation to cancel"
            )
        
    def get_providers(self):
        """List all providers available."""
        return registry.get_providers()
    def get_models(self, feature, provider):
        """List all models available for a provider and feature"""
        return registry.get_models(feature=feature, provider=provider)
    def get_schema(self, feature, model):
        """List schema for particular implementation"""
        schema = FEATURE_SCHEMAS[feature][model]
        return schema
    def list_implementations(self):
        """List all implementations available
        Models listed and sorted by provider or feature
        """
        providers = registry.get_providers()
        features = registry.get_features()
        for provider in providers:
            print(f"{provider}: ")
            for feature in features:
                feature_models = registry.get_models(feature=feature, provider=provider)
                if len(feature_models) > 0:
                    print(f"\t{feature}: ")
                    for model in feature_models:
                        print(f"\t\t{model}")

    
    def _create_implementation(self, impl_class, provider):
        """Create an implementation instance with appropriate config.
        
        Internal method to instantiate provider-specific implementations
        with the correct configuration parameters.
        """
        if provider == "local":
            # return impl_class(model_path=self._config.get_model_path())
            return impl_class()
        elif provider == "modal":
            return impl_class(api_key=self._config.get_api_key("modal"))
        elif provider == "baseten":
            return impl_class(api_key=self._config.get_api_key("baseten"))
        else:
            return impl_class()

tm = TioMagic()
"""Pre-initialized TioMagic instance for direct module usage.

This allows users to import and use TioMagic without explicit instantiation:

Example:
    >>> from tiomagic import tm
    >>> tm.configure(provider="modal", api_key="...")
    >>> job = tm.text_to_video(...)
"""
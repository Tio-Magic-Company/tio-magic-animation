import base64
from datetime import datetime
from io import BytesIO
import sys
from typing import Any, Dict
import PIL.Image
from uuid import uuid4
from .core.registry import registry
from .core.config import Configuration
from .core.jobs import Job, JobStatus
from .core.validation import validate_parameters

class TioMagic:
    def __init__(self):
        self._config = Configuration()

    def configure(self, provider=None, api_key=None, model_path=None):
        if provider:
            self._config.set_provider(provider)
        if api_key:
            self._config.set_api_key(provider, api_key)
        if model_path:
            self._config.set_model_path(provider, model_path)

    def text_to_video(self, model=None, required_args: Dict[str, Any]= None, **kwargs):
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("text_to_video", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("implementation: ", implementation)
        
        # if required_args is None or required_args['prompt'] is None:
        #     print(f"Argument 'prompt' is required for text to video generation")
        #     return
        valid, params, error_msg = validate_parameters("text_to_video", model, required_args, kwargs)
        if not valid:
            raise ValueError(f"Parameter validation failed: {error_msg}")
        
        print(f"Validated parameters: ", params)

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

            print(f"text to video create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(required_args, **kwargs))
            print(f"Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            print(f"Error starting generation: {e}")
        
        return job
    
    def image_to_video(self, model=None, required_args: Dict[str, Any]= None, **kwargs):
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("image_to_video", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("implementation: ", implementation)

        valid, params, error_msg = validate_parameters("image_to_video", model, required_args, kwargs)
        if not valid:
            raise ValueError(f"Parameter validation failed: {error_msg}")
        
        print(f"Validated parameters: ", params)

        job_id = str(uuid4())
        job = Job(
            job_id=job_id,
            feature="image_to_video",
            model=model,
            provider=provider
        )
        job.save()
        try:
            print(f"image to video create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(required_args, **kwargs))
            print(f"Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            print(f"Error starting generation: {e}")
        
        return job
    
    def interpolate(self, model=None, required_args: Dict[str, Any] = None, **kwargs):
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("interpolate", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("implementation: ", implementation)

        valid, params, error_msg = validate_parameters("interpolate", model, required_args, kwargs)
        if not valid:
            raise ValueError(f"Parameter validation failed: {error_msg}")
        
        print(f"Validated parameters: ", params)
        
        job_id = str(uuid4())
        job = Job(
            job_id=job_id,
            feature="interpolate",
            model=model,
            provider=provider
        )
        job.save()

        try:
            print(f"interpolate create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(
                required_args,
                **kwargs))
            print(f"Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            print(f"Error starting generation: {e}")
        
        return job

    def pose_guidance(self, model=None, required_args: Dict[str, Any] = None, **kwargs):
        provider = self._config.get_provider()
        impl_class = registry.get_implementation("pose_guidance", model, provider)
        implementation = self._create_implementation(impl_class, provider)
        print("implementation: ", implementation)

        valid, params, error_msg = validate_parameters("pose_guidance", model, required_args, kwargs)
        if not valid:
            raise ValueError(f"Parameter validation failed: {error_msg}")
        
        print(f"Validated parameters: ", params)

        job_id = str(uuid4())
        job = Job(
            job_id=job_id,
            feature="pose_guidance",
            model=model,
            provider=provider
        )
        job.save()

        try:
            print(f"pose guidance create new job with provider {provider} and model {model}")
            job.start(lambda: implementation.generate(
                required_args,
                **kwargs))
            print(f"Generation started! Job ID: {job_id}")
        except Exception as e:
            job.update(status=JobStatus.FAILED)
            print(f"Error starting generation: {e}")
        
        return job

    def check_generation_status(self, job_id):
        job = Job.get_job(job_id)
        if not job:
            print(f"Job {job_id} not found")
            sys.exit(1)
        
        # Check current status
        print("Check job: ", job.job_id)

        if job.generation and job.generation["call_id"]:
            try:
                # assumption it is modal for now
                impl_class = registry.get_implementation(job.feature, job.model, job.provider)
                implementation = self._create_implementation(impl_class, job.provider)
                job.check_status(lambda: implementation.check_generation_status(job.generation))
                job.update(last_updated= datetime.now().strftime("%Y%m%d_%H%M%S"))
                # checks generation status and updates generation_log.json

            except Exception as e:
                job.update(status=JobStatus.FAILED)
                print(f"Error checking generation status: {e}")
        job.save()

    def _create_implementation(self, impl_class, provider):
        """create an implementation instance with appropriate config"""
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
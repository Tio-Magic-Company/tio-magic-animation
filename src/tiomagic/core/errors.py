"""Exception hierarchy for TioMagic Animation API.

This module defines custom exceptions for handling various error scenarios
in the TioMagic animation system.
"""

from typing import Optional, Dict, Any


class TioMagicError(Exception):
    """Base exception for all TioMagic errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class UnknownModelError(TioMagicError):
    """Raised when a model string can't be found in the registry."""
    
    def __init__(self, model: str, available_models: Optional[list] = None):
        message = f"Unknown model: '{model}'"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        super().__init__(message, {"model": model, "available_models": available_models})
        self.model = model
        self.available_models = available_models


class UnknownProviderError(TioMagicError):
    """Raised when a provider string can't be found."""
    
    def __init__(self, provider: str, available_providers: Optional[list] = None):
        message = f"Unknown provider: '{provider}'"
        if available_providers:
            message += f". Available providers: {', '.join(available_providers)}"
        super().__init__(message, {"provider": provider, "available_providers": available_providers})
        self.provider = provider
        self.available_providers = available_providers


class JobTimeoutError(TioMagicError):
    """Raised when Job.wait() exceeds time limit before the job finishes."""
    
    def __init__(self, job_id: str, timeout_seconds: int):
        message = f"Job '{job_id}' timed out after {timeout_seconds} seconds"
        super().__init__(message, {"job_id": job_id, "timeout_seconds": timeout_seconds})
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds


class JobExecutionError(TioMagicError):
    """Raised when provider reports that job has explicitly failed."""
    
    def __init__(self, job_id: str, reason: str, provider_error: Optional[str] = None):
        message = f"Job '{job_id}' failed: {reason}"
        details = {"job_id": job_id, "reason": reason}
        if provider_error:
            details["provider_error"] = provider_error
        super().__init__(message, details)
        self.job_id = job_id
        self.reason = reason
        self.provider_error = provider_error


class AuthenticationError(TioMagicError):
    """Raised when there's a missing or invalid API key or token."""
    
    def __init__(self, provider: Optional[str] = None, message: Optional[str] = None):
        if message is None:
            message = "Authentication failed"
            if provider:
                message += f" for provider '{provider}'"
        super().__init__(message, {"provider": provider})
        self.provider = provider


class APIError(TioMagicError):
    """Raised for network-related issues, like non-200 responses or JSON parsing failures."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_body: Optional[str] = None, url: Optional[str] = None):
        details = {}
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate long responses
        if url:
            details["url"] = url
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body
        self.url = url


class DeploymentError(TioMagicError):
    """Raised when target infrastructure is unavailable (e.g., Modal app name is wrong)."""
    
    def __init__(self, service: str, reason: str, app_name: Optional[str] = None):
        message = f"Deployment error for {service}: {reason}"
        details = {"service": service, "reason": reason}
        if app_name:
            details["app_name"] = app_name
        super().__init__(message, details)
        self.service = service
        self.reason = reason
        self.app_name = app_name


class ValidationError(TioMagicError):
    """Raised when user provided invalid input parameters."""
    
    def __init__(self, field: str, message: str, value: Any = None, 
                 constraints: Optional[Dict[str, Any]] = None):
        full_message = f"Validation error for '{field}': {message}"
        details = {"field": field, "message": message}
        if value is not None:
            details["provided_value"] = value
        if constraints:
            details["constraints"] = constraints
        super().__init__(full_message, details)
        self.field = field
        self.value = value
        self.constraints = constraints


class RateLimitError(TioMagicError):
    """Raised when provider's rate limit has been exceeded."""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None, 
                 limit_type: Optional[str] = None):
        message = f"Rate limit exceeded for provider '{provider}'"
        details = {"provider": provider}
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            details["retry_after"] = retry_after
        if limit_type:
            details["limit_type"] = limit_type
        super().__init__(message, details)
        self.provider = provider
        self.retry_after = retry_after
        self.limit_type = limit_type


class ResourceNotFoundError(TioMagicError):
    """Raised when a requested resource (image, video, etc.) cannot be found."""
    
    def __init__(self, resource_type: str, resource_id: str, location: Optional[str] = None):
        message = f"{resource_type} '{resource_id}' not found"
        if location:
            message += f" at {location}"
        super().__init__(message, {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "location": location
        })
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.location = location


class ProcessingError(TioMagicError):
    """Raised when there's an error processing media (images, videos)."""
    
    def __init__(self, media_type: str, operation: str, reason: str, 
                 file_path: Optional[str] = None):
        message = f"Failed to {operation} {media_type}: {reason}"
        details = {
            "media_type": media_type,
            "operation": operation,
            "reason": reason
        }
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, details)
        self.media_type = media_type
        self.operation = operation
        self.reason = reason
        self.file_path = file_path

class ConfigurationError(TioMagicError):
    """Raised when there's a configuration issue."""
    
    def __init__(self, config_type: str, message: str, missing_field: Optional[str] = None):
        full_message = f"Configuration error for {config_type}: {message}"
        details = {"config_type": config_type}
        if missing_field:
            details["missing_field"] = missing_field
        super().__init__(full_message, details)
        self.config_type = config_type
        self.missing_field = missing_field


class ImplementationError(TioMagicError):
    """Raised when implementation cannot be instantiated or is missing required methods."""
    
    def __init__(self, provider: str, feature: str, reason: str, 
                 implementation_class: Optional[str] = None):
        message = f"Failed to create {feature} implementation for provider '{provider}': {reason}"
        details = {
            "provider": provider,
            "feature": feature,
            "reason": reason
        }
        if implementation_class:
            details["implementation_class"] = implementation_class
        super().__init__(message, details)
        self.provider = provider
        self.feature = feature
        self.reason = reason

class GenerationError(TioMagicError):
    """Raised when the AI model generation process fails.
    
    This is distinct from JobExecutionError which is about job infrastructure.
    GenerationError specifically captures failures during the actual AI generation
    process (model inference, GPU errors, output generation, etc.)
    """
    
    def __init__(self, 
                 app_name: str,
                 model: str,
                 feature: str,
                 reason: str,
                 generation_params: Optional[Dict[str, Any]] = None,
                 gpu_info: Optional[Dict[str, Any]] = None,
                 traceback: Optional[str] = None):
        message = f"Generation failed for {feature} using model '{model}': {reason}"
        details = {
            "model": model,
            "feature": feature,
            "reason": reason
        }
        
        if generation_params:
            # Don't include sensitive data like full prompts
            safe_params = {
                k: v for k, v in generation_params.items() 
                if k in ["height", "width", "num_frames", "guidance_scale", "num_inference_steps"]
            }
            details["generation_params"] = safe_params
            
        if gpu_info:
            details["gpu_info"] = gpu_info
            
        if traceback:
            # Truncate long tracebacks
            details["traceback"] = traceback[-1000:] if len(traceback) > 1000 else traceback
            
        super().__init__(message, details)
        self.app_name = app_name
        self.model = model
        self.feature = feature
        self.reason = reason
        self.generation_params = generation_params
        self.gpu_info = gpu_info
        self.traceback = traceback
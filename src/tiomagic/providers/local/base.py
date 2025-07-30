import os
from typing import Any, Dict

from ...core.errors import ConfigurationError, GenerationError, ValidationError
from ...core.constants import Generation


class LocalProviderBase:
    """Base class to interface with local implementation
    """
    def __init__(self):
        self.module_path = os.path.abspath(__file__)
        # These should be set by subclasses
        self.app_name = None
        self.feature = None
    def _validate_config(self):
        """Validate that required configuration is set
        """
        if not self.app_name:
            raise ConfigurationError(config_type="app_name", message="subclass must set app_name", missing_field="app_name")
        if not self.feature:
            raise ConfigurationError(config_type="feature", message="subclass must set feature", missing_field="feature")
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare the payload for model request.
        
        Args:
            required_args: The required arguments of the model
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the request payload
        """
        prompt = required_args.get('prompt', None)
        if required_args is None or prompt is None:
            raise ValidationError(
                field="prompt",
                message="Arguemt 'prompt' is required generation",
                value=prompt
            )

        payload = {"prompt": prompt}            

        for key, value in required_args.items():
            if value is not None:
                payload[key] = value
        # Add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        return payload
    def generate(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """This is the main entry point for generation.
        Args:
            required_args: The generation prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generation result with call_id and status
        """
        self._validate_config()

        from datetime import datetime
        print(f"Running {self.app_name} generate with prompt: {required_args['prompt']}")

        generation = Generation(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            required_args=required_args,
            optional_args=kwargs
        )

        try:
            self._prepare_payload(required_args, **kwargs)

            # This should be implemented by subclasses
            # For now, return a basic response
            generation.update(
                call_id="local-generation",
                status="completed",
                message="Local generation completed"
            )

            return generation.to_dict()

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            generation.update(message=f"Error in generate: {str(e)}")
            raise GenerationError(reason=str(e))


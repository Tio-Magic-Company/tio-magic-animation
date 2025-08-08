"""Base class for Modal model implementations in TioMagic.
"""
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse
import modal
import os
from typing import Any, Dict, Optional
from enum import Enum

from ...core.errors import APIError, ConfigurationError, GenerationError
from ...core._utils import check_modal_app_deployment, create_timestamp
from ...core.constants import Generation
from ...core._jobs import JobStatus
import requests

class GPUType(str, Enum):
    """Supported GPU types in Modal"""
    B200 = "B200"
    H200 = "H200"
    H100 = "H100"
    A100_80GB = "A100-80GB"
    A100_40GB = "A100-40GB"
    L40S = "L40S"
    A10G = "A10G"
    L4 = "L4"
    T4 = "T4"

# Class constants to be overridden by subclasses
APP_NAME: str = None                # Modal app name
MODEL_ID: str = None                # HuggingFace model ID
GPU_CONFIG: GPUType = GPUType.A100_80GB  # Default GPU type
CACHE_PATH: str = "/cache"           # Path to cache directory
OUTPUTS_PATH: str = "/outputs"       # Path to outputs directory
TIMEOUT: int = 1800                 # Container timeout in seconds
SCALEDOWN_WINDOW: int = 900         # Container scaledown window in seconds

class ModalProviderBase:
    """Base class to interface with Modal app
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.module_path = os.path.abspath(__file__)

        # These should be set by subclasses
        self.app_name = None
        self.modal_app = None
        self.modal_class_name = None
    def _validate_config(self):
        """Validate that required configuration is set
        """
        if not self.app_name:
            raise ConfigurationError(config_type="app_name", message="subclass must set app_name", missing_field="app_name")
        if not self.modal_app:
            raise ConfigurationError(config_type="modal_app", message="subclass must set modal_app", missing_field="modal_app")
        if not self.modal_class_name:
            raise ConfigurationError(config_type="modal_class_name", message="subclass must set modal_class_name", missing_field="modal_class_name")

    def generate(self, required_args: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate content using Modal web inference.
        This is the main entry point for generation that creates a Modal instance.
        Args:
            required_args: The generation prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generation result with call_id and status
        """
        self._validate_config()

        print(f"--> Running {self.app_name} generate with prompt: {required_args['prompt']}")

        generation = Generation(
            timestamp=create_timestamp(),
            required_args=required_args,
            optional_args=kwargs
        )

        try:
            # Check deployment status
            deployment_status = check_modal_app_deployment(self.modal_app, self.app_name)
            print("--> Deployment Status: ", deployment_status)

            if not deployment_status['success']:
                generation.update(message=f"--> App '{self.app_name}' is not deployed: {deployment_status['message']}")
                return generation.to_dict()

            if not deployment_status['endpoints']:
                generation.update(message=f"--> No endpoints found for app '{self.app_name}': {deployment_status['message']}")
                return generation.to_dict()

            # Get Modal WebAPI class
            modal_web_api_class = modal.Cls.from_name(self.app_name, 'WebAPI')
            print('--> Modal instance', modal_web_api_class)

            web_inference_url = modal_web_api_class().web_inference.get_web_url()
            print("--> Web url: ", web_inference_url)

            # Prepare request payload
            payload = self._prepare_payload(required_args, **kwargs)
            # print("payload: ", payload)

            # Make web inference request
            call_id = self._make_web_inference_request(web_inference_url, payload, generation)
            if not call_id:
                return generation.to_dict()

            # Update generation with call_id
            message = "--> Generation queued."
            generation.update(call_id=call_id, status=JobStatus.QUEUED, message=message)
            print(f"--> Received call_id: {call_id}")

            return generation.to_dict()
        except Exception as e:
            generation.update(message=f"Error in generate: {str(e)}")
            raise GenerationError(app_name=APP_NAME,
                                  model=MODEL_ID,
                                  reason=str(e))

    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare the payload for web inference request.
        
        Args:
            required_args: The required arguments of the model
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the request payload
        """
        print("--> Preparing payload")

        # Base payload - subclasses can override to add model-specific parameters
        if required_args is None or required_args['prompt'] is None:
            raise ValueError("required_args and prompt are required")

        payload = {"prompt": required_args['prompt']}            

        for key, value in required_args.items():
            if value is not None:
                payload[key] = value
        # Add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        return payload

    def _make_web_inference_request(self, url: str, payload: Dict[str, Any], generation: Generation) -> Optional[str]:
        """Make the web inference request and return call_id.
        
        Args:
            url: The web inference URL
            payload: The request payload
            generation: The generation object to update with errors
            
        Returns:
            The call_id if successful, None otherwise
        """
        from json import JSONDecodeError
        try:
            print("--> Making Web Inference Request - this may take a few minutes if it is your first time deploying" )
            response = requests.post(
                url, 
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            print("response: ", response)

            if response.status_code != 200:
                generation.update(message=f"Error calling web_inference: {response.status_code}, {response.text}")
                return None

            try:
                response_data = response.json()
                print(f"--> Web inference response: {response_data}")
            except JSONDecodeError:
                generation.update(message=f"Error parsing response as JSON: {response.text}")
                return None

            if "call_id" not in response_data:
                generation.update(message="No call_id in response")
                return None

            return response_data["call_id"]

        except requests.RequestException as e:
            generation.update(message=f"Request error: {str(e)}")
            raise APIError(response_body=str(e))

    def check_generation_status(self, generation: Generation) -> Generation:
        """Check the status of a previous generate call.
        
        Args:
            generation: The generation object containing call_id
            
        Returns:
            Updated generation object with status and results
        """
        from modal.functions import FunctionCall

        try:
            fc = FunctionCall.from_id(generation["call_id"])
            video_bytes = fc.get(timeout=0)

            # Save video to file
            video_path = self._save_video_file(video_bytes, generation["call_id"])

            print(f"--> Generation Complete. Video saved to: {video_path}")
            generation.update(
                status=JobStatus.COMPLETED, 
                timestamp=create_timestamp(),
                message=f"video completed and saved to {video_path}",
                result_video=video_path
            )

        except TimeoutError:
            print("--> Generation is not complete yet. Please check back later")
            generation.update(
                status=JobStatus.RUNNING,
                message="generation is not complete yet",
                timestamp=create_timestamp()
            )

        except Exception as e:
            self._handle_generation_error(e, generation)
            raise GenerationError(reason=str(e))

        return generation

    def cancel_job(self, generation: Generation) -> Generation:
        from modal.functions import FunctionCall
        print("--> Canceling job")
        try:
            fc = FunctionCall.from_id(generation["call_id"])
            fc.cancel()
            generation.update(
                status=JobStatus.CANCELED,
                message="generation canceled",
                timestamp = create_timestamp()
            )
        except Exception as e:
            print("EXCEPTION IN MODAL CANCEL JOB", e)
            self._handle_generation_error(e, generation)
            raise GenerationError(reason=str(e))
        return generation

    def _save_video_file(self, video_bytes: bytes, call_id: str) -> str:
        """Save video bytes to a file.
        
        Args:
            video_bytes: The video data
            call_id: The generation call ID
            
        Returns:
            Path to the saved video file
        """
        timestamp = create_timestamp()
        video_filename = f"{call_id}_{timestamp}.mp4"

        # Get the directory of this file and navigate to repo root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        output_videos_dir = os.path.join(repo_root, "output_videos")
        # Ensure the output_videos directory exists
        os.makedirs(output_videos_dir, exist_ok=True)
        video_path = os.path.join(output_videos_dir, video_filename)

        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        return video_path

    def _handle_generation_error(self, error: Exception, generation: Generation):
        """Handle generation errors.
        
        Args:
            error: The exception that occurred
            generation: The generation object to update
        """
        if error.args:
            print("e args", error.args)
            inner_e = error.args[0]
            if "HTTPError 403" in inner_e:
                generation.update(message="permission denied on video download")
            else:
                generation.update(status=JobStatus.CANCELED, message=inner_e)
        else:
            generation.update(message=str(error))


class GenericWebAPI:
    """Generic WebAPI class that can be reused across different models."""

    # Class variable to be set by subclasses
    feature_handlers: Dict[str, Any] = {}

    @modal.fastapi_endpoint(method="POST")
    def web_inference(self, data: dict = Body(...)):
        print("POST WEB INFERENCE")
        feature_type = data.pop("feature_type", None)

        if not feature_type:
            return {"error": f"A 'feature_type' is required. Must be one of: {list(self.feature_handlers.keys())}"}

        if feature_type not in self.feature_handlers:
            return {"error": f"Unknown feature_type: {feature_type}. Must be one of: {list(self.feature_handlers.keys())}"}

        # Route to appropriate class
        handler_class = self.feature_handlers[feature_type]
        
        # If user specifies gpu, timeout, scaledown window, applied here
        if 'modal_options' in data and data['modal_options']:
            modal_options = data['modal_options']
            for key, value in modal_options.items():
                if value is not None:
                    print(f"Applying modal option: {key} = {value}")
                    handler_class = handler_class.with_options(**{key: value})
            data.pop('modal_options')
        return handler_class.handle_web_inference(data)

    @modal.fastapi_endpoint(method="GET")
    def get_result(self, call_id: str, feature_type: str = None):
        """Unified FastAPI endpoint to poll for results from any class.
        """
        import io
        from modal import FunctionCall

        print(f"--> Polling for call_id: {call_id}, feature_type: {feature_type}")

        try:
            call = FunctionCall.from_id(call_id)
            video_bytes = call.get(timeout=0)  # Use a short timeout to check for completion
        except TimeoutError:
            return JSONResponse({"status": "processing"}, status_code=202)
        except Exception as e:
            print(f"Error fetching result for {call_id}: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

        # Determine filename based on feature type
        if feature_type in self.feature_handlers:
            filename = f"{feature_type}-output_{call_id}.mp4"
        else:
            filename = f"output_{call_id}.mp4"

        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4", headers=headers)

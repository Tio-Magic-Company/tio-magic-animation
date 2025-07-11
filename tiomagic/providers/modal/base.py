"""
Base class for Modal model implementations in TioMagic.
"""

from abc import abstractmethod
import modal
import os
from typing import Any, Dict, Optional, Union
from enum import Enum
from datetime import datetime
from pathlib import Path

# from ...core.interfaces import TextToVideoInterface

from ...core.utils import check_modal_app_deployment, Generation
from ...core.jobs import JobStatus
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

class ModalBase:
    """Base class for Modal model implementations
    
    This class defines the contract that all Modal implementations must follow.
    It provides common utilities and standardized methods for Modal deployment.
    """
    
    # Class constants to be overridden by subclasses
    APP_NAME: str = None                # Modal app name
    MODEL_ID: str = None                # HuggingFace model ID
    GPU_CONFIG: GPUType = GPUType.A10G  # Default GPU type
    CACHE_DIR: str = "/cache"           # Path to cache directory
    OUTPUTS_DIR: str = "/outputs"       # Path to outputs directory
    TIMEOUT: int = 600                  # Container timeout in seconds
    SCALEDOWN_WINDOW: int = 900         # Container scaledown window in seconds

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Modal base class.
        
        Args:
            api_key: Optional Modal API key. If not provided, will use environment variables.
        """
        if api_key:
            os.environ["MODAL_TOKEN_ID"] = api_key.split(":")[0]
            os.environ["MODAL_TOKEN_SECRET"] = api_key.split(":")[1]
            
        self._validate_class_constants()
        self.outputs_path = Path(self.OUTPUTS_DIR)
        self._setup_modal_resources()
    
    def _validate_class_constants(self):
        """Validate that required class constants are defined."""
        if not self.APP_NAME:
            raise ValueError("Subclass must define APP_NAME")
        if not self.MODEL_ID:
            raise ValueError("Subclass must define MODEL_ID")
    
    def _setup_modal_resources(self):
        """Set up Modal app, image, volumes and other resources."""
        # Create volumes for cache and outputs
        cache_volume_name = f"{self.APP_NAME}-cache"
        outputs_volume_name = f"{self.APP_NAME}-outputs" 
        
        cache_volume = modal.Volume.from_name(
            cache_volume_name, create_if_missing=True
        )
        outputs_volume = modal.Volume.from_name(
            outputs_volume_name, create_if_missing=True
        )
        
        # Set up the Modal image with required dependencies
        image = self._build_image()
        
        # Create the Modal app
        app = modal.App(
            self.APP_NAME,
            image=image,
            secrets=[modal.Secret.from_name("huggingface-secret")]
        )
        
        # Store references
        self.cache_volume = cache_volume
        self.outputs_volume = outputs_volume
        self.image = image
        self.app = app
        
        # Get the implementation class defined in the module
        # print(f"**APP NAME: , {self.APP_NAME}")
        # self._get_modal_class()

    def _build_image(self) -> modal.Image:
        """Build the base Modal image with required dependencies.
        
        This method can be overridden by subclasses to add model-specific dependencies.
        
        Returns:
            modal.Image: The Modal image
        """
        return modal.Image.debian_slim(python_version="3.11").env({"HF_HUB_CACHE": self.CACHE_DIR})
    
    # def _get_modal_class(self):
        """Look for and validate the Modal implementation class in the module.
        
        This method finds the Modal class that should be defined at the module level
        in the subclass's module. It will try to find either:
        1. A class with the specific name derived from APP_NAME
        2. Any class with Modal decorators in the module
        
        Raises:
            ValueError: If no suitable Modal implementation class is found
        """
        # import sys
        # import inspect
        
        # Get the module where the subclass is defined
        # module = sys.modules[self.__class__.__module__]
        
        # # Expected class name based on APP_NAME
        # expected_class_name = self.APP_NAME.replace("-", "_").title() + "Implementation"
        # print("expected class name: ", expected_class_name)
        
        # Find all classes in the module that have Modal decorators
        # modal_classes = []
        # for name, obj in inspect.getmembers(module, inspect.isclass):
        #     print("NAME: ", name, "OBJ: ", obj, hasattr(obj, 'model_implementation'))
        #     # Check if it has Modal's cls decorator
        #     if hasattr(obj, 'cls_id') or hasattr(obj, 'cls_decorator_kwargs'):
        #         modal_classes.append((name, obj))
                
                # If this class matches the expected name, use it
                # if name == expected_class_name:
                #     self.model_implementation = obj
                #     return
        
        # If we didn't find the expected class but found other Modal classes
        # if modal_classes:
        #     class_name, cls = modal_classes[0]
        #     print(f"Warning: Using Modal class '{class_name}' instead of expected '{expected_class_name}'")
        #     self.model_implementation = cls
        #     return
            
        # No Modal classes found
        # available_classes = ", ".join(name for name, _ in inspect.getmembers(module, inspect.isclass))
        # raise ValueError(
        #     f"No Modal implementation class found in {self.__class__.__module__}. "
        #     f"Expected class '{expected_class_name}' with @app.cls decorator. "
        #     f"Available classes: {available_classes}. "
        #     f"Please define a Modal class at the module level with the "
        #     f"@app.cls decorator and the required Modal methods."
        # )
    
    def _validate_web_inputs(self, data: dict) -> dict:
        """Validate inputs from web API.
        
        Subclasses can override this to provide custom validation.
        
        Args:
            data: Input data from web API
            
        Returns:
            dict: Validated inputs or error response
        """
        # Check for required prompt
        if "prompt" not in data:
            return {"error": "A 'prompt' field is required."}
        
        return data
    
    @abstractmethod
    def _load_pipeline(self, start_time: float) -> Any:
        """Load and return the model pipeline.
        
        This method must be implemented by subclasses to load the model pipeline.
        The timer is provided to allow for logging of load times.
        
        Args:
            start_time: Start time for loading models (for logging purposes)
            
        Returns:
            Any: The model pipeline object
        """
        pass
    
    @abstractmethod
    def _run_generation(self, pipeline: Any, **kwargs) -> Union[bytes, str, Path]:
        """Run generation using the loaded pipeline.
        
        This method must be implemented by subclasses to run the model.
        
        Args:
            pipeline: The loaded model pipeline from _load_pipeline
            **kwargs: Generation parameters
            
        Returns:
            Union[bytes, str, Path]: Generated content or path to generated file
        """
        pass

    def _get_result_filename(self, call_id: str) -> str:
        """Get filename for the generated result.
        
        Subclasses can override this to provide custom filenames.
        
        Args:
            call_id: Modal function call ID
            
        Returns:
            str: Filename for the result
        """
        app_name = self.APP_NAME.replace("-", "_")
        return f"{app_name}_output_{call_id}.mp4"
    
    def _get_result_media_type(self) -> str:
        """Get media type for the generated result.
        
        Subclasses can override this to provide custom media types.
        
        Returns:
            str: Media type string
        """
        return "video/mp4"
    
    def generate(self, prompt: str, **kwargs) -> bytes:
        """Generate content using the model.
        
        This is the main entry point for generation.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters
            
        Returns:
            bytes: Generated content
        """
        # Create a Modal function call to the implementation
        instance = self.model_implementation()
        
        # Pass the base instance for access to methods
        setattr(instance, "_outer_self", self)
        
        # Call the generate method on the Modal class
        return instance.generate.remote(prompt=prompt, **kwargs)
    
    def deploy(self):
        """Deploy the Modal app.
        
        This method deploys the app to Modal.
        """
        self.app.deploy()

class ModalProviderBase:
    """
    Base class to interface with Modal app
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.module_path = os.path.abspath(__file__)
        
        # These should be set by subclasses
        self.app_name = None
        self.modal_app = None
        self.modal_class_name = None
    def _validate_config(self):
        """
        Validate that required configuration is set
        """
        if not self.app_name:
            raise ValueError("Subclass must set app_name")
        if not self.modal_app:
            raise ValueError("Subclass must set modal_app")
        if not self.modal_class_name:
            raise ValueError("Subclass must set modal_class_name")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate content using Modal web inference.
        This is the main entry point for generation that creates a Modal instance.
        Args:
            prompt: The generation prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generation result with call_id and status
        """
        self._validate_config()

        from datetime import datetime
        print(f"Running {self.app_name} generate with prompt: {prompt}")
        
        generation = Generation(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            prompt=prompt,
            optional_parameters=kwargs
        )

        try:
            # Check deployment status
            deployment_status = check_modal_app_deployment(self.modal_app, self.app_name)
            print("return status: ", deployment_status)
            
            if not deployment_status['success']:
                generation.update(message=f"App '{self.app_name}' is not deployed: {deployment_status['message']}")
                return generation.to_dict()
                
            if not deployment_status['endpoints']:
                generation.update(message=f"No endpoints found for app '{self.app_name}': {deployment_status['message']}")
                return generation.to_dict()
            
            # Get Modal class and web URLs
            print("app name: ", self.app_name)
            # print("modal class name: ", self.modal_class_name)
            # modal_class = modal.Cls.from_name(self.app_name, self.modal_class_name)
            # print("modal instance, ", modal_class)
            
            # web_inference_url = modal_class().web_inference.get_web_url()
            # print("web_url: ", web_inference_url)

            # Get Modal WebAPI class
            modal_web_api_class = modal.Cls.from_name(self.app_name, 'WebAPI')
            print('modal instance', modal_web_api_class)

            web_inference_url = modal_web_api_class().web_inference.get_web_url()
            print("web url: ", web_inference_url)

            # Prepare request payload
            payload = self._prepare_payload(prompt, **kwargs)
            
            # Make web inference request
            call_id = self._make_web_inference_request(web_inference_url, payload, generation)
            if not call_id:
                return generation.to_dict()
            
            # Update generation with call_id
            message = "generation queued. This may take a few minutes if it is the first time you are running this model"
            generation.update(call_id=call_id, status=JobStatus.QUEUED, message=message)
            print(f"Got call_id: {call_id}")
            
            return generation.to_dict()
        except Exception as e:
            print(f"Error in generate: {str(e)}")
            generation.update(message=f"Error in generate: {str(e)}")
            raise

    def _prepare_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare the payload for web inference request.
        
        Args:
            prompt: The generation prompt
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the request payload
        """
        # Base payload - subclasses can override to add model-specific parameters
        payload = {"prompt": prompt}
        
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
            print("enter try - this may take a few minutes if it is your first time deploying" )
            print("post payload", payload)
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
                print(f"Web inference response: {response_data}")
            except JSONDecodeError:
                generation.update(message=f"Error parsing response as JSON: {response.text}")
                return None
            
            if "call_id" not in response_data:
                generation.update(message="No call_id in response")
                return None
                
            return response_data["call_id"]

        except requests.RequestException as e:
            print("request error")
            generation.update(message=f"Request error: {str(e)}")
            return None
    
    def check_generation_status(self, generation: Generation) -> Generation:
        """Check the status of a previous generate call.
        
        Args:
            generation: The generation object containing call_id
            
        Returns:
            Updated generation object with status and results
        """
        from modal.functions import FunctionCall
        from datetime import datetime
        print("MODAL CHECK GENERATION STATUS")

        try:
            print("enter try", generation["call_id"])
            fc = FunctionCall.from_id(generation["call_id"])
            video_bytes = fc.get(timeout=0)

            # Save video to file
            video_path = self._save_video_file(video_bytes, generation["call_id"])
            
            print(f"Video saved to: {video_path}")
            generation.update(
                status=JobStatus.COMPLETED, 
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                message=f"video completed and saved to {video_path}",
                result_video=video_path
            )
            
        except TimeoutError:
            print("NOT READY YET - MODAL CHECK GENERATION STATUS")
            generation.update(
                status=JobStatus.RUNNING,
                message="generation is not complete yet",
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
        except Exception as e:
            print("EXCEPTION IN MODAL CHECK GEN STATUS", e)
            self._handle_generation_error(e, generation)
        
        return generation
    
    def _save_video_file(self, video_bytes: bytes, call_id: str) -> str:
        """Save video bytes to a file.
        
        Args:
            video_bytes: The video data
            call_id: The generation call ID
            
        Returns:
            Path to the saved video file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"generated_video_{call_id}_{timestamp}.mp4"
        
        # Get the directory of this file and save to the same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        video_path = os.path.join(repo_root, video_filename)
        
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
     


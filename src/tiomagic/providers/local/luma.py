import time
import os
from typing import Any, Dict

import requests

from ...core.errors import GenerationError, ProcessingError
from ...core._utils._utils import create_timestamp

from ...core.registry import registry
from ...core._utils import is_local_path
from ...core.constants import Generation
from .base import LocalProviderBase
from ...core.constants import FeatureType

from lumaai import LumaAI
# async options available, check Luma Documentation

APP_NAME = "test-luma-ray-2"
MODEL_NAME = "ray-2"

class LumaRay2I2V(LocalProviderBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("LUMAAI_API_KEY")
        if not api_key:
            raise ValueError("LUMAAI_API_KEY not found in environment variables. Please set it in your .env file.")
        self.client = LumaAI(
            auth_token=os.environ.get("LUMAAI_API_KEY"),
        )
        self.app_name = APP_NAME
        self.feature = FeatureType.IMAGE_TO_VIDEO
        self.operations = None
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.IMAGE_TO_VIDEO

        if payload['image'] is None:
            raise ValueError("Argument 'image' is required for Image to Video generation")

        if is_local_path(payload['image']):
            raise ProcessingError(
                media_type="image",
                operation="load local image",
                reason="Luma AI API only takes image URLs, not local files"
            )
        
        keyframes = {
                "frame0": {
                    "type": "image",
                    "url": payload['image']
                }
            }
        payload['keyframes'] = keyframes

        return payload
    def generate(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        self._validate_config()
        print(f"Running {self.app_name} generate with prompt: {required_args['prompt']}")

        generation_obj = Generation(
            timestamp=create_timestamp(),
            required_args=required_args,
            optional_args=kwargs
        )
        print('-->generation object done', generation_obj)

        try:
            payload = self._prepare_payload(required_args, **kwargs)
            print('-->payload done')

            generation = self.client.generations.create(
                model=MODEL_NAME,
                prompt = payload['prompt'],
                keyframes = payload['keyframes']
            )
            print("--> generation creation done", generation)

            completed = False
            while not completed:
                generation = self.client.generations.get(id=generation.id)
                if generation.state == "completed":
                    completed = True
                elif generation.state == "failed":
                    generation_obj.update(
                        status="failed",
                        message=f"Generation failed: {generation.failure_reason}"
                    )
                    raise GenerationError(
                        app_name=APP_NAME,
                        model=MODEL_NAME,
                        feature=FeatureType.IMAGE_TO_VIDEO,
                        reason=f"{generation.failure_reason}",
                    )
                time.sleep(3)
            save_video(generation, generation_obj)

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            generation_obj.update(message=f"Error in generate: {str(e)}")

            raise GenerationError(app_name=APP_NAME, 
                                  model=MODEL_NAME,
                                  reason=str(e)
                                  )

class LumaRay2Interpolate(LocalProviderBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("LUMAAI_API_KEY")
        if not api_key:
            raise ValueError("LUMAAI_API_KEY not found in environment variables. Please set it in your .env file.")
        self.client = LumaAI(
            auth_token=os.environ.get("LUMAAI_API_KEY"),
        )
        self.app_name = APP_NAME
        self.feature = FeatureType.INTERPOLATE
        self.operations = None
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.INTERPOLATE

        if payload['first_frame'] is None:
            raise ValueError("Argument 'first_frame' is required for Interpolate generation")
        if payload['last_frame'] is None:
            raise ValueError("Argument 'last_frame' is required for Interpolate generation")

        if is_local_path(payload['first_frame']) or is_local_path(payload['last_frame']):
            raise ProcessingError(
                media_type="image",
                operation="load local image",
                reason="Luma AI API only takes image URLs, not local files"
            )
        
        keyframes = {
                "frame0": {
                    "type": "image",
                    "url": payload['first_frame']
                },
                "frame1": {
                    "type": "image",
                    "url": payload['last_frame']
                }
            }
        payload['keyframes'] = keyframes

        return payload
    def generate(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        self._validate_config()
        print(f"Running {self.app_name} generate with prompt: {required_args['prompt']}")

        generation_obj = Generation(
            timestamp = create_timestamp(),
            required_args=required_args,
            optional_args=kwargs
        )
        print('-->generation object done', generation_obj)

        try:
            payload = self._prepare_payload(required_args, **kwargs)
            print('-->payload done')

            generation = self.client.generations.create(
                model=MODEL_NAME,
                prompt = payload['prompt'],
                keyframes = payload['keyframes']
            )
            print("--> generation creation done", generation)

            completed = False
            while not completed:
                generation = self.client.generations.get(id=generation.id)
                if generation.state == "completed":
                    completed = True
                elif generation.state == "failed":
                    generation_obj.update(
                        status="failed",
                        message=f"Generation failed: {generation.failure_reason}"
                    )
                    raise GenerationError(
                        app_name=APP_NAME,
                        model=MODEL_NAME,
                        feature=FeatureType.IMAGE_TO_VIDEO,
                        reason=f"{generation.failure_reason}",
                    )
                time.sleep(3)
            save_video(generation, generation_obj)
            
        except Exception as e:
            print(f"Error in generate: {str(e)}")
            generation_obj.update(message=f"Error in generate: {str(e)}")

            raise GenerationError(app_name=APP_NAME, 
                                  model=MODEL_NAME,
                                  reason=str(e)
                                  )

def save_video(generation, generation_obj):
    video_url = generation.assets.video

    response = requests.get(video_url, stream=True)

    # Save video to file
    timestamp = create_timestamp()
    video_filename = f"{MODEL_NAME}_output_{timestamp}.mp4"

    # Get the directory of this file and navigate to repo root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    output_videos_dir = os.path.join(repo_root, "output_videos")
    video_path = os.path.join(output_videos_dir, video_filename)

    # Create output_videos directory if it doesn't exist
    os.makedirs(output_videos_dir, exist_ok=True)
    
    # Save the video to the output_videos directory
    with open(video_path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded as {video_path}")

    generation_obj.update(
            call_id=generation.id,
            status="completed",
            message=f"Video generated and saved to {video_path}",
            result_video=video_path
        )

registry.register(
    feature=FeatureType.IMAGE_TO_VIDEO,
    model="luma-ray-2",
    provider="local",
    implementation=LumaRay2I2V
) 
registry.register(
    feature=FeatureType.INTERPOLATE,
    model="luma-ray-2",
    provider="local",
    implementation=LumaRay2Interpolate
)
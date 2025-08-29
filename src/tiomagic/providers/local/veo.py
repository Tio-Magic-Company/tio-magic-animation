import time
from fastapi.responses import JSONResponse
import os
from typing import Any, Dict

from ...core.errors import GenerationError

from ...core.registry import registry
from ...core._utils import is_local_path, create_timestamp
from ...core.constants import Generation
from .base import LocalProviderBase
import base64
from ...core.constants import FeatureType


from google import genai
from google.genai import types


class Veo20Generate001(LocalProviderBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.app_name = "veo-2.0-generate-001"
        self.model_name = "veo-2.0-generate-001"
        self.feature = FeatureType.IMAGE_TO_VIDEO
        self.operations = None


    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Veo 2.0 Generate 001 model.
        Break out required args into payload
        """
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.IMAGE_TO_VIDEO

        if payload['image'] is None:
            raise ValueError("Argument 'image' is required for Image to Video generation")

        if is_local_path(payload['image']):
            print(f'Uploading local image: {payload["image"]}')
            # uploaded_file = self.client.files.upload(file=payload['image'])
            # print(f'Initial upload complete: {uploaded_file.name}')
            print("--> payload image path ", len(payload['image']))
            file = types.Image.from_file(location=payload['image'])
            payload['image'] = file
        return payload

    def generate(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate video using Google Veo API
        """
        self._validate_config()

        from datetime import datetime
        print(f"Running {self.app_name} generate with prompt: {required_args['prompt']}")

        generation = Generation(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            required_args=required_args,
            optional_args=kwargs
        )
        print('-->generation object done', generation)


        try:
            payload = self._prepare_payload(required_args, **kwargs)
            print('-->payload done')

            # Call Google Veo API
            config = types.GenerateVideosConfig(**kwargs)

            operation = self.client.models.generate_videos(
                model=self.model_name,
                prompt = payload['prompt'],
                image= payload['image'],
                config=config
            )
            print("--> Operation creation done", operation)

            print(f"--> Generation started with operation: {operation.name}")

            # https://ai.google.dev/gemini-api/docs/video#generate-from-images
            # https://github.com/googleapis/python-genai/blob/main/google/genai/operations.py#L348
            # ASYNC FUTURE IMPLEMENTATION
            # # Store operation for later status checking
            # self.operations[operation.name] = {
            #     'operation': operation,
            #     'generation': generation,
            #     'started_at': datetime.now()
            # }

            # print(f"Generation started with operation ID: {operation.name}")
            # generation.update(
            #     call_id = operation,
            #     status=JobStatus.running
            # )
            # generation.to_dict()

            # # Return immediately with operation ID
            # return {
            #     'call_id': operation.name,
            #     'feature_type': 'image_to_video',
            # }

            # Wait for completion
            while not operation.done:
                print("Waiting for video generation to complete...")
                time.sleep(10)
                operation = self.client.operations.get(operation)
            
            # Download the video
            generated_video = operation.response.generated_videos[0]
            video_bytes = self.client.files.download(file=generated_video.video)
            # generated_video.video.save("veo-2-i2v.mp4")
            timestamp = create_timestamp()

            video_filename = f"veo_output_{timestamp}.mp4"

            # Get the directory of this file and save to the same directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
            output_videos_dir = os.path.join(repo_root, "output_videos")
            os.makedirs(output_videos_dir, exist_ok=True)
            video_path = os.path.join(output_videos_dir, video_filename)

            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            print(f"File downloaded as {video_path}")
            

            generation.update(
                call_id=operation.name,
                status="completed",
                message=f"Video generated and saved to {video_path}",
                result_video=video_path
            )

            return generation.to_dict()

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            generation.update(message=f"Error in generate: {str(e)}")

            raise GenerationError(app_name=self.app_name, 
                                  model=self.model_name,
                                  feature=FeatureType.IMAGE_TO_VIDEO,
                                  reason=str(e)
                                  )
    
class Veo30GeneratePreview(LocalProviderBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.app_name = "veo-3.0-generate-preview"
        self.model_name = "veo-3.0-generate-preview"
        self.feature = FeatureType.IMAGE_TO_VIDEO
        self.operations = None


    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to veo-3.0-generate-preview model.
        Break out required args into payload
        """
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.IMAGE_TO_VIDEO

        if payload['image'] is None:
            raise ValueError("Argument 'image' is required for Image to Video generation")

        if is_local_path(payload['image']):
            print(f'Uploading local image: {payload["image"]}')
            print("--> payload image path ", len(payload['image']))
            file = types.Image.from_file(location=payload['image'])
            payload['image'] = file
        return payload

    def generate(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate video using Google Veo API
        """
        self._validate_config()

        from datetime import datetime
        print(f"Running {self.app_name} generate with prompt: {required_args['prompt']}")

        generation = Generation(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            required_args=required_args,
            optional_args=kwargs
        )
        print('-->generation object done', generation)


        try:
            payload = self._prepare_payload(required_args, **kwargs)
            print('-->payload done')

            # Call Google Veo API
            config = types.GenerateVideosConfig(**kwargs)

            operation = self.client.models.generate_videos(
                model=self.model_name,
                prompt = payload['prompt'],
                image= payload['image'],
                config=config
            )
            print("--> Operation creation done", operation)

            print(f"--> Generation started with operation: {operation.name}")

            # Wait for completion
            while not operation.done:
                print("Waiting for video generation to complete...")
                time.sleep(10)
                operation = self.client.operations.get(operation)
            
            # Download the video
            generated_video = operation.response.generated_videos[0]
            video_bytes = self.client.files.download(file=generated_video.video)
            # generated_video.video.save("veo-2-i2v.mp4")
            timestamp = create_timestamp()

            video_filename = f"veo_output_{timestamp}.mp4"

            # Get the directory of this file and save to the same directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
            output_videos_dir = os.path.join(repo_root, "output_videos")
            os.makedirs(output_videos_dir, exist_ok=True)
            video_path = os.path.join(output_videos_dir, video_filename)

            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            print(f"File downloaded as {video_path}")
            

            generation.update(
                call_id=operation.name,
                status="completed",
                message=f"Video generated and saved to {video_path}",
                result_video=video_path
            )

            return generation.to_dict()

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            generation.update(message=f"Error in generate: {str(e)}")

            raise GenerationError(app_name=self.app_name, 
                                  model=self.model_name,
                                  feature=FeatureType.IMAGE_TO_VIDEO,
                                  reason=str(e)
                                  )
registry.register(
    feature="image_to_video",
    model="veo-2.0-generate-001",
    provider="local",
    implementation=Veo20Generate001
)

registry.register(
    feature="image_to_video",
    model="veo-3.0-generate-preview",
    provider="local",
    implementation=Veo30GeneratePreview
)

def base64_to_bytes(data_url):
    if data_url.startswith("data:"):
        header, b64data = data_url.split(",", 1)
    else:
        b64data = data_url
    return base64.b64decode(b64data)

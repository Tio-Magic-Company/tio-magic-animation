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

APP_NAME = "test-veo-2.0-generate-001"
MODEL_NAME = 'veo-2.0-generate-001'
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables")

class Veo20Generate001(LocalProviderBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.app_name = APP_NAME
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


        #     # Wait a moment for processing
        #     import time
        #     time.sleep(2)

        #     # Retrieve the file to ensure it's fully processed
        #     retrieved_file = self.client.files.get(name=uploaded_file.name)

        #     # Check if file is active
        #     print(f'File state: {retrieved_file.state}')
        #     print(f'File details: name={retrieved_file.name}, mime_type={retrieved_file.mime_type}, size={retrieved_file.size_bytes}')

        #     # Wait for file to be active if needed
        #     max_attempts = 10
        #     attempt = 0
        #     while retrieved_file.state != "ACTIVE" and attempt < max_attempts:
        #         print(f"Waiting for file to become active... (attempt {attempt + 1})")
        #         time.sleep(2)
        #         retrieved_file = self.client.files.get(name=uploaded_file.name)
        #         attempt += 1

        #     if retrieved_file.state != "ACTIVE":
        #         raise ValueError(f"File upload failed or timed out. State: {retrieved_file.state}")

        #     payload['image'] = retrieved_file
        #     print(f'File ready for use: {retrieved_file.name}')

        # elif isinstance(payload['image'], str) and payload['image'].startswith('data:'):
        #     # Similar handling for base64...
        #     pass

        return payload
            # image = PIL.Image.open(payload['image'])
            # payload['image'] = image
            # base64_image = local_image_to_base64(payload['image'])
            # bytes_image = base64_to_bytes(base64_image)
            # payload['image'] = {"bytesBase64Encoded": bytes_image, "mimeType": "image/png"}

            # payload['image'] = base64_image.split(",", 1)[1]
            # payload["image"] = base64_to_bytes(base64_image)
        # return payload

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
                model=MODEL_NAME,
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

            raise GenerationError(app_name=APP_NAME, 
                                  model=MODEL_NAME,
                                  feature=FeatureType.IMAGE_TO_VIDEO,
                                  reason=str(e)
                                  )
    # def check_generation_status(self, generation: Generation) -> Generation:
    #     """Check the status of a previous generate call.

    #     Args:
    #         generation: The generation object containing call_id

    #     Returns:
    #         Updated generation object with status and results
    #     """
    #     print('CHECK GEN STATUS')
    #     try:
    #         # where generation['call_id'] is the operation object itself
    #         test_operation = types.Operation(name=generation['call_id'])
    #         operation = self.client.operations.get(test_operation)
    #     except Exception as e:
    #         print('operation not found: ', str(e))

    #     if operation.done:
    #         print('opereation done')
    #         try:
    #             from datetime import datetime
    #             results = []

    #             # Download generated videos
    #             for n, generated_video in enumerate(operation.response.generated_videos):
    #                 # Download video
    #                 self.client.files.download(file=generated_video.video)

    #                 # Save video to file
    #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                 video_filename = f"veo_output_{timestamp}_{n}.mp4"

    #                 # Setup output path
    #                 current_dir = os.path.dirname(os.path.abspath(__file__))
    #                 repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    #                 output_videos_dir = os.path.join(repo_root, "output_videos")
    #                 os.makedirs(output_videos_dir, exist_ok=True)
    #                 video_path = os.path.join(output_videos_dir, video_filename)

    #                 # Save video
    #                 generated_video.video.save(video_path)

    #                 results.append({
    #                     'filename': video_filename,
    #                     'path': video_path
    #                 })
    #             generation.update(status=JobStatus.COMPLETED, message='Video generation completed', result_video=results)
    #         except Exception as e:
    #             print("Error downloading veo2 generated video, ", str(e))
    #     else:
    #         print("still waiting")
    #         generation.update(timestamp=create_timestamp(), message='Video generation in progress')

registry.register(
    feature="image_to_video",
    model=MODEL_NAME,
    provider="local",
    implementation=Veo20Generate001
)

def base64_to_bytes(data_url):
    if data_url.startswith("data:"):
        header, b64data = data_url.split(",", 1)
    else:
        b64data = data_url
    return base64.b64decode(b64data)

# veo-2.0-generate-001
# veo-3.0-generate-preview
# print("Local generation")
# operation = client.models.generate_videos(
#     model="veo-3.0-generate-preview",
#     prompt="Panning wide shot of a purring kitten sleeping in the sunshine",
#     config=types.GenerateVideosConfig(
#         person_generation="allow_all",  # "allow_adult" and "dont_allow" for Veo 2 only
#         aspect_ratio="16:9",  # "16:9", and "9:16" for Veo 2 only
#     ),
# )

# while not operation.done:
#     time.sleep(20)
#     operation = client.operations.get(operation)

# for n, generated_video in enumerate(operation.response.generated_videos):
#     client.files.download(file=generated_video.video)
#     generated_video.video.save(f"video{n}.mp4")



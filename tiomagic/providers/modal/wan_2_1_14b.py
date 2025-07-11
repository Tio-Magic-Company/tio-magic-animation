from pathlib import Path
from typing import Any, Dict
import modal
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse
from .base import GPUType, ModalProviderBase
from ...core.jobs import JobStatus
from ...core.registry import registry
from ...core.utils import check_modal_app_deployment, Generation

import os
import requests

VOLUME_NAME = "test-wan-2.1-t2v-14b-cache"
CACHE_PATH = "/cache" 
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
OUTPUTS_NAME = "test-wan-2.1-t2v-14b-outputs"
OUTPUTS_PATH = "/outputs"
APP_NAME = 'test-wan-2.1-text-to-video-14b'

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "opencv-python>=4.9.0.80",
        "diffusers>=0.31.0",
        "transformers>=4.49.0",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg",
        "gradio>=5.0.0",
        "numpy>=1.23.5,<2",
        "fastapi[standard]",
    ).env({"HF_HUB_CACHE": CACHE_PATH})
)

cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)

# Create Modal app at module level
app = modal.App(
    APP_NAME,
)


@app.cls(
    image=image,
    gpu=GPUType.H100.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=600,
    scaledown_window=900,
)
class T2VModal:
    """
    Modal app implementation
    """
    @modal.enter()
    def load_models(self):
        """
        This method is called once when the container starts.
        It downloads and initializes the models and pipeline.
        """
        import torch
        from diffusers import AutoModel, WanPipeline
        from diffusers.hooks.group_offloading import apply_group_offloading
        from transformers import UMT5EncoderModel
        import time

        print("Loading models...")
        start_time = time.time()
        print(f"✅ {time.time() - start_time:.2f}s: Starting model load...")

        # Load the models
        try:
            print(f"✅ {time.time() - start_time:.2f}s: Loading text_encoder...")
            text_encoder = UMT5EncoderModel.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
            )            
            print(f"✅ {time.time() - start_time:.2f}s: text_encoder loaded.")

            print(f"✅ {time.time() - start_time:.2f}s: Loading VAE...")
            vae = AutoModel.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                subfolder="vae",
                torch_dtype=torch.float32,
            )
            # vae.to("cuda")
            print(f"✅ {time.time() - start_time:.2f}s: VAE loaded.")

            print(f"✅ {time.time() - start_time:.2f}s: Loading transformer...")
            transformer = AutoModel.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )            
            print(f"✅ {time.time() - start_time:.2f}s: Transformer loaded.")


            print(f"✅ {time.time() - start_time:.2f}s: Creating pipeline...")
            # Apply group-offloading for memory efficiency
            onload_device = torch.device("cuda")
            offload_device = torch.device("cpu")
            apply_group_offloading(
                text_encoder,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=4,
            )
            transformer.enable_group_offload(
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="leaf_level",
                use_stream=True,
            )

            # Create and run the pipeline
            self.pipeline = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                vae=vae,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=torch.bfloat16,
            )
            self.pipeline.to("cuda")
            print(f"✅ {time.time() - start_time:.2f}s: Pipeline ready. Models loaded successfully.")

        except Exception as e:
            print(f"❌ {time.time() - start_time:.2f}s: An error occurred: {e}")
            raise
        print("Models loaded successfully.")

    @modal.method()
    def generate(self, prompt: str, negative_prompt: str):
        """
        This method runs the text-to-video generation on an existing container.
        """
        from diffusers.utils import export_to_video
        from datetime import datetime

        print("Generating video...")
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=81,
            guidance_scale=5.0,
        ).frames[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp4_name = f"wan2.1-pipeline-output_{timestamp}.mp4"

        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output, mp4_path, fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    
    @modal.fastapi_endpoint(method="POST")
    # consider proxy auth token here
    def web_inference(self, data: dict = Body(...)):
        """
        FastAPI endpoint that runs on the class instance.
        """
        import io
        from datetime import datetime

        print("Enter async fastapi call")
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt", "") # Added default

        if not prompt:
            return {"error": "A 'prompt' is required."}
        
        print(f"prompt: {prompt}")
        print(f"negative prompt: {negative_prompt}")

        call = self.generate.spawn(prompt, negative_prompt)

        return JSONResponse({"call_id": call.object_id})

    @modal.fastapi_endpoint(method="GET")
    def get_result(self, call_id: str):
        """
        FastAPI endpoint to poll for the result of a generation call.
        """
        import io
        from datetime import datetime
        from modal import FunctionCall

        print(f"Polling for call_id: {call_id}")
        try:
            call = FunctionCall.from_id(call_id)
            video_bytes = call.get(timeout=0) # Use a short timeout to check for completion
        except TimeoutError:
            return JSONResponse({"status": "processing"}, status_code=202)
        except Exception as e:
            print(f"Error fetching result for {call_id}: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

        filename = f"wan2.1-output_{call_id}.mp4"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4", headers=headers)


# Wrapper class for local code to use
# class Wan21TextToVideo14B:
#     """Adapter class to interface with the Modal implementation."""
#     # Class variable to track if app is deployed
#     _is_app_deployed = False
#     _app_url = None

#     def __init__(self, api_key=None):
#         """Initialize the adapter for the Modal implementation."""
#         # No direct initialization needed
#         self.api_key = api_key
        
#         # File path for the current module
#         self.module_path = os.path.abspath(__file__)
        
#         # App name used in Modal
#         self.app_name = APP_NAME
        
#     def generate(self, prompt, **kwargs):
#         """Generate content using Modal.
        
#         This is the main entry point for generation that creates a Modal instance.
#         """
#         from json import JSONDecodeError
#         from datetime import datetime
#         print(f"Running {self.app_name} generate with prompt: {prompt}")
        
#         generation = Generation(
#             timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
#             prompt=prompt,
#             optional_parameters=kwargs
#         )

#         try:
#             # Extract the negative_prompt from kwargs if it exists
#             negative_prompt = kwargs.get('negative_prompt', '')

#             deployment_status = check_modal_app_deployment(app, self.app_name)
#             print("return status: ", deployment_status)
#             if not deployment_status['success']:
#                 generation.update(message=f"App '{self.app_name}' is not deployed: {deployment_status['message']}")
#                 return generation.to_dict()
#             if not deployment_status['endpoints']:
#                 generation.update(message=f"No endpoints found for app '{self.app_name}': {deployment_status['message']}")
#                 return generation.to_dict()
            
#             # show_url_function = modal.Function.from_name("test-wan-2.1-text-to-video-14b", "show_url")
#             # result = show_url_function.remote()

#             # remote_function = modal.Function.from_name(self.app_name, "T2VModal.show_url")
#             modal_class = modal.Cls.from_name(self.app_name, "T2VModal")
#             print("modal instance, ", modal_class)
#             web_inference_url = modal_class().web_inference.get_web_url()
#             print("web_url: ", web_inference_url)
#             get_result_url = modal_class().get_result.get_web_url()
#             print("get result url: ", get_result_url)

#             print("calling web inference")
#             try:
#                 print("enter try")
#                 response = requests.post(
#                     web_inference_url, 
#                     json = {
#                         "prompt": prompt,
#                         "negative_prompt": negative_prompt,
#                     },
#                     headers={"Content-Type": "application/json"},
#                 )
#                 print("response: ", response)

#                 if response.status_code != 200:
#                     generation.update(message=f"Error calling web_inference: {response.status_code}, {response.text}")
#                     return generation.to_dict()
                
#                 try:
#                     response_data = response.json()
#                     print(f"Web inference response: {response_data}")
#                 except JSONDecodeError:
#                     generation.update(message=f"Error parsing response as JSON: {response.text}")
#                     return generation.to_dict()
                
#                 if "call_id" not in response_data:
#                     generation.update(message="No call_id in response")
#                     # save_to_log(result.to_dict())
#                     return generation.to_dict()
                    
#                 call_id = response_data["call_id"]
#                 message = "generation queued. This may take a few minutes if it is the first time you are running this model"
#                 generation.update(call_id=call_id, status=JobStatus.QUEUED, message=message)
#                 print(f"Got call_id: {call_id}")


#             except requests.RequestException as e:
#                 print("request error")
#                 generation.update(message=f"Request error: {str(e)}")
#                 # save_to_log(result.to_dict())
#                 return generation.to_dict()

#             # print("test async web inference, ", modal_class().web_inference.remote(prompt=prompt, negative_prompt=negative_prompt))
#             # modal_class().web_inference.remote()

#             print("end of generate for now")
#             # save_to_log(result.to_dict())
#             return generation.to_dict()
#             # {call_id, time start, status: pending} => .txt file


#             # # Call the remote method directly
#             # # This is the key part - using .remote() on the generate method
#             # print("starting local generate")
#             # result = modal_instance.generate.remote(prompt=prompt, negative_prompt=negative_prompt)
#             # return result

            
#         except Exception as e:
#             print(f"Error in generate: {str(e)}")
#             generation.update(message=f"Error in generate: {str(e)}")
#             # save_to_log(result.to_dict())
#             raise

#     def check_generation_status(self, generation: Generation):
#         """
#         Check the status of a previous generate call
#         """
#         from modal.functions import FunctionCall
#         from datetime import datetime
#         from ...core.jobs import JobStatus
#         print("MODAL CHECK GENERATION STATUS")

#         try:
#             print("enter try", generation["call_id"])
#             fc = FunctionCall.from_id(generation["call_id"])
#             video_bytes = fc.get(timeout=0)

#             # Save video to file
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             video_filename = f"generated_video_{generation['call_id']}_{timestamp}.mp4"
            
#             # Get the directory of this file and save to the same directory
#             current_dir = os.path.dirname(os.path.abspath(__file__))
#             repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
#             video_path = os.path.join(repo_root, video_filename)
            
#             with open(video_path, 'wb') as f:
#                 f.write(video_bytes)
            
#             print(f"Video saved to: {video_path}")
#             # video bytes are returned
#             generation.update(status=JobStatus.COMPLETED, 
#                               timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
#                               result_video=video_path)
#         except TimeoutError:
#             print("NOT READY YET - MODAL CHECK GENERATION STATUS")
#             generation.update(status=JobStatus.RUNNING,
#                               timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
#         except Exception as e:
#             print("EXCEPTION IN MODAL CHECK GEN STATUS", e)
#             if e.args:
#                 print("e args", e.args)
#                 inner_e = e.args[0]
#                 if "HTTPError 403" in inner_e:
#                     generation.update(message="permission denied on video download")
#                 else:
#                     generation.update(status=JobStatus.CANCELED, message=inner_e)
#             else:
#                 generation.update(message=str(e))
            
        
#         return generation

class Wan21TextToVideo14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "T2VModal"
    def _prepare_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 model."""
        payload = super()._prepare_payload(prompt, **kwargs)
        
        # Add negative_prompt if provided
        negative_prompt = kwargs.get('negative_prompt', '')
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
            
        return payload

# Register with the system registry
registry.register(
    feature="text_to_video",
    model="wan2.1-t2v-14b",
    provider="modal",
    implementation=Wan21TextToVideo14B
)
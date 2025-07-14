import modal
import os
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse
import PIL.Image

from .base import GPUType, ModalProviderBase
from typing import Any, Dict
from ...core.jobs import JobStatus
from ...core.registry import registry
from ...core.utils import check_modal_app_deployment, Generation, prepare_video_and_mask, load_image_robust, is_local_path, local_image_to_base64

# --- Configuration ---
APP_NAME = "test-wan-2.1-vace-t2v-14b"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f'{APP_NAME}-outputs'
OUTPUTS_PATH = "/outputs"

VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-14B-diffusers"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.1",
        'imageio',
        'onnxruntime',
        'imageio-ffmpeg',
        'python-multipart',
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        'ftfy',
        "accelerate",
        'matplotlib',
        'onnxruntime-gpu',
        'loguru',
        'numpy',
        'tqdm',
        'omegaconf',
        "opencv-python-headless",
        "safetensors",
        "fastapi"
    ).env({"HF_HUB_CACHE": CACHE_PATH, "TOKENIZERS_PARALLELISM": "false"})
)

cache_volume = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)

app = modal.App(APP_NAME)

@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)
class WebAPI:
    @modal.fastapi_endpoint(method="POST")
    def web_inference(self, data: dict = Body(...)):
        """
        Unified FastAPI endpoint that routes to the appropriate class based on the request.
        """
        print("DATA OF WEB INFERENCE, ", data)
        feature_type = data.get("feature_type")  # "text_to_video", "image_to_video", or "interpolate"
        
        if not feature_type:
            return {"error": "A 'feature_type' is required. Must be one of: 'text_to_video', 'image_to_video', 'interpolate'"}
        
        # Route to appropriate class
        if feature_type == "text_to_video":
            return T2V.handle_web_inference(data)
        elif feature_type == "image_to_video":
            return I2V.handle_web_inference(data)
        elif feature_type == "interpolate":
            return Interpolate.handle_web_inference(data)
        else:
            return {"error": f"Unknown feature_type: {feature_type}. Must be one of: 'text_to_video', 'image_to_video', 'interpolate'"}

    @modal.fastapi_endpoint(method="GET")
    def get_result(self, call_id: str, feature_type: str = None):
        """
        Unified FastAPI endpoint to poll for results from any class.
        """
        import io
        from modal import FunctionCall
        
        print(f"Polling for call_id: {call_id}, feature_type: {feature_type}")
        
        try:
            call = FunctionCall.from_id(call_id)
            video_bytes = call.get(timeout=0)  # Use a short timeout to check for completion
        except TimeoutError:
            return JSONResponse({"status": "processing"}, status_code=202)
        except Exception as e:
            print(f"Error fetching result for {call_id}: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
        
        # Determine filename based on feature type
        if feature_type == "text_to_video":
            filename = f"wan2.1-vace-t2v-output_{call_id}.mp4"
        elif feature_type == "image_to_video":
            filename = f"wan2.1-vace-i2v-output_{call_id}.mp4"
        elif feature_type == "interpolate":
            filename = f"wan2.1-vace-interpolate-output_{call_id}.mp4"
        else:
            filename = f"wan2.1-vace-output_{call_id}.mp4"
        
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4", headers=headers)

@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)
class T2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)
        
        flow_shift = 3.0
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, prompt: str, 
                 negative_prompt: str,
                height: int = 768,
                width: int = 768,
                num_frames: int = 81,
                num_inference_steps: int = 50,
                guidance_scale: float = 5.0,
                 ):
        from datetime import datetime
        from diffusers.utils import export_to_video

        frames = self.pipe(prompt=prompt, 
                           negative_prompt=negative_prompt,
                           height=height,
                           width=width,
                           num_frames=num_frames,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale).frames[0]
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        mp4_name = f"wan-vace-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes 
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle text-to-video generation."""
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt", "")
        
        if not prompt:
            return {"error": "A 'prompt' is required."}
        
        print(f"text_to_video - prompt: {prompt}")
        print(f"text_to_video - negative prompt: {negative_prompt}")
        
        # Create T2V instance and call generate
        t2v_instance = T2V()
        call = t2v_instance.generate.spawn(prompt, negative_prompt)
        
        return JSONResponse({"call_id": call.object_id, "feature_type": "text_to_video"})

class Wan21VaceTextToVideo14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "T2V"
    def _prepare_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace model."""
        payload = super()._prepare_payload(prompt, **kwargs)
        
        # Add feature_type for routing
        payload["feature_type"] = "text_to_video"
        
        # Add negative_prompt if provided
        negative_prompt = kwargs.get('negative_prompt', '')
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
            
        return payload
    
@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)
class I2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)
        
        flow_shift = 5.0
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, image: PIL.Image.Image, prompt: str, negative_prompt: str):
        import torch
        import PIL.Image
        from diffusers.utils import export_to_video
        import io
        from datetime import datetime

        print("Starting video generation process...")
        # image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Define video parameters
        height = 480
        width = 832
        num_frames = 81
        fps = 16

        # Prepare the data for the pipeline
        video, mask = prepare_video_and_mask(image, height, width, num_frames)

        # Run the diffusion pipeline
        output_frames = self.pipe(
            video=video,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).frames[0]
        print("Pipeline execution finished.")

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        mp4_name = f"wan21-vace-i2v-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output_frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle image-to-video generation."""
        
        prompt = data.get("prompt")
        image = data.get("image")
        negative_prompt = data.get("negative_prompt", "")
        
        if not prompt:
            return {"error": "A 'prompt' is required."}
        if not image:
            return {"error": "An 'image' is required."}
        print("image: ", image)
        
        # print(f"image_to_video - prompt: {prompt}")
        # print(f"image_to_video - image: {image}")
        # print(f"image_to_video - negative prompt: {negative_prompt}")
        
        try:
            print("loading image to PIL.Image.Image format...")
            image = load_image_robust(image)
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}

        # Create I2V instance and call generate
        i2v_instance = I2V()
        call = i2v_instance.generate.spawn(image, prompt, negative_prompt)
        
        return JSONResponse({"call_id": call.object_id, "feature_type": "image_to_video"})

class Wan21VaceImageToVideo14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Image-to-Video model."""

        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = "image_to_video"
        
        # verify required arguments
        if payload['image'] is None:
            raise ValueError("Argument 'image' is required for Image to Video generation")
        
        if is_local_path(payload['image']):
            payload['image'] = local_image_to_base64(payload['image'])
        return payload
   
@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)
class Interpolate:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)
        
        flow_shift = 5.0
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, first_frame: PIL.Image.Image, last_frame: PIL.Image.Image, prompt: str, negative_prompt: str):
        from diffusers.utils import export_to_video
        from datetime import datetime
        import torch
        print("Starting video generation process...")

        height = 512
        width = 512
        num_frames = 81
        video, mask = prepare_video_and_mask(img=first_frame, height=height, width=width, num_frames=num_frames, last_frame=last_frame)

        output_frames = self.pipe(
            video=video,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=torch.Generator().manual_seed(42),
        ).frames[0]
        print("Pipeline execution finished.")

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        mp4_name = f"wan21-vace-interpolate-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output_frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle frame interpolation."""  

        prompt = data.get("prompt")
        first_frame = data.get("first_frame")
        last_frame = data.get("last_frame")
        negative_prompt = data.get("negative_prompt", "")
        
        if not prompt:
            return {"error": "A 'prompt' is required."}
        if not first_frame or not last_frame:
            return {"error": "Both 'first_frame' and 'last_frame' are required."}
        
        try:
            # load_image_robust can handle both URLs and base64 strings
            first_frame = load_image_robust(first_frame)
            last_frame = load_image_robust(last_frame)            
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}
        
        # print(f"interpolate - prompt: {prompt}")
        # print(f"interpolate - first frame: {first_frame}")
        # print(f"interpolate - last frame: {last_frame}")
        # print(f"interpolate - negative prompt: {negative_prompt}")
        
        # Create Interpolate instance and call generate
        interpolate_instance = Interpolate()
        call = interpolate_instance.generate.spawn(first_frame, last_frame, prompt, negative_prompt)
        
        return JSONResponse({"call_id": call.object_id, "feature_type": "interpolate"})

class Wan21VaceInterpolate14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "Interpolate"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Prepare payload specific to Wan2.1 Vace Interpolate model.
        Break out required args into payload
        """
        payload = super()._prepare_payload(required_args, **kwargs)
        # payload = {"prompt": required_args['prompt']}
        payload["feature_type"] = "interpolate"
        
        if payload['first_frame'] is None or payload['last_frame'] is None:
            raise ValueError("Arguments 'first_frame' and 'last_frame' are required for Interpolation Video generation")

        if is_local_path(payload['first_frame']):
            # Convert local image to base64
            payload["first_frame"] = local_image_to_base64(payload['first_frame'])
        if is_local_path(payload['last_frame']):
            # Convert local image to base64
            payload["last_frame"] = local_image_to_base64(payload['last_frame'])
    
        return payload

# Register with the system registry
registry.register(
    feature="text_to_video",
    model="wan2.1-vace-14b",
    provider="modal",
    implementation=Wan21VaceTextToVideo14B
)
registry.register(
    feature="image_to_video",
    model="wan2.1-vace-14b",
    provider="modal",
    implementation=Wan21VaceImageToVideo14B
)
registry.register(
    feature="interpolate",
    model="wan2.1-vace-14b",
    provider="modal",
    implementation=Wan21VaceInterpolate14B
)



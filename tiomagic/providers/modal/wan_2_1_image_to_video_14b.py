import modal
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core.utils import load_image_robust, is_local_path, local_image_to_base64, create_timestamp

APP_NAME = "test-wan-2.1-image-to-video-14b"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = Path("/cache")
OUTPUTS_NAME = f"{APP_NAME}-outputs"
OUTPUTS_PATH = Path("/outputs")

MODEL_ID = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"

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
        "fastapi",
    ).env({"HF_HUB_CACHE": CACHE_PATH})
)

cache_volume = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)

app = modal.App(APP_NAME)

@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume}, 
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2400, #40 minutes
    scaledown_window=900 #stay idle for 15 mins before scaling down
)
class WebAPI:
    @modal.fastapi_endpoint(method="POST")
    def web_inference(self, data: dict = Body(...)):
        """
        Unified FastAPI endpoint that routes to the appropriate class based on the request.
        """
        print("DATA OF WEB INFERENCE, ", data)
        feature_type = data.pop("feature_type", None)  # "text_to_video", "image_to_video", or "interpolate"
        
        if not feature_type:
            return {"error": "A 'feature_type' is required. Must be one of: 'text_to_video', 'image_to_video', 'interpolate'"}
        
        # Route to appropriate class
        if feature_type == "image_to_video":
            return I2V.handle_web_inference(data)
        elif feature_type == 'interpolate':
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
        if feature_type == "image_to_video":
            filename = f"wan2.1-vace-i2v-output_{call_id}.mp4"
        else:
            filename = f"wan2.1-vace-output_{call_id}.mp4"
        
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4", headers=headers)
class Interpolate:
    @modal.enter()
    def load_models(self):
        import numpy as np
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from transformers import CLIPVisionModel
        import time

        print("Loading models...")
        start_time = time.time()
        print(f"✅ {time.time() - start_time:.2f}s: Starting model load...")

        try:
            print(f"✅ {time.time() - start_time:.2f}s: Loading image_encoder...")
            image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
            print(f"✅ {time.time() - start_time:.2f}s: image_encoder loaded.")
            
            print(f"✅ {time.time() - start_time:.2f}s: Loading VAE...")
            vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
            print(f"✅ {time.time() - start_time:.2f}s: VAE loaded.")

            print(f"✅ {time.time() - start_time:.2f}s: Creating pipeline...")

            self.pipe = WanImageToVideoPipeline.from_pretrained(
                MODEL_ID, 
                vae=vae, 
                image_encoder=image_encoder, 
                torch_dtype=torch.bfloat16
            )
            self.pipe.to("cuda")
            print(f"✅ {time.time() - start_time:.2f}s: Pipeline ready. Models loaded successfully.")
        except Exception as e:
            print(f"❌ {time.time() - start_time:.2f}s: An error occurred: {e}")
            raise
        print("Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video

        first_frame = data.get('first_frame')
        last_frame = data.get('last_frame')
        # LOAD_IMAGE these frames
        height = data.get('height', 480)
        width = data.get('width', 832)

        data['first_frame'], height, width = self.aspect_ratio_resize(first_frame)
        if last_frame.size != data['first_frame'].size:
            data['last_frame'], _, _ = self.center_crop_resize(last_frame, height, width)
        output = self.pipe(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"wan21-i2v-14b-interpolate-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        prompt = data.get("prompt")
        first_frame = data.get("first_frame")
        last_frame = data.get("last_frame")
        
        if not prompt:
            return {"error": "A 'prompt' is required."}
        if not first_frame or not last_frame:
            return {"error": "Both 'first_frame' and 'last_frame' are required."}
        
        try:
            # load_image_robust can handle both URLs and base64 strings
            first_frame = load_image_robust(first_frame)
            last_frame = load_image_robust(last_frame)   
            data['first_frame'] = first_frame
            data['last_frame'] = last_frame         
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}
        
        # Create Interpolate instance and call generate
        interpolate_instance = Interpolate()
        call = interpolate_instance.generate.spawn(data)
        
        return JSONResponse({"call_id": call.object_id, "feature_type": "interpolate"})

class Wan21ImageToVideoInterpolate14b(ModalProviderBase):
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

# Create a subclass with the handlers
class WebAPI(GenericWebAPI):
    feature_handlers = {
        "interpolate": Interpolate
    }

# Apply Modal decorator
WebAPI = app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)(WebAPI)


registry.register(
    feature="interpolate",
    model="wan2.1-image-to-video-14b",
    provider="modal",
    implementation=Wan21ImageToVideoInterpolate14b
)
        
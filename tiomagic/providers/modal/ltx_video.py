import modal
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core.utils import load_image_robust, is_local_path, local_image_to_base64, create_timestamp

# --- Configuration ---
APP_NAME = "test-ltx-video-i2v"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f'{APP_NAME}-outputs'
OUTPUTS_PATH = "/outputs"

MODEL_ID = "Lightricks/LTX-Video"

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate==1.6.0",
        "diffusers==0.33.1",
        "hf_transfer==0.1.9",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.5.1",
        "sentencepiece==0.2.0",
        "torch==2.7.0",
        "transformers==4.51.3",
        "fastapi[standard]"
    )
    .env({"HF_HUB_CACHE": CACHE_PATH, "TOKENIZERS_PARALLELISM": "false"})
)

cache_volume = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)

app = modal.App(APP_NAME)
@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class I2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import LTXImageToVideoPipeline

        print("loading models...")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")

        print('models loaded')
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video

        try:
            print("Starting video generation process...")
            output_frames = self.pipe(**data).frames[0]
            print("Pipeline execution finished.")

            timestamp = create_timestamp()
            mp4_name = f"{APP_NAME}-output_{timestamp}.mp4"
            mp4_path = Path(OUTPUTS_PATH) / mp4_name
            export_to_video(output_frames, str(mp4_path))
            outputs_volume.commit()

            with open(mp4_path, "rb") as f:
                video_bytes = f.read()
            return video_bytes
        except Exception as e:
            print(f"Error generating {APP_NAME}: {str(e)}")
    @staticmethod
    def handle_web_inference(data: dict[str, Any]):
        prompt = data.get("prompt")
        image = data.get("image")
        
        if not prompt:
            return {"error": "A 'prompt' is required."}
        if not image:
            return {"error": "An 'image' is required."}
        try:
            image = load_image_robust(image)
            data['image'] = image
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}

        i2v_instance = I2V()
        call = i2v_instance.generate.spawn(data)
        return JSONResponse({"call_id": call.object_id, "feature_type": "image_to_video"})

class LTXVideoImageToVideo(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Image-to-Video model."""
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = "image_to_video"
        
        if is_local_path(payload['image']):
            payload['image'] = local_image_to_base64(payload['image'])
        return payload

# Create a subclass with the handlers
class WebAPI(GenericWebAPI):
    feature_handlers = {
        "image_to_video": I2V,
    }
# Apply Modal decorator
WebAPIClass = app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)(WebAPI)

registry.register(
    feature="image_to_video",
    model="ltx-video",
    provider="modal",
    implementation=LTXVideoImageToVideo
)


            

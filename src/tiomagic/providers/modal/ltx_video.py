import modal
from pathlib import Path
from fastapi.responses import JSONResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core._utils import load_image_robust, is_local_path, local_image_to_base64, create_timestamp, extract_image_dimensions
from ...core.constants import FeatureType
from ...core.errors import (
    DeploymentError, GenerationError, ProcessingError
)

# --- Configuration ---
APP_NAME = "test-ltx-video-i2v"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f'{APP_NAME}-outputs'
OUTPUTS_PATH = "/outputs"

MODEL_ID = "Lightricks/LTX-Video"
MODEL_REVISION_ID = "a6d59ee37c13c58261aa79027d3e41cd41960925"


GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .apt_install("python3-opencv")
    .pip_install(
        "accelerate==1.4.0",
        "diffusers==0.32.2",
        "fastapi[standard]==0.115.8",
        "huggingface-hub[hf_transfer]==0.29.1",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "opencv-python==4.11.0.86",
        "pillow==11.1.0",
        "sentencepiece==0.2.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.49.0",
        "fastapi[standard]"
    )
    .env({"HF_HUB_CACHE": CACHE_PATH, "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
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
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
            torch_dtype=torch.bfloat16,
        )
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
            mp4_name = f"{APP_NAME}-i2v-output_{timestamp}.mp4"
            mp4_path = Path(OUTPUTS_PATH) / mp4_name
            export_to_video(output_frames, str(mp4_path))
            outputs_volume.commit()

            with open(mp4_path, "rb") as f:
                video_bytes = f.read()

            return video_bytes
        except Exception as e:
            raise GenerationError(app_name=APP_NAME,
                                  model=MODEL_ID, 
                                  feature = FeatureType.IMAGE_TO_VIDEO,
                                  reason=str(e),
                                  generation_params=data,)
    @staticmethod
    def handle_web_inference(data: dict[str, Any]):
        image = data.get("image")
        
        try:
            image = load_image_robust(image)
            data['image'] = image
            if 'height' not in data and 'width' not in data:
                data = extract_image_dimensions(image, data)
        except Exception as e:
            raise ProcessingError(
                media_type="image",
                operation="load and process",
                reason=str(e),
                file_path=data.get("image") if isinstance(data.get("image"), str) else None
            )
        
        try:
            i2v_instance = I2V()
            call = i2v_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="Modal",
                reason=f"Failed to spawn I2V job: {str(e)}",
                app_name=APP_NAME
            )
        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.IMAGE_TO_VIDEO})

class LTXVideoImageToVideo(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Image-to-Video model."""
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.IMAGE_TO_VIDEO

        if is_local_path(payload['image']):
            payload['image'] = local_image_to_base64(payload['image'])
        return payload

# Create a subclass with the handlers
class WebAPI(GenericWebAPI):
    feature_handlers = {
        FeatureType.IMAGE_TO_VIDEO: I2V,
    }
# Apply Modal decorator
WebAPI = app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)(WebAPI)

registry.register(
    feature=FeatureType.IMAGE_TO_VIDEO,
    model="ltx-video",
    provider="modal",
    implementation=LTXVideoImageToVideo
)




import modal
from pathlib import Path
from fastapi.responses import JSONResponse

from ...core.errors import DeploymentError, ProcessingError

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core._utils import load_image_robust, is_local_path, local_image_to_base64, create_timestamp, extract_image_dimensions
from ...core.constants import FeatureType

APP_NAME = "test-cogvideox-5b-i2v"
CACHE_PATH = "/cache"
CACHE_NAME = f"{APP_NAME}-cache"
OUTPUTS_NAME = f"{APP_NAME}-outputs"
OUTPUTS_PATH = "/outputs"

COGVIDEOX_MODEL_ID = "THUDM/CogVideoX-5b-I2V"

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "diffusers>=0.32.1",
        "accelerate>=1.1.1",
        "transformers>=4.46.2",
        "numpy==1.26.0",
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "sentencepiece>=0.2.0",
        "SwissArmyTransformer>=0.4.12",
        "gradio>=5.5.0",
        "imageio>=2.35.1",
        "imageio-ffmpeg>=0.5.1",
        "openai>=1.54.0",
        "moviepy>=2.0.0",
        "scikit-video>=1.1.11",
        "pydantic>=2.10.3",
        "torchao"
    ).env({"HF_HUB_CACHE": CACHE_PATH})
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
        from diffusers import CogVideoXImageToVideoPipeline

        print("Loading models...")

        self.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        )
        self.pipeline.to("cuda")

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video

        frames = self.pipeline(
            **data,
            use_dynamic_cfg=True,
        ).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"{APP_NAME}-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=8)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes 
    @staticmethod
    def handle_web_inference(data: dict):
        image = data.get("image")

        try:
            image = load_image_robust(image)
            data['image'] = image
            if 'height' not in data and 'width' not in data:
                print("--> Warning: This model only supports 720 x 480 resolution")
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

class CogVideoX5BImageToVideo(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare specific payload."""
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
    model="cogvideox-5b-image-to-video",
    provider="modal",
    implementation=CogVideoX5BImageToVideo
)
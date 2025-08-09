import modal
from pathlib import Path
from fastapi.responses import JSONResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core._utils import load_image_robust, is_local_path, local_image_to_base64, create_timestamp, extract_image_dimensions
from ...core.constants import FeatureType
from ...core.errors import (
    DeploymentError, ProcessingError
)

APP_NAME = "wan-2.1-i2v-14b-720p"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f"{APP_NAME}-outputs"
OUTPUTS_PATH = "/outputs"

# https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers/tree/main
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # stay idle for 15 minutes before scaling down

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
        "peft",
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
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from transformers import CLIPVisionModel
        print("Loading models...")
        image_encoder = CLIPVisionModel.from_pretrained(
            MODEL_ID, 
            subfolder="image_encoder", 
            torch_dtype=torch.float32
        )
        vae = AutoencoderKLWan.from_pretrained(
            MODEL_ID, 
            subfolder="vae", 
            torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID, 
            vae=vae, 
            image_encoder=image_encoder, 
            torch_dtype=torch.bfloat16
        )
        
        # Completely remove any LoRA to ensure clean base model
        if hasattr(self.pipe, 'unload_lora_weights'):
            print("unloading lora weights")
            self.pipe.unload_lora_weights()

        self.pipe.to("cuda")
        # self.pipe.enable_model_cpu_offload()

        print("Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video

        output = self.pipe(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"{APP_NAME}-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: Dict[str, Any]):
        image = data.get("image")

        try:
            # load_image_robust can handle both URLs and base64 strings
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

class Wan21I2V14b720p(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 I2V 14b 720 model.
        Break out required args into payload
        """
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.IMAGE_TO_VIDEO

        if payload['image'] is None:
            raise ValueError("Argument 'image' is required for Image to Video generation")

        if is_local_path(payload['image']):
            # Convert local image to base64
            payload["image"] = local_image_to_base64(payload['image'])
        return payload


# Create a subclass with the handlers
class WebAPI(GenericWebAPI):
    feature_handlers = {
        FeatureType.IMAGE_TO_VIDEO: I2V
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
    model="wan2.1-i2v-14b-720p",
    provider="modal",
    implementation=Wan21I2V14b720p
)
import modal
from pathlib import Path
from fastapi.responses import JSONResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core._utils import load_image_robust, is_local_path, local_image_to_base64, create_timestamp, extract_image_dimensions
from ...core.constants import FeatureType
from ...core.schemas import FEATURE_SCHEMAS
from ...core.errors import (
    DeploymentError, ProcessingError
)


APP_NAME = "wan-2.1-flf2v-14b-720p"
MODEL_NAME = "wan2.1-flf2v-14b-720p"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f"{APP_NAME}-outputs"
OUTPUTS_PATH = "/outputs"

MODEL_ID = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"

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
        "fastapi",
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
class Interpolate:
    @modal.enter()
    def load_models(self):
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
    def aspect_ratio_resize(self, image, max_area=720 * 1280):
        import numpy as np
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image, height, width

    def center_crop_resize(self, image, height, width):
        import torchvision.transforms.functional as TF

        # Calculate resize ratio to match first frame dimensions
        resize_ratio = max(width / image.width, height / image.height)

        # Resize the image
        width = round(image.width * resize_ratio)
        height = round(image.height * resize_ratio)
        size = [width, height]
        image = TF.center_crop(image, size)

        return image, height, width
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        interpolate_schema = FEATURE_SCHEMAS["interpolate"][MODEL_NAME]

        first_frame = data.get('first_frame')
        last_frame = data.get('last_frame')
        # LOAD_IMAGE these frames
        height = data.get('height', interpolate_schema["optional"]["height"]["default"])
        width = data.get('width', interpolate_schema["optional"]["height"]["default"])

        data['first_frame'], height, width = self.aspect_ratio_resize(first_frame)
        if last_frame.size != data['first_frame'].size:
            data['last_frame'], _, _ = self.center_crop_resize(last_frame, height, width)
        data['image'] = data.pop('first_frame')
        data['last_image'] = data.pop('last_frame')
        output = self.pipe(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-interpolate-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        first_frame = data.get("first_frame")
        last_frame = data.get("last_frame")

        try:
            # load_image_robust can handle both URLs and base64 strings
            first_frame = load_image_robust(first_frame)
            last_frame = load_image_robust(last_frame)   
            data['first_frame'] = first_frame
            data['last_frame'] = last_frame
            if 'height' not in data and 'width' not in data:
                data = extract_image_dimensions(first_frame, data)
        except Exception as e:
            raise ProcessingError(
                media_type="image",
                operation="load and process",
                reason=str(e),
                file_path=data.get("image") if isinstance(data.get("image"), str) else None
            )

        # Create Interpolate instance and call generate
        try:
            interpolate_instance = Interpolate()
            call = interpolate_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="Modal",
                reason=f"Failed to spawn Interpolate job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.INTERPOLATE})

class Wan21FlfvInterpolate14b720p(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "Interpolate"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Interpolate model.
        Break out required args into payload
        """
        payload = super()._prepare_payload(required_args, **kwargs)
        # payload = {"prompt": required_args['prompt']}
        payload["feature_type"] = FeatureType.INTERPOLATE

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
        FeatureType.INTERPOLATE: Interpolate
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
    feature=FeatureType.INTERPOLATE,
    model="wan2.1-flf2v-14b-720p",
    provider="modal",
    implementation=Wan21FlfvInterpolate14b720p
)

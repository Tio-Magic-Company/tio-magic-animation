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

APP_NAME = "framepack-i2v-hy"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f'{APP_NAME}-outputs'
OUTPUTS_PATH = "/outputs"

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 3000 # 50 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "torch==2.4", 
        "transformers", 
        "accelerate", 
        "safetensors",
        "sentencepiece", 
        "av", 
        "einops", 
        "Pillow", 
        "numpy",
        "imageio[ffmpeg]",
        "fastapi[standard]",
    )
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
        from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        from transformers import SiglipImageProcessor, SiglipVisionModel

        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
        )
        self.pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda")

        self.pipe.vae.enable_tiling()
        print("✅ Complete Framepack pipeline loaded and ready.")

    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        frames = self.pipe(
            **data,
            sampling_type="inverted_anti_drifting").frames[0]

        timestamp = create_timestamp()
        mp4_name = f"framepack-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes 
    @staticmethod
    def handle_web_inference(data: Dict[str, Any]): 
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

        # Create I2V instance and call generate
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
class FramepackI2VHYImageToVideo(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Image-to-Video model."""
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.IMAGE_TO_VIDEO

        if is_local_path(payload['image']):
            payload['image'] = local_image_to_base64(payload['image'])
        return payload

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
        from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        from transformers import SiglipImageProcessor, SiglipVisionModel

        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
        )
        self.pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )

        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        print("✅ Complete Framepack pipeline loaded and ready.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video

        # adjust toolkit requirements to match framepack params
        data['image'] = data.pop('first_frame')
        data['last_image'] = data.pop('last_frame')

        frames = self.pipe(
            **data,
            sampling_type="inverted_anti_drifting").frames[0]

        timestamp = create_timestamp()
        mp4_name = f"framepack-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes 
    @staticmethod
    def handle_web_inference(data: Dict[str, Any]):
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
class FramepackI2VHYInterpolate(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "Interpolate"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Framepack I2V HY Interpolate model.
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
        FeatureType.INTERPOLATE: Interpolate,
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
    feature=FeatureType.INTERPOLATE,
    model="framepack-i2v-hy",
    provider="modal",
    implementation=FramepackI2VHYInterpolate
)
registry.register(
    feature=FeatureType.IMAGE_TO_VIDEO,
    model="framepack-i2v-hy",
    provider="modal",
    implementation=FramepackI2VHYImageToVideo
)


import modal
from pathlib import Path
from fastapi.responses import JSONResponse

from ...core.schemas import FEATURE_SCHEMAS

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

# MODEL_ID = "Lightricks/LTX-Video"
# MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
MODEL_ID = "Lightricks/LTX-Video-0.9.7-dev"
MODEL_UPSCALER = "Lightricks/ltxv-spatial-upscaler-0.9.7"

MODEL_REVISION_ID = "a6d59ee37c13c58261aa79027d3e41cd41960925"


GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .apt_install("python3-opencv")
    .pip_install(
        "accelerate",
        "git+https://github.com/huggingface/diffusers.git",
        "huggingface-hub[hf_transfer]",
        "imageio",
        "imageio-ffmpeg",
        "opencv-python",
        "pillow",
        "sentencepiece",
        "torch",
        "torchvision",
        "transformers",
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
        from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
#             revision=MODEL_REVISION_ID,

        print("loading models...")
        self.pipe = LTXConditionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            MODEL_UPSCALER, 
            vae=self.pipe.vae, 
            torch_dtype=torch.bfloat16)

        self.pipe.to("cuda")
        self.pipe_upsample.to("cuda")
        self.pipe.vae.enable_tiling()

        print('models loaded')
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from diffusers.utils import export_to_video, load_video

        try:
            print("Starting video generation process...")
            #  compress image to video
            video = load_video(export_to_video([data['image']]))
            condition1 = LTXVideoCondition(video=video, frame_index=0)
            downscale_factor = 2/3

            prompt = data.get('prompt')
            negative_prompt = data.get('negative_prompt')
            num_frames = data.get('num_frames')
            num_inference_steps = data.get('num_inference_steps')
            expected_width = data.get('width')
            expected_height = data.get('height')

            # generate video at smaller resolution
            downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
            downscaled_height, downscaled_width = self.round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
            latents = self.pipe(
                conditions=[condition1],
                width=downscaled_width,
                height=downscaled_height,
                prompt=prompt,
                negative_prompt = negative_prompt,
                num_frames = num_frames,
                num_inference_steps = num_inference_steps,
                generator=torch.Generator().manual_seed(0),
                output_type="latent"
            ).frames

            # upscale generated video using latent upsampler with fewer inference steps
            # the available latent upsampler upscales the height/width by 2x
            upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
            upscaled_latents = self.pipe_upsample(
                latents=latents,
                output_type="latent"
            ).frames

            # denoise the upscaled video with few steps to improve texture
            video = self.pipe(
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=upscaled_width,
                height=upscaled_height,
                num_frames=num_frames,
                denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
                num_inference_steps=10,
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=torch.Generator().manual_seed(0),
                output_type="pil",
            ).frames[0]
            
            # downscale video to expected resolution
            output_frames = [frame.resize((expected_width, expected_height)) for frame in video]
            
            # output_frames = self.pipe(**data).frames[0]
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
    def round_to_nearest_resolution_acceptable_by_vae(self, height, width):
        height = height - (height % self.pipe.vae_spatial_compression_ratio)
        width = width - (width % self.pipe.vae_spatial_compression_ratio)
        return height, width
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

        i2v_schema = FEATURE_SCHEMAS[FeatureType.IMAGE_TO_VIDEO]['ltx-video']
        payload['negative_prompt'] = payload.get('negative_prompt', "")
        payload['num_frames'] = payload.get('num_frames', i2v_schema["optional"]["num_frames"]["default"])
        payload['num_inference_steps'] = payload.get('num_inference_steps', i2v_schema["optional"]["num_inference_steps"]["default"])
        payload['width'] = payload.get('width', i2v_schema["optional"]["width"]["default"])
        payload['height'] = payload.get('height', i2v_schema["optional"]["height"]["default"])


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




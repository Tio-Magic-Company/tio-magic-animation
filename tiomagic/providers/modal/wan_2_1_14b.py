import modal
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core.utils import create_timestamp
from ...core.feature_types import FeatureType

VOLUME_NAME = "test-wan-2.1-t2v-14b-cache"
CACHE_PATH = "/cache" 
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
OUTPUTS_NAME = "test-wan-2.1-t2v-14b-outputs"
OUTPUTS_PATH = "/outputs"
APP_NAME = 'test-wan-2.1-text-to-video-14b'

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

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
app = modal.App(APP_NAME)

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class T2V:
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
    def generate(self, data: Dict[str, Any]):
        """
        This method runs the text-to-video generation on an existing container.
        """
        from diffusers.utils import export_to_video

        print("Generating video...")
        output = self.pipeline(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"wan2.1-pipeline-output_{timestamp}.mp4"

        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output, mp4_path, fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes

class Wan21TextToVideo14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "T2V"
    def _prepare_payload(self, required_args, **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 model."""
        payload = super()._prepare_payload(required_args, **kwargs)
        
        # Add feature_type for routing
        payload["feature_type"] = FeatureType.TEXT_TO_VIDEO
            
        return payload

class WebAPI(GenericWebAPI):
    feature_handlers = {
        FeatureType.TEXT_TO_VIDEO: T2V
    }
WebAPI = app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)(WebAPI)

# Register with the system registry
registry.register(
    feature=FeatureType.TEXT_TO_VIDEO,
    model="wan2.1-t2v-14b",
    provider="modal",
    implementation=Wan21TextToVideo14B
)
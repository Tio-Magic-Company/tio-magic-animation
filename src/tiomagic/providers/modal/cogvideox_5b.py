import modal
from pathlib import Path
from fastapi.responses import JSONResponse

from ...core.errors import DeploymentError

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core._utils import create_timestamp
from ...core.constants import FeatureType

# --- Configuration ---
APP_NAME = "cogvideox-5b"
CACHE_PATH = "/cache"
CACHE_NAME = f"{APP_NAME}-cache"
OUTPUTS_NAME = f"{APP_NAME}-outputs"
OUTPUTS_PATH = "/outputs"

COGVIDEOX_MODEL_ID = "THUDM/CogVideoX-5b"

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
class T2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import CogVideoXPipeline, AutoModel
        from diffusers.quantizers import PipelineQuantizationConfig
        from diffusers.quantizers.quantization_config import TorchAoConfig

        print("Loading models...")

        pipeline_quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig(quant_type="int8wo")
            }
        )

        # fp8 layerwise weight-casting
        transformer = AutoModel.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )

        self.pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            transformer=transformer,
            quantization_config=pipeline_quant_config,
            torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

        # model-offloading
        self.pipe.enable_model_cpu_offload()
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        frames = self.pipe(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"cogvideox-5b-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: Dict[str, Any]):
        """handle text-to-video generation"""
        try:
            t2v_instance = T2V()
            call = t2v_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="Modal",
                reason=f"Failed to spawn T2V job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.TEXT_TO_VIDEO})

class CogVideoXTextToVideo5B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "T2V"
    def _prepare_payload(self, required_args, **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace model."""
        payload = super()._prepare_payload(required_args, **kwargs)

        # Add feature_type for routing
        payload["feature_type"] = FeatureType.TEXT_TO_VIDEO

        return payload

# Create a subclass with the handlers
class WebAPI(GenericWebAPI):
    feature_handlers = {
        FeatureType.TEXT_TO_VIDEO: T2V,
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
    feature=FeatureType.TEXT_TO_VIDEO,
    model="cogvideox-5b",
    provider="modal",
    implementation=CogVideoXTextToVideo5B,
)
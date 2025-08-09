from typing import Any, Dict
from fastapi.responses import JSONResponse
import modal
from pathlib import Path
from PIL import Image

from ...core._utils._modal_helpers import load_image_robust
from ...core.registry import registry
from ...core.constants import FeatureType
from ...core.errors import DeploymentError, ProcessingError
from ...core._utils._utils import create_timestamp, is_local_path, local_image_to_base64
from .base import GPUType, GenericWebAPI, ModalProviderBase


# --- Configuration ---
APP_NAME = "pusa-v1"
CACHE_PATH = "/cache"
CACHE_NAME = f"{APP_NAME}-cache"
OUTPUTS_NAME = f"{APP_NAME}-outputs"
OUTPUTS_PATH = "/outputs"
MODEL_NAME = "pusa-v1"


GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "cupy-cuda12x",
        "transformers==4.46.2",
        "controlnet-aux==0.0.7",
        "imageio",
        "imageio[ffmpeg]",
        "safetensors",
        "einops",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "xfuser>=0.4.3",
        "absl-py",
        "peft",
        "lightning",
        "pandas",
        "deepspeed",
        "wandb",
        "av",
        "diffusers",
        "accelerate",
        "numpy",
        "pillow",
        "tqdm",
        "safetensors",
        "huggingface-hub",
        "fastapi[standard]",
        "modelscope",
    ).env({"HF_HUB_CACHE": CACHE_PATH}))

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
    def setup(self):
        """Initialize - download code and models if needed, then load"""
        import sys
        import subprocess
        from pathlib import Path
        
        # Define paths in the volume
        repo_path = Path("/cache/Pusa-VidGen")  # The cloned repository
        code_path = repo_path / "PusaV1"  # The actual PusaV1 code is in this subfolder
        models_path = Path("/cache/models/PusaV1")
        
        # Step 1: Download/update Pusa-VidGen repo if needed
        if not repo_path.exists():
            print("Cloning Pusa-VidGen repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/Yaofang-Liu/Pusa-VidGen.git",
                str(repo_path)
            ], check=True)
            cache_volume.commit()
        else:
            print("Pusa-VidGen repository already exists in volume")
            # Optional: Update the code
            # subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)
        
        # Step 2: Install PusaV1 from the subfolder
        print("Installing PusaV1...")
        if not code_path.exists():
            raise FileNotFoundError(f"PusaV1 directory not found at {code_path}")
        
        try:
            # Install from the PusaV1 subfolder
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(code_path)], check=True)
            print("Installation successful!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Pip install failed with exit code {e.returncode}")
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
            # Don't fail completely - try to continue by adding to path
            print("\n⚠️ Attempting to continue without pip install...")

        # Add both the repo and PusaV1 folder to Python path
        sys.path.insert(0, str(repo_path))
        sys.path.insert(0, str(code_path))
        
        # Step 3: Download models if needed
        wan_model_path = models_path / "Wan2.1-T2V-14B"
        pusa_checkpoint = models_path / "pusa_v1.pt"
        
        if not pusa_checkpoint.exists() or not wan_model_path.exists():
            print("Downloading models...")
            self._download_models(models_path)
        else:
            print("Models already exist in volume")
        
        # Step 4: Load models
        print("Loading models...")
        self._load_models(models_path)
        print("Setup complete!")
    
    def _download_models(self, models_path: Path):
        """Download model files to volume"""
        import subprocess
        import os
        
        models_path.mkdir(parents=True, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN", "")
        
        # Download PusaV1 specific files
        print("Downloading PusaV1 models...")
        subprocess.run([
            "hf", "download", 
            "RaphaelLiu/PusaV1", 
            "--local-dir", str(models_path),
            "--token", hf_token
        ], check=True)
        
        # Download base Wan model if it doesn't exist
        wan_path = models_path / "Wan2.1-T2V-14B"
        if not wan_path.exists():
            print("Downloading Wan2.1-T2V-14B...")
            subprocess.run([
                "hf", "download",
                "Wan-AI/Wan2.1-T2V-14B",
                "--local-dir", str(wan_path),
                "--token", hf_token
            ], check=True)
        
        # Commit changes to volume
        cache_volume.commit()
        print("Models downloaded and saved to volume!")
    
    def _load_models(self, models_path: Path):
        """Load models from volume paths"""
        import os
        import torch
        from diffsynth import ModelManager, WanVideoPusaPipeline
        
        # Initialize model manager with just device
        self.model_manager = ModelManager(device="cuda")
        
        # Set up paths
        wan_base_dir = models_path / "Wan2.1-T2V-14B"
        
        # Get all .safetensors files
        model_files = sorted([
            str(wan_base_dir / f) 
            for f in os.listdir(wan_base_dir) 
            if f.endswith('.safetensors')
        ])
        
        print(f"Found {len(model_files)} safetensors files")
        
        # Load all model files
        self.model_manager.load_models(
            [
                model_files,  # All safetensors files
                str(wan_base_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
                str(wan_base_dir / "Wan2.1_VAE.pth"),
            ],
            torch_dtype=torch.bfloat16,
        )
        
        print("Base models loaded successfully")
        
        # Load PusaV1 as a LoRA
        pusa_checkpoint = models_path / "pusa_v1.pt"
        if not pusa_checkpoint.exists():
            raise FileNotFoundError(f"PusaV1 checkpoint not found at {pusa_checkpoint}")
        
        print(f"Loading PusaV1 LoRA from: {pusa_checkpoint}")
        self.model_manager.load_lora(
            str(pusa_checkpoint), 
            lora_alpha=1.0  # You might want to make this configurable
        )
        
        # Create pipeline
        self.pipe = WanVideoPusaPipeline.from_model_manager(
            self.model_manager,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        
        frames = self.pipe(**data)

        timestamp = create_timestamp()
        mp4_name = f"{APP_NAME}-output_{timestamp}.mp4"
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

class PusaV1TextToVideo(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "T2V"
    def _prepare_payload(self, required_args, **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Pusa V1 model."""
        payload = super()._prepare_payload(required_args, **kwargs)

        # Add feature_type for routing
        payload["feature_type"] = FeatureType.TEXT_TO_VIDEO

        # print("payload: ", payload)
        return payload


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
    def setup(self):
        """Initialize - download code and models if needed, then load"""
        import sys
        import subprocess
        from pathlib import Path
        
        # Define paths in the volume
        repo_path = Path("/cache/Pusa-VidGen")  # The cloned repository
        code_path = repo_path / "PusaV1"  # The actual PusaV1 code is in this subfolder
        models_path = Path("/cache/models/PusaV1")
        
        # Step 1: Download/update Pusa-VidGen repo if needed
        if not repo_path.exists():
            print("Cloning Pusa-VidGen repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/Yaofang-Liu/Pusa-VidGen.git",
                str(repo_path)
            ], check=True)
            cache_volume.commit()
        else:
            print("Pusa-VidGen repository already exists in volume")
            # Optional: Update the code
            # subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)
        
        # Step 2: Install PusaV1 from the subfolder
        print("Installing PusaV1...")
        if not code_path.exists():
            raise FileNotFoundError(f"PusaV1 directory not found at {code_path}")
        
        try:
            # Install from the PusaV1 subfolder
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(code_path)], check=True)
            print("Installation successful!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Pip install failed with exit code {e.returncode}")
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
            # Don't fail completely - try to continue by adding to path
            print("\n⚠️ Attempting to continue without pip install...")

        # Add both the repo and PusaV1 folder to Python path
        sys.path.insert(0, str(repo_path))
        sys.path.insert(0, str(code_path))
        
        # Step 3: Download models if needed
        wan_model_path = models_path / "Wan2.1-T2V-14B"
        pusa_checkpoint = models_path / "pusa_v1.pt"
        
        if not pusa_checkpoint.exists() or not wan_model_path.exists():
            print("Downloading models...")
            self._download_models(models_path)
        else:
            print("Models already exist in volume")
        
        # Step 4: Load models
        print("Loading models...")
        self._load_models(models_path)
        print("Setup complete!")
    
    def _download_models(self, models_path: Path):
        """Download model files to volume"""
        import subprocess
        import os
        
        models_path.mkdir(parents=True, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN", "")
        
        # Download PusaV1 specific files
        print("Downloading PusaV1 models...")
        subprocess.run([
            "hf", "download", 
            "RaphaelLiu/PusaV1", 
            "--local-dir", str(models_path),
            "--token", hf_token
        ], check=True)
        
        # Download base Wan model if it doesn't exist
        wan_path = models_path / "Wan2.1-T2V-14B"
        if not wan_path.exists():
            print("Downloading Wan2.1-T2V-14B...")
            subprocess.run([
                "hf", "download",
                "Wan-AI/Wan2.1-T2V-14B",
                "--local-dir", str(wan_path),
                "--token", hf_token
            ], check=True)
        
        # Commit changes to volume
        cache_volume.commit()
        print("Models downloaded and saved to volume!")
    
    def _load_models(self, models_path: Path):
        """Load models from volume paths"""
        import os
        import torch
        from diffsynth import ModelManager, PusaMultiFramesPipeline
        
        # Initialize model manager with just device
        self.model_manager = ModelManager(device="cuda")
        
        # Set up paths
        wan_base_dir = models_path / "Wan2.1-T2V-14B"
        
        # Get all .safetensors files
        model_files = sorted([
            str(wan_base_dir / f) 
            for f in os.listdir(wan_base_dir) 
            if f.endswith('.safetensors')
        ])
        
        print(f"Found {len(model_files)} safetensors files")
        
        # Load all model files
        self.model_manager.load_models(
            [
                model_files,  # All safetensors files
                str(wan_base_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
                str(wan_base_dir / "Wan2.1_VAE.pth"),
            ],
            torch_dtype=torch.bfloat16,
        )
        
        print("Base models loaded successfully")
        
        # Load PusaV1 as a LoRA
        pusa_checkpoint = models_path / "pusa_v1.pt"
        if not pusa_checkpoint.exists():
            raise FileNotFoundError(f"PusaV1 checkpoint not found at {pusa_checkpoint}")
        
        print(f"Loading PusaV1 LoRA from: {pusa_checkpoint}")
        self.model_manager.load_lora(
            str(pusa_checkpoint), 
            lora_alpha=1.4  # You might want to make this configurable
        )
        
        # Create pipeline
        self.pipe = PusaMultiFramesPipeline.from_model_manager(
            self.model_manager,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video

        image = data.pop('image', None) #Expect PIL Image
        
        # Convert to RGB mode and resize to 1280x720 using LANCZOS resampling
        if image is not None:
            image = image.convert("RGB")
            image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        
        cond_position = data.pop("cond_position", None)
        if cond_position is None:
            cond_position = 0
        noise_multipliers = data.pop('noise_multipliers', None)
        if noise_multipliers is None:
            noise_multipliers = 0.0
        # Create multi_frame_images with just the first frame
        multi_frame_images = {
            cond_position: (image, noise_multipliers)  # Frame 0: use start image with no noise
        }
        data['multi_frame_images'] = multi_frame_images
        
        frames = self.pipe(**data)

        timestamp = create_timestamp()
        mp4_name = f"{APP_NAME}-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle image-to-video generation."""
        print("***MODAL HANDLE WEB INFERENCE METHOD***")
        image = data.get("image")
        
        try:
            image = load_image_robust(image)
            data['image'] = image
            # if 'height' not in data and 'width' not in data:
            #     data = extract_image_dimensions(image, data)
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
class PusaV1ImageToVideo(ModalProviderBase):
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
        FeatureType.TEXT_TO_VIDEO: T2V,
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
    feature=FeatureType.TEXT_TO_VIDEO,
    model=MODEL_NAME,
    provider="modal",
    implementation=PusaV1TextToVideo,
)
registry.register(
    feature=FeatureType.IMAGE_TO_VIDEO,
    model=MODEL_NAME,
    provider="modal",
    implementation=PusaV1ImageToVideo,
)




from fastapi import Body
from fastapi.responses import JSONResponse
import modal
from pathlib import Path

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core._utils import create_timestamp
from ...core.constants import FeatureType
from ...core.errors import (
    APIError,
    DeploymentError
)

"""Configuration Section.
    - Defines
        - Modal app name
        - model ID from HuggingFace
        - cache path to store model data on Modal Volume
        - outputs path to store video outputs on Modal Volume
        - GPU requirements
"""
VOLUME_NAME = "wan-2.1-t2v-14b-cache"
CACHE_PATH = "/cache" 
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
OUTPUTS_NAME = "wan-2.1-t2v-14b-outputs"
OUTPUTS_PATH = "/outputs"
APP_NAME = 'wan-2.1-text-to-video-14b'

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

"""Modal Image Definition.
    - Creates a container image with all required depencies (requirements.txt)
    - This image is cached and reused across function calls
"""
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
    """Modal app implementation
    """
    @modal.enter()
    def load_models(self):
        """This method is called once when the container starts.
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
                        # Completely remove any LoRA to ensure clean base model
            if hasattr(self.pipeline, 'unload_lora_weights'):
                print("unloading lora weights")
                self.pipeline.unload_lora_weights()

            self.pipeline.to("cuda")
            # self.pipeline.enable_model_cpu_offload()
            print(f"✅ {time.time() - start_time:.2f}s: Pipeline ready. Models loaded successfully.")

        except Exception as e:
            raise APIError(response_body=str(e))
        print("Models loaded successfully.")

    @modal.method()
    def generate(self, data: Dict[str, Any]):
        """This method runs the text-to-video generation on an existing container.
        Executes the actual video generating using the loaded pipeline.
        Saves the generated video to Modal's volume storage.
        Returns the video as bytes.
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
    @staticmethod
    def handle_web_inference(data: dict):
        """Creates a Modal instance and spawns async job
        Returns immediately with call_id, stored for user"""
        # Create T2V instance and call generate
        try:
            t2v_instance = T2V()
            call = t2v_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="modal",
                reason=f"Failed to spawn T2V job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.TEXT_TO_VIDEO})
    

class Wan21TextToVideo14B(ModalProviderBase):
    def __init__(self, api_key=None):
        """Initializes the local client that will communicate with Modal.
        Sets references to the Modal app and class names
        """
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "T2V"
    def _prepare_payload(self, required_args, **kwargs) -> Dict[str, Any]:
        """Prepares the data to send to Modal, specific to this model.
        Adds the feature_type field for routing.
        Inherits base validation from parent class
        """
        payload = super()._prepare_payload(required_args, **kwargs)

        # Add feature_type for routing
        payload["feature_type"] = FeatureType.TEXT_TO_VIDEO

        return payload

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class FusionXT2V:
    @modal.enter()
    def load_models(self):
        import torch        
        from diffusers import AutoModel, WanPipeline
        from diffusers.hooks.group_offloading import apply_group_offloading
        from transformers import UMT5EncoderModel

        print("Loading models into GPU memory...")
        text_encoder = UMT5EncoderModel.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
            )  
        vae = AutoModel.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                subfolder="vae",
                torch_dtype=torch.bfloat16,
            )
        transformer = AutoModel.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
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
        self.pipeline.vae.to(device="cuda")
        print("loading lora weights, enabling")
        # Load T2V FusionX
        self.pipeline.load_lora_weights("vrgamedevgirl84/Wan14BT2VFusioniX", weight_name="FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors", adapter_names="T2V FusionX")
        self.pipeline.enable_lora()

        # self.pipeline.to("cuda")
        # self.pipeline.enable_model_cpu_offload()   

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        # for every value in data, pass into pipe
        frames = self.pipeline(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"Wan2.1-T2V-14B-Diffusers-fusionx-t2v-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes 
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle text-to-video generation."""
        # Create T2V instance and call generate
        try:
            t2v_instance = FusionXT2V()
            call = t2v_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="modal",
                reason=f"Failed to spawn T2V job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.TEXT_TO_VIDEO})
    
class Wan2114bTextToVideoFusionX(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "FusionXT2V"
    def _prepare_payload(self, required_args, **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan FusionX model."""
        payload = super()._prepare_payload(required_args, **kwargs)

        # Add feature_type for routing
        payload["feature_type"] = FeatureType.TEXT_TO_VIDEO
        payload["model"] = "wan2.1-14b-t2v-fusionx"

        return payload

class WebAPI(GenericWebAPI):
    """Establishes features this model handles"""
    feature_handlers = {
        FeatureType.TEXT_TO_VIDEO: T2V
    }
    @modal.fastapi_endpoint(method="POST")
    def web_inference(self, data: dict = Body(...)):
        """Only implemented for models with LoRAs.
        Routing to LoRA-variant of model
        """

        feature_type = data.pop("feature_type", None)
        model = data.pop("model", None)  # Extract model from data
        if not feature_type:
            return {"error": f"A 'feature_type' is required. Must be one of: {list(self.feature_handlers.keys())}"}

        if feature_type not in self.feature_handlers:
            return {"error": f"Unknown feature_type: {feature_type}. Must be one of: {list(self.feature_handlers.keys())}"}

        # Route to appropriate class based on model and feature_type
        if feature_type == FeatureType.TEXT_TO_VIDEO and model == "wan2.1-14b-t2v-fusionx":
            # Route to FusionT2V for the fusionx model
            handler_class = FusionXT2V
        else:
            # Route to default handler
            handler_class = self.feature_handlers[feature_type]
       
        # If user specifies gpu, timeout, scaledown window, applied here
        if 'modal_options' in data and data['modal_options']:
            modal_options = data['modal_options']
            for key, value in modal_options.items():
                if value is not None:
                    print(f"Applying modal option: {key} = {value}")
                    handler_class = handler_class.with_options(**{key: value})
            data.pop('modal_options')
        return handler_class.handle_web_inference(data)

"""Transforms the WebAPI class into a Modal-compatible class"""
WebAPI = app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)(WebAPI)

"""Links the feature/model/provider combination to the implementation class.
Essential to user access to model and its feature(s)
"""
registry.register(
    feature=FeatureType.TEXT_TO_VIDEO,
    model="wan2.1-t2v-14b",
    provider="modal",
    implementation=Wan21TextToVideo14B
)
registry.register(
    feature=FeatureType.TEXT_TO_VIDEO,
    model="wan2.1-14b-t2v-fusionx",
    provider="modal",
    implementation=Wan2114bTextToVideoFusionX,
)
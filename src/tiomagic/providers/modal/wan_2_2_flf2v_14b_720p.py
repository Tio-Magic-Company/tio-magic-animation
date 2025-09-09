import modal
from pathlib import Path
from fastapi.responses import JSONResponse
import tempfile
import random
import gc
import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, Optional

from .base import GPUType, GenericWebAPI, ModalProviderBase
from ...core.registry import registry
from ...core._utils import load_image_robust, is_local_path, local_image_to_base64, create_timestamp, extract_image_dimensions
from ...core.constants import FeatureType
from ...core.schemas import FEATURE_SCHEMAS
from ...core.errors import (
    DeploymentError, ProcessingError
)

# NOTE: Optimization utilities from original script preserved below
from contextvars import ContextVar
from io import BytesIO
from unittest.mock import patch
import contextlib
from typing import Callable, ParamSpec, cast
from torch._inductor.package.package import package_aoti
from torch.export.pt2_archive._package import AOTICompiledModel
from torch.export.pt2_archive._package_weights import Weights
from torch.utils._pytree import tree_map_only
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig


APP_NAME = "wan-2.2-i2v-a14b-diffusers"
MODEL_NAME = "wan2.2-i2v-a14b"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f"{APP_NAME}-outputs"
OUTPUTS_PATH = "/outputs"

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
# NOTE: Additional model needed for optimized transformers
TRANSFORMER_MODEL_ID = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers"

GPU_CONFIG: GPUType = GPUType.H200
TIMEOUT: int = 1800 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # stay idle for 15 minutes before scaling down

# Dimension constants
MAX_DIMENSION = 832
MIN_DIMENSION = 480
DIMENSION_MULTIPLE = 16
SQUARE_SIZE = 480

# Frame and timing constants
FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

# Seed constant
MAX_SEED = np.iinfo(np.int32).max

# Default negative prompt
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,过曝，"
cuda_tag = "12.6.3-devel-ubuntu22.04"  # matches cu126
image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.11")
    .entrypoint([])  # quiet the base image
    .apt_install("git", "build-essential")  # for any native builds
    .pip_install(
        # your deps (keep versions you already set)
        "torch<2.9",
        "torchvision>=0.19.0",
        "torchao",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.49.0",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "opencv-python>=4.9.0.80",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "numpy>=1.23.5,<2",
        "fastapi",
        "Pillow",
        "python-dotenv",
        "peft",
        "ftfy",
    )
    # If you want the exact nightly wheel, keep your pip command:
    .run_commands(
        "pip install --upgrade --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu126 'torch<2.9'"
    )
    .env({
        "HF_HUB_CACHE": CACHE_PATH,
        # These are usually already correct on devel images, but it doesn't hurt:
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH",
    })
)


cache_volume = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)
# NOTE: May need LoRA weights volume if not downloading from HuggingFace
lora_volume = modal.Volume.from_name(f"{APP_NAME}-lora", create_if_missing=True)

app = modal.App(APP_NAME)


# ============= Optimization Utils Section (from original) =============

P = ParamSpec('P')

INDUCTOR_CONFIGS_OVERRIDES = {
    'aot_inductor.package_constants_in_so': False,
    'aot_inductor.package_constants_on_disk': True,
    'aot_inductor.package': True,
}

# Dynamic shaping constants
LATENT_FRAMES_DIM = torch.export.Dim('num_latent_frames', min=8, max=81)
LATENT_PATCHED_HEIGHT_DIM = torch.export.Dim('latent_patched_height', min=30, max=52)
LATENT_PATCHED_WIDTH_DIM = torch.export.Dim('latent_patched_width', min=30, max=52)

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        2: LATENT_FRAMES_DIM,
        3: 2 * LATENT_PATCHED_HEIGHT_DIM,
        4: 2 * LATENT_PATCHED_WIDTH_DIM,
    },
}

INDUCTOR_CONFIGS = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}


class ZeroGPUWeights:
    def __init__(self, constants_map: dict[str, torch.Tensor], to_cuda: bool = False):
        if to_cuda:
            self.constants_map = {name: tensor.to('cuda') for name, tensor in constants_map.items()}
        else:
            self.constants_map = constants_map
    
    def __reduce__(self):
        constants_map: dict[str, torch.Tensor] = {}
        for name, tensor in self.constants_map.items():
            tensor_ = torch.empty_like(tensor, device='cpu').pin_memory()
            constants_map[name] = tensor_.copy_(tensor).detach().share_memory_()
        return ZeroGPUWeights, (constants_map, True)


class ZeroGPUCompiledModel:
    def __init__(self, archive_file: torch.types.FileLike, weights: ZeroGPUWeights):
        self.archive_file = archive_file
        self.weights = weights
        self.compiled_model: ContextVar[AOTICompiledModel | None] = ContextVar('compiled_model', default=None)
    
    def __call__(self, *args, **kwargs):
        if (compiled_model := self.compiled_model.get()) is None:
            compiled_model = cast(AOTICompiledModel, torch._inductor.aoti_load_package(self.archive_file))
            compiled_model.load_constants(self.weights.constants_map, check_full_update=True, user_managed=True)
            self.compiled_model.set(compiled_model)
        return compiled_model(*args, **kwargs)
    
    def __reduce__(self):
        return ZeroGPUCompiledModel, (self.archive_file, self.weights)


def aoti_compile(
    exported_program: torch.export.ExportedProgram,
    inductor_configs: dict[str, Any] | None = None,
):
    inductor_configs = (inductor_configs or {}) | INDUCTOR_CONFIGS_OVERRIDES
    gm = cast(torch.fx.GraphModule, exported_program.module())
    assert exported_program.example_inputs is not None
    args, kwargs = exported_program.example_inputs
    artifacts = torch._inductor.aot_compile(gm, args, kwargs, options=inductor_configs)
    archive_file = BytesIO()
    from torch._inductor.package.package import package_aoti
    files: list[str | Weights] = [file for file in artifacts if isinstance(file, str)]
    package_aoti(archive_file, files)
    weights, = (artifact for artifact in artifacts if isinstance(artifact, Weights))
    zerogpu_weights = ZeroGPUWeights({name: weights.get_weight(name)[0] for name in weights})
    return ZeroGPUCompiledModel(archive_file, zerogpu_weights)


@contextlib.contextmanager
def capture_component_call(
    pipeline: Any,
    component_name: str,
    component_method='forward',
):
    class CapturedCallException(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    class CapturedCall:
        def __init__(self):
            self.args: tuple[Any, ...] = ()
            self.kwargs: dict[str, Any] = {}

    component = getattr(pipeline, component_name)
    captured_call = CapturedCall()

    def capture_call(*args, **kwargs):
        raise CapturedCallException(*args, **kwargs)

    with patch.object(component, component_method, new=capture_call):
        try:
            yield captured_call
        except CapturedCallException as e:
            captured_call.args = e.args
            captured_call.kwargs = e.kwargs


def drain_module_parameters(module: torch.nn.Module):
    """
    Replaces all parameters in a module with empty tensors on the CPU.
    Handles torchao's Float8Tensor by dequantizing first.
    """
    # Import the custom tensor type to check against it
    from torchao.quantization import Float8Tensor

    new_state_dict = {}
    for name, tensor in module.state_dict().items():
        # Check if the tensor is the special quantized type
        if isinstance(tensor, Float8Tensor):
            # Dequantize it back to a normal tensor first, then create an empty one like it
            dequantized_tensor = tensor.dequantize()
            placeholder = torch.nn.Parameter(torch.empty_like(dequantized_tensor, device='cpu'))
        else:
            # If it's a normal tensor, proceed as before
            placeholder = torch.nn.Parameter(torch.empty_like(tensor, device='cpu'))
        
        new_state_dict[name] = placeholder

    module.load_state_dict(new_state_dict, assign=True)


def optimize_pipeline_(pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):
    """Optimize the pipeline with LoRA fusion and compilation"""
    
    def compile_transformer():
        # LoRA fusion
        # NOTE: LoRA weights location may need adjustment based on volume setup
        pipeline.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
            adapter_name="lightx2v"
        )
        kwargs_lora = {}
        kwargs_lora["load_into_transformer_2"] = True
        pipeline.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
            adapter_name="lightx2v_2", **kwargs_lora
        )
        pipeline.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
        pipeline.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
        pipeline.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])
        pipeline.unload_lora_weights()
        
        # Capture a single call to get the args/kwargs structure
        with capture_component_call(pipeline, 'transformer') as call:
            pipeline(*args, **kwargs)
        
        dynamic_shapes = tree_map_only((torch.Tensor, bool), lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

        # Quantization
        quantize_(pipeline.transformer, Float8DynamicActivationFloat8WeightConfig())
        quantize_(pipeline.transformer_2, Float8DynamicActivationFloat8WeightConfig())
        
        # Compilation
        exported_1 = torch.export.export(
            mod=pipeline.transformer,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )
        
        exported_2 = torch.export.export(
            mod=pipeline.transformer_2,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        compiled_1 = aoti_compile(exported_1, INDUCTOR_CONFIGS)
        compiled_2 = aoti_compile(exported_2, INDUCTOR_CONFIGS)
        
        return compiled_1, compiled_2

    # Quantize text encoder
    quantize_(pipeline.text_encoder, Int8WeightOnlyConfig())
    
    # Get the two dynamically-shaped compiled models
    compiled_transformer_1, compiled_transformer_2 = compile_transformer()

    # Assignment
    pipeline.transformer.forward = compiled_transformer_1
    drain_module_parameters(pipeline.transformer)

    pipeline.transformer_2.forward = compiled_transformer_2
    drain_module_parameters(pipeline.transformer_2)


# ============= Main Modal Class =============

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume, "/lora": lora_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Interpolate:
    @modal.enter()
    def load_models(self):
        import time
        from diffusers import FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

        print("Loading models...")
        start_time = time.time()
        
        try:
            print(f"✅ {time.time() - start_time:.2f}s: Loading transformers...")
            
            # Load optimized transformers
            transformer = WanTransformer3DModel.from_pretrained(
                TRANSFORMER_MODEL_ID,
                subfolder='transformer',
                torch_dtype=torch.bfloat16,
                device_map='cuda',
            )
            
            transformer_2 = WanTransformer3DModel.from_pretrained(
                TRANSFORMER_MODEL_ID,
                subfolder='transformer_2',
                torch_dtype=torch.bfloat16,
                device_map='cuda',
            )
            
            print(f"✅ {time.time() - start_time:.2f}s: Creating pipeline...")
            
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                MODEL_ID,
                transformer=transformer,
                transformer_2=transformer_2,
                torch_dtype=torch.bfloat16,
            )
            
            # Configure scheduler
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, 
                shift=8.0
            )
            
            self.pipe.to("cuda")
            
            # Apply optimizations if enabled
            self.optimize = True  # Can be made configurable
            if self.optimize:
                print(f"✅ {time.time() - start_time:.2f}s: Applying optimizations...")
                self._optimize_pipeline()
            
            print(f"✅ {time.time() - start_time:.2f}s: Pipeline ready. Models loaded successfully.")
            
        except Exception as e:
            print(f"❌ {time.time() - start_time:.2f}s: An error occurred: {e}")
            raise

    def _optimize_pipeline(self):
        """Apply optimizations to the pipeline"""
        # Clear GPU cache
        for i in range(3):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Call the optimization function with a placeholder image
        optimize_pipeline_(
            self.pipe,
            image=Image.new('RGB', (MAX_DIMENSION, MIN_DIMENSION)),
            prompt='prompt',
            height=MIN_DIMENSION,
            width=MAX_DIMENSION,
            num_frames=MAX_FRAMES_MODEL,
        )

    def process_image_for_video(self, image: Image.Image) -> tuple[Image.Image, int, int]:
        """
        Resize an image based on video generation rules.
        Returns processed image and its dimensions.
        """
        width, height = image.size
        
        # Handle square images
        if width == height:
            resized = image.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
            return resized, SQUARE_SIZE, SQUARE_SIZE
        
        # Determine target dimensions while preserving aspect ratio
        aspect_ratio = width / height
        new_width, new_height = float(width), float(height)
        
        # Scale down if too large
        if new_width > MAX_DIMENSION or new_height > MAX_DIMENSION:
            if aspect_ratio > 1:  # Landscape
                scale = MAX_DIMENSION / new_width
            else:  # Portrait
                scale = MAX_DIMENSION / new_height
            new_width *= scale
            new_height *= scale
        
        # Scale up if too small
        if new_width < MIN_DIMENSION or new_height < MIN_DIMENSION:
            if aspect_ratio > 1:  # Landscape
                scale = MIN_DIMENSION / new_height
            else:  # Portrait
                scale = MIN_DIMENSION / new_width
            new_width *= scale
            new_height *= scale
        
        # Round to nearest multiple of DIMENSION_MULTIPLE
        final_width = int(round(new_width / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
        final_height = int(round(new_height / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
        
        # Ensure final dimensions meet minimum requirements
        final_width = max(final_width, MIN_DIMENSION if aspect_ratio < 1 else SQUARE_SIZE)
        final_height = max(final_height, MIN_DIMENSION if aspect_ratio > 1 else SQUARE_SIZE)
        
        resized = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
        return resized, final_height, final_width

    def resize_and_crop_to_match(self, target_image: Image.Image, height: int, width: int) -> tuple[Image.Image, int, int]:
        """
        Resize and center-crop the target image to match given dimensions.
        """
        target_width, target_height = target_image.size
        
        scale = max(width / target_width, height / target_height)
        new_width, new_height = int(target_width * scale), int(target_height * scale)
        
        resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        
        cropped = resized.crop((left, top, left + width, top + height))
        return cropped, height, width

    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        
        # Get schema for defaults
        interpolate_schema = FEATURE_SCHEMAS.get("interpolate", {}).get(MODEL_NAME, {})
        
        # Extract frames
        first_frame = data.get('first_frame')
        last_frame = data.get('last_frame')
        
        # Process first frame to determine dimensions
        first_frame, height, width = self.process_image_for_video(first_frame)
        
        # Match last frame to first frame dimensions
        if last_frame.size != first_frame.size:
            last_frame, _, _ = self.resize_and_crop_to_match(last_frame, height, width)
        
        # Extract parameters with defaults
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT)
        
        # Calculate duration and frames
        duration_seconds = data.get('duration_seconds', 2.1)
        num_frames = np.clip(
            int(round(duration_seconds * FIXED_FPS)), 
            MIN_FRAMES_MODEL, 
            MAX_FRAMES_MODEL
        )
        
        # Other generation parameters
        steps = data.get('num_inference_steps', 8)
        guidance_scale = float(data.get('guidance_scale', 1.0))
        guidance_scale_2 = float(data.get('guidance_scale_2', 1.0))
        
        # Handle seed
        seed = data.get('seed')
        randomize_seed = data.get('randomize_seed', False)
        if randomize_seed or seed is None:
            seed = random.randint(0, MAX_SEED)
        else:
            seed = int(seed)
        
        # Generate video
        print(f"Generating {num_frames} frames at {width}x{height} (seed: {seed})...")
        
        output = self.pipe(
            image=first_frame,
            last_image=last_frame,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_inference_steps=steps,
            generator=torch.Generator(device='cuda').manual_seed(seed),
        )
        
        frames_list = output.frames[0]
        
        # Save video
        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-interpolate-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames_list, str(mp4_path), fps=FIXED_FPS)
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
            
            # Extract dimensions from first frame if not specified
            if 'height' not in data and 'width' not in data:
                data = extract_image_dimensions(first_frame, data)
        except Exception as e:
            raise ProcessingError(
                media_type="image",
                operation="load and process",
                reason=str(e),
                file_path=data.get("first_frame") if isinstance(data.get("first_frame"), str) else None
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


class Wan22I2vInterpolateA14b(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = FeatureType.INTERPOLATE
        
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan 2.2 I2V Interpolate model."""
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.INTERPOLATE

        # Handle frame arguments (support both naming conventions)
        first_frame = payload.get('first_frame') or payload.get('start_frame')
        last_frame = payload.get('last_frame') or payload.get('end_frame')
        
        if first_frame is None or last_frame is None:
            raise ValueError("Arguments 'first_frame' and 'last_frame' are required for Interpolation Video generation")

        payload['first_frame'] = first_frame
        payload['last_frame'] = last_frame
        
        # Remove alternate names if present
        payload.pop('start_frame', None)
        payload.pop('end_frame', None)

        if is_local_path(payload['first_frame']):
            payload["first_frame"] = local_image_to_base64(payload['first_frame'])
        if is_local_path(payload['last_frame']):
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
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume, "/lora": lora_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)(WebAPI)


registry.register(
    feature=FeatureType.INTERPOLATE,
    model="wan2.2-flf2v-a14b",
    provider="modal",
    implementation=Wan22I2vInterpolateA14b
)
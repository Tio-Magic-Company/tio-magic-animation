# wan_i2v_local.py
#
# Single-file, local, no-Gradio, no-Spaces script to run Wan 2.2 I2V with AOTInductor + torchao quantization.
# Provides an OOP class WanI2VGenerator with a .generate(...) method.
#
# Requirements (installed in your environment ahead of time):
# - torch (nightly with AOTInductor packaging; 2.8+ nightly recommended)
# - torchvision
# - diffusers >= 0.29
# - transformers
# - accelerate
# - torchao
# - PIL, numpy
#
# Example:
#   from PIL import Image
#   gen = WanI2VGenerator(optimize=True)  # compile and quantize
#   frames, seed = gen.generate(Image.open("start.png"), Image.open("end.png"), "a smooth camera pan")
#   # or save directly:
#   video_path, seed = gen.generate(Image.open("start.png"), Image.open("end.png"), "a smooth camera pan", output_video_path="out.mp4")

import os
import random
import tempfile
from io import BytesIO
from typing import Any, Callable, Optional, Tuple, List, Union, cast
from unittest.mock import patch
import contextlib

import numpy as np
import torch
from torch.utils._pytree import tree_map_only
from PIL import Image

from torch._inductor.package.package import package_aoti
from torch.export.pt2_archive._package import AOTICompiledModel
from torch.export.pt2_archive._package_weights import Weights

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.export_utils import export_to_video
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig


# -------------------------------
# Shared constants and utilities
# -------------------------------

# Flexible dimension rules used for preprocessing input images
MAX_DIMENSION = 832
MIN_DIMENSION = 480
DIMENSION_MULTIPLE = 16
SQUARE_SIZE = 480

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

default_negative_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
    "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走,过曝，"
)


# -------------------------------
# AOTInductor packaging helpers
# -------------------------------

INDUCTOR_CONFIGS_DEFAULT = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}
INDUCTOR_CONFIGS_OVERRIDES = {
    'aot_inductor.package_constants_in_so': False,
    'aot_inductor.package_constants_on_disk': True,
    'aot_inductor.package': True,
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
        self.compiled_model = torch.utils._contextlib.ContextVar('compiled_model', default=None)  # type: ignore[attr-defined]

    def __call__(self, *args, **kwargs):
        compiled_model = self.compiled_model.get()  # type: ignore[attr-defined]
        if compiled_model is None:
            compiled_model = cast(AOTICompiledModel, torch._inductor.aoti_load_package(self.archive_file))
            compiled_model.load_constants(self.weights.constants_map, check_full_update=True, user_managed=True)
            self.compiled_model.set(compiled_model)  # type: ignore[attr-defined]
        return compiled_model(*args, **kwargs)

    def __reduce__(self):
        return ZeroGPUCompiledModel, (self.archive_file, self.weights)


def aoti_compile(
    exported_program: torch.export.ExportedProgram,
    inductor_configs: Optional[dict[str, Any]] = None,
):
    options = (inductor_configs or {}) | INDUCTOR_CONFIGS_DEFAULT | INDUCTOR_CONFIGS_OVERRIDES
    gm = cast(torch.fx.GraphModule, exported_program.module())
    assert exported_program.example_inputs is not None
    args, kwargs = exported_program.example_inputs

    artifacts = torch._inductor.aot_compile(gm, args, kwargs, options=options)
    archive_file = BytesIO()
    files: List[Union[str, Weights]] = [file for file in artifacts if isinstance(file, str)]
    package_aoti(archive_file, files)

    weights, = (artifact for artifact in artifacts if isinstance(artifact, Weights))
    zerogpu_weights = ZeroGPUWeights({name: weights.get_weight(name)[0] for name in weights})

    return ZeroGPUCompiledModel(archive_file, zerogpu_weights)


@contextlib.contextmanager
def capture_component_call(
    pipeline: Any,
    component_name: str,
    component_method: str = 'forward',
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
    state_dict_meta = {name: {'device': tensor.device, 'dtype': tensor.dtype} for name, tensor in module.state_dict().items()}
    state_dict = {name: torch.nn.Parameter(torch.empty_like(tensor, device='cpu')) for name, tensor in module.state_dict().items()}
    module.load_state_dict(state_dict, assign=True)
    for name, param in state_dict.items():
        meta = state_dict_meta[name]
        param.data = torch.Tensor([]).to(**meta)


# -------------------------------
# Dynamic shape definitions for export
# -------------------------------

# VAE temporal scale factor is 1, latent_frames = num_frames. Range is [8, 81].
LATENT_FRAMES_DIM = torch.export.Dim('num_latent_frames', min=8, max=81)

# App range for pixel dimensions: [480, 832]. VAE scale factor is 8.
# Latent dimension range: [60, 104]. Patched latent dimension range: [30, 52].
LATENT_PATCHED_HEIGHT_DIM = torch.export.Dim('latent_patched_height', min=30, max=52)
LATENT_PATCHED_WIDTH_DIM = torch.export.Dim('latent_patched_width', min=30, max=52)

# hidden_states shape: (batch, channels, num_frames, height, width)
TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        2: LATENT_FRAMES_DIM,
        3: 2 * LATENT_PATCHED_HEIGHT_DIM,  # guarantees even height
        4: 2 * LATENT_PATCHED_WIDTH_DIM,   # guarantees even width
    },
}


# -------------------------------
# Optimization (quantize + export + AOTI compile)
# -------------------------------

def optimize_pipeline_(pipeline: Callable[..., Any], *args, **kwargs):
    # Fuse LoRA into the two transformers
    pipeline.load_lora_weights(
        "Kijai/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v"
    )
    kwargs_lora = {"load_into_transformer_2": True}
    pipeline.load_lora_weights(
        "Kijai/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v_2",
        **kwargs_lora
    )
    pipeline.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
    pipeline.fuse_lora(adapter_names=["lightx2v"], lora_scale=3.0, components=["transformer"])
    pipeline.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1.0, components=["transformer_2"])
    pipeline.unload_lora_weights()

    # Capture one forward call to the first transformer to obtain example args/kwargs without running the whole pipeline
    with capture_component_call(pipeline, 'transformer') as call:
        pipeline(*args, **kwargs)

    # Build dynamic_shapes mapping
    dynamic_shapes = tree_map_only((torch.Tensor, bool), lambda t: None, call.kwargs)
    dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

    # Quantize: transformer weights and dynamic activations to FP8; text encoder to W8A16 (int8 weight-only)
    quantize_(pipeline.transformer, Float8DynamicActivationFloat8WeightConfig())
    quantize_(pipeline.transformer_2, Float8DynamicActivationFloat8WeightConfig())
    quantize_(pipeline.text_encoder, Int8WeightOnlyConfig())

    # Export to torch.export and compile via AOTInductor + package
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

    compiled_1 = aoti_compile(exported_1, INDUCTOR_CONFIGS_DEFAULT)
    compiled_2 = aoti_compile(exported_2, INDUCTOR_CONFIGS_DEFAULT)

    # Replace forward with compiled functions, and free parameters to save memory
    pipeline.transformer.forward = compiled_1
    drain_module_parameters(pipeline.transformer)

    pipeline.transformer_2.forward = compiled_2
    drain_module_parameters(pipeline.transformer_2)


# -------------------------------
# Image preprocessing utilities
# -------------------------------

def process_image_for_video(image: Image.Image) -> Image.Image:
    """
    Resizes an image based on the following rules for video generation:
    1. The longest side will be scaled down to MAX_DIMENSION if it's larger.
    2. The shortest side will be scaled up to MIN_DIMENSION if it's smaller.
    3. The final dimensions will be rounded to the nearest multiple of DIMENSION_MULTIPLE.
    4. Square images are resized to a fixed SQUARE_SIZE.
    The aspect ratio is preserved as closely as possible.
    """
    width, height = image.size

    # Rule 4: Handle square images
    if width == height:
        return image.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)

    # Determine target dimensions while preserving aspect ratio
    aspect_ratio = width / height
    new_width, new_height = float(width), float(height)

    # Rule 1: Scale down if too large
    if new_width > MAX_DIMENSION or new_height > MAX_DIMENSION:
        if aspect_ratio > 1:  # Landscape
            scale = MAX_DIMENSION / new_width
        else:  # Portrait
            scale = MAX_DIMENSION / new_height
        new_width *= scale
        new_height *= scale

    # Rule 2: Scale up if too small
    if new_width < MIN_DIMENSION or new_height < MIN_DIMENSION:
        if aspect_ratio > 1:  # Landscape
            scale = MIN_DIMENSION / new_height
        else:  # Portrait
            scale = MIN_DIMENSION / new_width
        new_width *= scale
        new_height *= scale

    # Rule 3: Round to the nearest multiple of DIMENSION_MULTIPLE
    final_width = int(round(new_width / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
    final_height = int(round(new_height / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)

    # Ensure final dimensions are at least the minimum
    final_width = max(final_width, MIN_DIMENSION if aspect_ratio < 1 else SQUARE_SIZE)
    final_height = max(final_height, MIN_DIMENSION if aspect_ratio > 1 else SQUARE_SIZE)

    return image.resize((final_width, final_height), Image.Resampling.LANCZOS)


def resize_and_crop_to_match(target_image: Image.Image, reference_image: Image.Image) -> Image.Image:
    """Resizes and center-crops the target image to match the reference image's dimensions."""
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


# -------------------------------
# Main OOP wrapper
# -------------------------------

class WanI2VGenerator:
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        transformer_repo: str = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
        optimize: bool = True,
        compile_placeholder_dims: Tuple[int, int] = (MAX_DIMENSION, MIN_DIMENSION),  # (width, height)
    ):
        """
        Initialize the pipeline. If optimize=True, compiles transformers with AOTInductor and quantizes modules.
        """
        assert torch.cuda.is_available(), "CUDA GPU is required."
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.transformer_repo = transformer_repo

        print("Loading Wan 2.2 I2V models...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            transformer=WanTransformer3DModel.from_pretrained(
                self.transformer_repo,
                subfolder='transformer',
                torch_dtype=self.dtype,
                device_map=self.device,
            ),
            transformer_2=WanTransformer3DModel.from_pretrained(
                self.transformer_repo,
                subfolder='transformer_2',
                torch_dtype=self.dtype,
                device_map=self.device,
            ),
            torch_dtype=self.dtype,
        )
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, shift=8.0)
        self.pipe.to(self.device)

        if optimize:
            print("Optimizing (quantize + compile) transformers... this may take a few minutes.")
            placeholder_w, placeholder_h = compile_placeholder_dims
            optimize_pipeline_(
                self.pipe,
                image=Image.new('RGB', (placeholder_w, placeholder_h)),
                prompt='placeholder prompt',
                height=placeholder_h,
                width=placeholder_w,
                num_frames=MAX_FRAMES_MODEL,
            )
            print("Optimization complete.")

    def generate(
        self,
        start_image_pil: Image.Image,
        end_image_pil: Image.Image,
        prompt: str,
        negative_prompt: str = default_negative_prompt,
        duration_seconds: float = 2.1,
        steps: int = 8,
        guidance_scale: float = 1.0,
        guidance_scale_2: float = 1.0,
        seed: int = 42,
        randomize_seed: bool = False,
        output_video_path: Optional[str] = None,
        fps: int = FIXED_FPS,
    ) -> Tuple[Union[List[Image.Image], str], int]:
        """
        Generate a video from start and end images with the given text prompt.
        If output_video_path is provided, writes MP4 and returns (video_path, seed).
        Otherwise returns (frames_list, seed).

        Returns:
          Tuple[Union[List[PIL.Image.Image], str], int]
        """
        if start_image_pil is None or end_image_pil is None:
            raise ValueError("Both start_image_pil and end_image_pil must be provided.")

        # Preprocess images with fixed rules and match sizes
        processed_start_image = process_image_for_video(start_image_pil)
        processed_end_image = resize_and_crop_to_match(end_image_pil, processed_start_image)
        target_height, target_width = processed_start_image.height, processed_start_image.width

        # Seed and frame count
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
        num_frames = int(np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))

        # Run pipeline
        result = self.pipe(
            image=processed_start_image,
            last_image=processed_end_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_height,
            width=target_width,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            guidance_scale_2=float(guidance_scale_2),
            num_inference_steps=int(steps),
            generator=torch.Generator(device=self.device).manual_seed(current_seed),
        )
        frames_list: List[Image.Image] = result.frames[0]

        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
            export_to_video(frames_list, output_video_path, fps=fps)
            return output_video_path, current_seed

        return frames_list, current_seed


# -------------------------------
# Optional CLI usage
# -------------------------------

if __name__ == "__main__":
    # Example usage (edit paths/prompts before running)
    # This will load, optimize (compile), and then generate once.
    start_path = "start.png"  # replace with your file
    end_path = "end.png"      # replace with your file
    if os.path.exists(start_path) and os.path.exists(end_path):
        gen = WanI2VGenerator(optimize=True)
        frames_or_path, seed = gen.generate(
            Image.open(start_path).convert("RGB"),
            Image.open(end_path).convert("RGB"),
            prompt="a smooth cinematic transition between scenes",
            duration_seconds=2.1,
            steps=8,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
            seed=42,
            randomize_seed=False,
            output_video_path="wan_i2v_out.mp4",
        )
        print(f"Done. Output: {frames_or_path}, seed: {seed}")
    else:
        print("Please set valid paths for start.png and end.png (or run this module to import the class and call .generate()).")
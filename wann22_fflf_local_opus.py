import os
import sys
import contextlib
from contextvars import ContextVar
from io import BytesIO
from typing import Any, Optional, Callable, ParamSpec, cast
from unittest.mock import patch
import tempfile
import random
import gc

import torch
import numpy as np
from PIL import Image
from torch._inductor.package.package import package_aoti
from torch.export.pt2_archive._package import AOTICompiledModel
from torch.export.pt2_archive._package_weights import Weights
from torch.utils._pytree import tree_map_only
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video


# ============= Optimization Utils Section =============

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
    state_dict_meta = {name: {'device': tensor.device, 'dtype': tensor.dtype} for name, tensor in module.state_dict().items()}
    state_dict = {name: torch.nn.Parameter(torch.empty_like(tensor, device='cpu')) for name, tensor in module.state_dict().items()}
    module.load_state_dict(state_dict, assign=True)
    for name, param in state_dict.items():
        meta = state_dict_meta[name]
        param.data = torch.Tensor([]).to(**meta)


# ============= Optimization Section =============

P = ParamSpec('P')

# Dynamic shaping constants
LATENT_FRAMES_DIM = torch.export.Dim('num_latent_frames', min=8, max=81)
LATENT_PATCHED_HEIGHT_DIM = torch.export.Dim('latent_patched_height', min=30, max=52)
LATENT_PATCHED_WIDTH_DIM = torch.export.Dim('latent_patched_width', min=30, max=52)

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        2: LATENT_FRAMES_DIM,
        3: 2 * LATENT_PATCHED_HEIGHT_DIM,  # Guarantees even height
        4: 2 * LATENT_PATCHED_WIDTH_DIM,   # Guarantees even width
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


def optimize_pipeline_(pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):
    """Optimize the pipeline with LoRA fusion and compilation"""
    
    def compile_transformer():
        # LoRA fusion
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


# ============= Main Video Generator Class =============

class Wan22VideoGenerator:
    """
    OOP-based video generator using Wan 2.2 I2V model.
    Generates videos by interpolating between two frames guided by text prompts.
    """
    
    # Model constants
    MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    
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
    
    def __init__(self, device: str = 'cuda', optimize: bool = True, install_nightly_torch: bool = True):
        """
        Initialize the video generator.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            optimize: Whether to apply optimization (LoRA fusion, quantization, compilation)
            install_nightly_torch: Whether to install PyTorch 2.8 nightly (required for optimization)
        """
        self.device = device
        self.optimize = optimize
        
        # Install PyTorch nightly if requested
        if install_nightly_torch and optimize:
            print("Installing PyTorch 2.8 nightly...")
            os.system('pip install --upgrade --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu126 "torch<2.9"')
        
        # Initialize the pipeline
        self._initialize_pipeline()
        
        # Apply optimizations if requested
        if self.optimize:
            self._optimize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the diffusion pipeline"""
        print("Loading models into memory. This may take a few minutes...")
        
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.MODEL_ID,
            transformer=WanTransformer3DModel.from_pretrained(
                'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
                subfolder='transformer',
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ),
            transformer_2=WanTransformer3DModel.from_pretrained(
                'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
                subfolder='transformer_2',
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ),
            torch_dtype=torch.bfloat16,
        )
        
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, 
            shift=8.0
        )
        self.pipe.to(self.device)
        
        print("Pipeline loaded successfully.")
    
    def _optimize_pipeline(self):
        """Apply optimizations to the pipeline"""
        print("Optimizing pipeline...")
        
        # Clear GPU cache
        for i in range(3):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Call the optimization function with a placeholder image
        optimize_pipeline_(
            self.pipe,
            image=Image.new('RGB', (self.MAX_DIMENSION, self.MIN_DIMENSION)),
            prompt='prompt',
            height=self.MIN_DIMENSION,
            width=self.MAX_DIMENSION,
            num_frames=self.MAX_FRAMES_MODEL,
        )
        
        print("Pipeline optimization complete.")
    
    def process_image_for_video(self, image: Image.Image) -> Image.Image:
        """
        Resize an image based on video generation rules.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Processed PIL Image
        """
        width, height = image.size
        
        # Handle square images
        if width == height:
            return image.resize((self.SQUARE_SIZE, self.SQUARE_SIZE), Image.Resampling.LANCZOS)
        
        # Determine target dimensions while preserving aspect ratio
        aspect_ratio = width / height
        new_width, new_height = float(width), float(height)
        
        # Scale down if too large
        if new_width > self.MAX_DIMENSION or new_height > self.MAX_DIMENSION:
            if aspect_ratio > 1:  # Landscape
                scale = self.MAX_DIMENSION / new_width
            else:  # Portrait
                scale = self.MAX_DIMENSION / new_height
            new_width *= scale
            new_height *= scale
        
        # Scale up if too small
        if new_width < self.MIN_DIMENSION or new_height < self.MIN_DIMENSION:
            if aspect_ratio > 1:  # Landscape
                scale = self.MIN_DIMENSION / new_height
            else:  # Portrait
                scale = self.MIN_DIMENSION / new_width
            new_width *= scale
            new_height *= scale
        
        # Round to nearest multiple of DIMENSION_MULTIPLE
        final_width = int(round(new_width / self.DIMENSION_MULTIPLE) * self.DIMENSION_MULTIPLE)
        final_height = int(round(new_height / self.DIMENSION_MULTIPLE) * self.DIMENSION_MULTIPLE)
        
        # Ensure final dimensions meet minimum requirements
        final_width = max(final_width, self.MIN_DIMENSION if aspect_ratio < 1 else self.SQUARE_SIZE)
        final_height = max(final_height, self.MIN_DIMENSION if aspect_ratio > 1 else self.SQUARE_SIZE)
        
        return image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    
    def resize_and_crop_to_match(self, target_image: Image.Image, reference_image: Image.Image) -> Image.Image:
        """
        Resize and center-crop the target image to match reference dimensions.
        
        Args:
            target_image: Image to resize and crop
            reference_image: Image whose dimensions to match
            
        Returns:
            Resized and cropped PIL Image
        """
        ref_width, ref_height = reference_image.size
        target_width, target_height = target_image.size
        
        scale = max(ref_width / target_width, ref_height / target_height)
        new_width, new_height = int(target_width * scale), int(target_height * scale)
        
        resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        left = (new_width - ref_width) // 2
        top = (new_height - ref_height) // 2
        
        return resized.crop((left, top, left + ref_width, top + ref_height))
    
    def generate(
        self,
        start_frame: Image.Image,
        last_frame: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        duration_seconds: float = 2.1,
        steps: int = 8,
        guidance_scale: float = 1.0,
        guidance_scale_2: float = 1.0,
        seed: Optional[int] = None,
        randomize_seed: bool = False,
        output_path: Optional[str] = None,
        return_frames: bool = False
    ) -> tuple[str, int, Optional[list]]:
        """
        Generate a video interpolating between start and end frames.
        
        Args:
            start_frame: PIL Image for the first frame
            last_frame: PIL Image for the last frame
            prompt: Text description of the transition
            negative_prompt: Text describing what to avoid (optional)
            duration_seconds: Duration of the video in seconds
            steps: Number of inference steps
            guidance_scale: Guidance scale for high noise
            guidance_scale_2: Guidance scale for low noise
            seed: Random seed for generation
            randomize_seed: Whether to use a random seed
            output_path: Path to save the video (optional, uses temp file if None)
            return_frames: Whether to return the list of frames
            
        Returns:
            Tuple of (video_path, used_seed, frames_list or None)
        """
        if start_frame is None or last_frame is None:
            raise ValueError("Both start_frame and last_frame must be provided")
        
        # Use default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT
        
        # Process images
        processed_start = self.process_image_for_video(start_frame)
        processed_end = self.resize_and_crop_to_match(last_frame, processed_start)
        
        target_height = processed_start.height
        target_width = processed_start.width
        
        # Handle seed
        if randomize_seed or seed is None:
            current_seed = random.randint(0, self.MAX_SEED)
        else:
            current_seed = int(seed)
        
        # Calculate number of frames
        num_frames = np.clip(
            int(round(duration_seconds * self.FIXED_FPS)), 
            self.MIN_FRAMES_MODEL, 
            self.MAX_FRAMES_MODEL
        )
        
        print(f"Generating {num_frames} frames at {target_width}x{target_height} (seed: {current_seed})...")
        
        # Generate video
        output = self.pipe(
            image=processed_start,
            last_image=processed_end,
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
        
        frames_list = output.frames[0]
        
        # Save video
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
                video_path = tmpfile.name
        else:
            video_path = output_path
        
        export_to_video(frames_list, video_path, fps=self.FIXED_FPS)
        
        print(f"Video saved to: {video_path}")
        
        # Return results
        if return_frames:
            return video_path, current_seed, frames_list
        else:
            return video_path, current_seed, None
    
    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'pipe'):
            del self.pipe
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()


# ============= Example Usage =============

if __name__ == "__main__":
    # Initialize the generator
    generator = Wan22VideoGenerator(device='cuda', optimize=True)
    
    # Load example images
    start_image = Image.open("start_frame.jpg")  # Replace with your image path
    end_image = Image.open("end_frame.jpg")      # Replace with your image path
    
    # Generate video
    video_path, seed, _ = generator.generate(
        start_frame=start_image,
        last_frame=end_image,
        prompt="the character turns around smoothly",
        duration_seconds=2.1,
        steps=8,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=42,
        randomize_seed=False,
        output_path="output_video.mp4"
    )
    
    print(f"Video generated successfully!")
    print(f"Path: {video_path}")
    print(f"Seed used: {seed}")
    
    # Cleanup
    generator.cleanup()
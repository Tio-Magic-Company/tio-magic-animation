from fastapi import Body
import modal
from pathlib import Path
from fastapi.responses import JSONResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core._utils import prepare_video_and_mask, load_image_robust, is_local_path, local_image_to_base64, create_timestamp, load_video_robust, extract_image_dimensions
from ...core.constants import FeatureType
from ...core.schemas import FEATURE_SCHEMAS
from ...core.errors import (
    DeploymentError, ValidationError, ProcessingError
)


# --- Configuration ---
APP_NAME = "wan-2.1-vace-t2v-14b"
MODEL_NAME = "wan2.1-vace-14b"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f'{APP_NAME}-outputs'
OUTPUTS_PATH = "/outputs"

VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-14B-diffusers"

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 2700 # 30 minutes
SCALEDOWN_WINDOW: int = 900 # 15 minutes

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        'imageio',
        'onnxruntime',
        'imageio-ffmpeg',
        'python-multipart',
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        'ftfy',
        "accelerate",
        'matplotlib',
        'onnxruntime-gpu',
        'loguru',
        'numpy',
        'tqdm',
        'omegaconf',
        "opencv-python-headless",
        "safetensors",
        "fastapi",
        "python-dotenv",
        "peft"
    ).run_commands("pip install easy-dwpose --no-deps")
    .env({"HF_HUB_CACHE": CACHE_PATH, "TOKENIZERS_PARALLELISM": "false"})
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
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)

        # Completely remove any LoRA to ensure clean base model
        if hasattr(self.pipe, 'unload_lora_weights'):
            print("unloading lora weights")
            self.pipe.unload_lora_weights()
        
        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        t2v_schema = FEATURE_SCHEMAS["text_to_video"][MODEL_NAME]

        # dynamic flow shift
        flow_shift = data.pop('flow_shift') if ('flow_shift' in data and data['flow_shift']) else t2v_schema["optional"]["flow_shift"]["default"]
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        # for every value in data, pass into pipe
        frames = self.pipe(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-t2v-output_{timestamp}.mp4"
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
            t2v_instance = T2V()
            call = t2v_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="modal",
                reason=f"Failed to spawn T2V job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.TEXT_TO_VIDEO})

class Wan21VaceTextToVideo14B(ModalProviderBase):
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
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)

        # Completely remove any LoRA to ensure clean base model
        if hasattr(self.pipe, 'unload_lora_weights'):
            print("unloading lora weights")
            self.pipe.unload_lora_weights()

        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        import torch
        from diffusers.utils import export_to_video
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        print("***MODAL GENERATE METHOD***")
        print("Starting video generation process...")

        i2v_schema = FEATURE_SCHEMAS["image_to_video"][MODEL_NAME]

        # dynamic flow shift
        flow_shift = data.pop('flow_shift') if ('flow_shift' in data and data['flow_shift']) else i2v_schema["optional"]["flow_shift"]["default"]
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        # Define video parameters
        height = data.get('height', i2v_schema["optional"]["height"]["default"])
        width = data.get('width', i2v_schema["optional"]["width"]["default"])
        print('width and height in generate, ', width, height)
        num_frames = data.get('num_frames', i2v_schema["optional"]["num_frames"]["default"])

        # Prepare the data for the pipeline
        video, mask = prepare_video_and_mask(data.get('image'), height, width, num_frames)
        data.pop('image')
        print('data run in the pipeline: ', data)

        # Run the diffusion pipeline
        output_frames = self.pipe(
            video=video,
            mask=mask,
            **data,
            generator=torch.Generator("cuda").manual_seed(42),
        ).frames[0]
        print("Pipeline execution finished.")

        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-i2v-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output_frames, str(mp4_path), fps=16)
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

class Wan21VaceImageToVideo14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Image-to-Video model."""
        print("***CHILD PREPARE PAYLOAD***")

        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = "image_to_video"

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
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        # Force GPU memory cleanup first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)

        # Completely remove any LoRA to ensure clean base model
        if hasattr(self.pipe, 'unload_lora_weights'):
            print("unloading lora weights")
            self.pipe.unload_lora_weights()

        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        import torch
        print("Starting video generation process...")

        interpolate_schema = FEATURE_SCHEMAS["interpolate"][MODEL_NAME]
        # dynamic flow shift
        flow_shift = data.pop('flow_shift') if ('flow_shift' in data and data['flow_shift']) else interpolate_schema["optional"]["flow_shift"]["default"]
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        height = data.get('height', interpolate_schema["optional"]["height"]["default"])
        width = data.get('width', interpolate_schema["optional"]["width"]["default"])
        num_frames = data.get('num_frames', interpolate_schema["optional"]["num_frames"]["default"])
        video, mask = prepare_video_and_mask(img=data.get('first_frame'), height=height, width=width, num_frames=num_frames, last_frame=data.get('last_frame'))
        data.pop('first_frame')
        data.pop('last_frame')

        output_frames = self.pipe(
            video=video,
            mask=mask,
            **data,
            generator=torch.Generator().manual_seed(42),
        ).frames[0]
        print("Pipeline execution finished.")

        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-interpolate-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output_frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle frame interpolation."""  
        first_frame = data.get("first_frame")
        last_frame = data.get("last_frame")
        try:
            # load_image_robust can handle both URLs and base64 strings, return PIL.Image
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
                file_path=data.get('image') if isinstance(data.get("image"), str) else None
            )

        # Create Interpolate instance and call generate
        try:
            interpolate_instance = Interpolate()
            call = interpolate_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="modal",
                reason=f"Failed to spawn Interpolate job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.INTERPOLATE})

class Wan21VaceInterpolate14B(ModalProviderBase):
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

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class PoseGuidance:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        # from controlnet_aux import OpenposeDetector
        from easy_dwpose import DWposeDetector
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)

        # Completely remove any LoRA to ensure clean base model
        if hasattr(self.pipe, 'unload_lora_weights'):
            print("unloading lora weights")
            self.pipe.unload_lora_weights()

        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.open_pose = DWposeDetector(device=device)
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from PIL import Image
        import PIL.Image
        from diffusers.utils import export_to_video, load_video
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        import io
        import tempfile

        pose_guidance_schema = FEATURE_SCHEMAS["pose_guidance"][MODEL_NAME]
        
        # dynamic flow shift
        flow_shift = data.pop('flow_shift') if ('flow_shift' in data and data['flow_shift']) else pose_guidance_schema["optional"]["flow_shift"]["default"]
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        # if guiding_video provided, process video bytes and extract poses into PIL.Image
        if data.get('guiding_video', None) is not None:
            print("Processing 'guiding_video' to extract poses")
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(data['guiding_video'])
                f.flush()
                video_frames = load_video(f.name)

            num_frames = data.get('num_frames', pose_guidance_schema["optional"]["num_frames"]["default"])
            if len(video_frames) > num_frames:
                video_frames = video_frames[:num_frames]
            elif len(video_frames) < num_frames:
                num_frames = len(video_frames)

            width = data.get('width', pose_guidance_schema["optional"]["width"]["default"])
            height = data.get('height', pose_guidance_schema["optional"]["height"]["default"])
            video_frames = [frame.convert("RGB").resize((width, height)) for frame in video_frames]
            print(f"Extracting poses from {len(video_frames)} frames...")
            openpose_video = [self.open_pose(frame, output_type="pil", include_hands=True, include_face=True) for frame in video_frames]
            data.pop('guiding_video')
        else:
            # convert pose_video bytes to list of PIL.Image
            video_frames = data.pop('pose_video')
            openpose_video = [Image.fromarray(frame) for frame in video_frames]


        # 2. Process the start image
        print("Processing start image...")
        image = data.pop('image')
        if isinstance(image, PIL.Image.Image):
            start_image = image.convert("RGB")
        else:
            start_image = PIL.Image.open(io.BytesIO(image)).convert("RGB")


        print("Running inference pipeline...")
        output_frames = self.pipe(
            video=openpose_video,
            reference_images=[start_image],
            **data
        ).frames[0]

        print("Pipeline execution finished.")

        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-pose-guidance-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output_frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: Dict[str, Any]):
        image = data.get('image')
        try:
            # base64 or URL to PIL.Image.Image
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
            video_fields = ['guiding_video', 'pose_video']
            for field in video_fields:
                if field in data:
                    # convert URL -> bytes, base64 -> bytes
                    video = load_video_robust(data[field])
                    data[field] = video
        except Exception as e:
            raise ProcessingError(
                media_type="video",
                operation="load and process",
                reason=str(e),
            )

        try:
            pose_guidance_instance = PoseGuidance()
            call = pose_guidance_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="modal",
                reason=f"Failed to spawn Pose Guidance job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.POSE_GUIDANCE})

class Wan21VacePoseGuidance14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "PoseGuidance"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Pose Guidance model.
        Break out required args into payload
        """
        import base64
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = FeatureType.POSE_GUIDANCE

        if is_local_path(payload['image']):
            payload['image'] = local_image_to_base64(payload['image'])

        # convert local video path to base64
        video_fields = ['guiding_video', 'pose_video']
        for field in video_fields:
            if field in payload and is_local_path(payload[field]):
                print(f"converting local {field} to base64")
                path = payload[field]
                with open(path, "rb") as f:
                    payload[field] = base64.b64encode(f.read()).decode("utf-8")

        return payload

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class PhantomFusionXT2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(
            VACE_MODEL_ID, 
            subfolder="vae", 
            torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(
            VACE_MODEL_ID, 
            vae=self.vae, 
            torch_dtype=torch.bfloat16)

        self.pipe.load_lora_weights(
            "vrgamedevgirl84/Wan14BT2VFusioniX", 
            weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
            adapter_name="phantom"
        )
        self.pipe.enable_lora()
        
        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        phantom_fusionx_schema = FEATURE_SCHEMAS["text_to_video"]["wan2.1-vace-14b-phantom-fusionx"]
        
        # dynamic flow shift
        flow_shift = data.pop('flow_shift') if ('flow_shift' in data and data['flow_shift']) else phantom_fusionx_schema["optional"]["flow_shift"]["default"]
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        # for every value in data, pass into pipe
        frames = self.pipe(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-phantom-fusionx-t2v-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes 
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle text-to-video generation."""
        prompt = data.get("prompt")

        if not prompt:
            return {"error": "A 'prompt' is required."}

        print(f"text_to_video - prompt: {prompt}")

        print("handle web inference data: ", data)
        # Create T2V instance and call generate
        try:
            t2v_instance = PhantomFusionXT2V()
            call = t2v_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="modal",
                reason=f"Failed to spawn FusionX T2V job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.TEXT_TO_VIDEO})

class Wan21VaceTextToVideo14BPhantomFusionX(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "PhantomFusionXT2V"
    def _prepare_payload(self, required_args, **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Fusion Xmodel."""
        payload = super()._prepare_payload(required_args, **kwargs)

        # Add feature_type for routing
        payload["feature_type"] = FeatureType.TEXT_TO_VIDEO
        payload["model"] = "wan2.1-vace-14b-phantom-fusionx"

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
class FusionXI2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)

        # Load I2V FusionX
        self.pipe.load_lora_weights("vrgamedevgirl84/Wan14BT2VFusioniX", weight_name="FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors", adapter_names=["I2V FusionX"])
        self.pipe.enable_lora()

        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        import torch
        from diffusers.utils import export_to_video
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        print("***MODAL GENERATE METHOD***")
        print("Starting video generation process...")

        i2v_schema = FEATURE_SCHEMAS["image_to_video"]["wan2.1-vace-14b-i2v-fusionx"]
        
        # dynamic flow shift
        flow_shift = data.pop('flow_shift') if ('flow_shift' in data and data['flow_shift']) else i2v_schema["optional"]["flow_shift"]["default"]
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        # Define video parameters
        height = data.get('height', i2v_schema["optional"]["height"]["default"])
        width = data.get('width', i2v_schema["optional"]["width"]["default"])
        print('width and height in generate, ', width, height)
        num_frames = data.get('num_frames', i2v_schema["optional"]["num_frames"]["default"])

        # Prepare the data for the pipeline
        video, mask = prepare_video_and_mask(data.get('image'), height, width, num_frames)
        data.pop('image')
        print('data run in the pipeline: ', data)

        # Run the diffusion pipeline
        output_frames = self.pipe(
            video=video,
            mask=mask,
            **data,
            generator=torch.Generator("cuda").manual_seed(42),
        ).frames[0]
        print("Pipeline execution finished.")

        timestamp = create_timestamp()
        mp4_name = f"{MODEL_NAME}-wan2.1-vace-14b-i2v-fusionx-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output_frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle image-to-video generation."""
        prompt = data.get("prompt")
        image = data.get("image")
        if not prompt:
            raise ValidationError(
                field="prompt",
                message="Argument 'prompt' is required for image-to-video generation",
                value=prompt
            )
        
        if not image:
            raise ValidationError(
                field="image", 
                message="Argument 'image' is required for image-to-video generation",
                value=image
            )

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
            i2v_instance = FusionXI2V()
            call = i2v_instance.generate.spawn(data)
        except Exception as e:
            raise DeploymentError(
                service="Modal",
                reason=f"Failed to spawn FusionX I2V job: {str(e)}",
                app_name=APP_NAME
            )

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.IMAGE_TO_VIDEO})

class Wan21VaceImageToVideo14BFusionX(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Image-to-Video FusionX model."""
        print("***CHILD PREPARE PAYLOAD***")

        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = "image_to_video"
        payload["model"] = "wan2.1-vace-14b-i2v-fusionx"

        if is_local_path(payload['image']):
            payload['image'] = local_image_to_base64(payload['image'])
        return payload
    

# Create a subclass with the handlers
class WebAPI(GenericWebAPI):
    feature_handlers = {
        FeatureType.TEXT_TO_VIDEO: T2V,
        FeatureType.IMAGE_TO_VIDEO: I2V,
        FeatureType.INTERPOLATE: Interpolate,
        FeatureType.POSE_GUIDANCE: PoseGuidance,
    }
    @modal.fastapi_endpoint(method="POST")
    def web_inference(self, data: dict = Body(...)):
        feature_type = data.pop("feature_type", None)
        model = data.pop("model", None)  # Extract model from data
        if not feature_type:
            return {"error": f"A 'feature_type' is required. Must be one of: {list(self.feature_handlers.keys())}"}

        if feature_type not in self.feature_handlers:
            return {"error": f"Unknown feature_type: {feature_type}. Must be one of: {list(self.feature_handlers.keys())}"}

        # Route to appropriate class based on model and feature_type
        if feature_type == FeatureType.TEXT_TO_VIDEO and model == "wan2.1-vace-14b-phantom-fusionx":
            # Route to FusionXT2V for the fusionx model
            handler_class = PhantomFusionXT2V
        if feature_type == FeatureType.IMAGE_TO_VIDEO and model == "wan2.1-vace-14b-i2v-fusionx":
            handler_class = FusionXI2V
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

# Apply Modal decorator
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
    model=MODEL_NAME,
    provider="modal",
    implementation=Wan21VaceTextToVideo14B
)
registry.register(
    feature=FeatureType.IMAGE_TO_VIDEO,
    model=MODEL_NAME,
    provider="modal",
    implementation=Wan21VaceImageToVideo14B
)
registry.register(
    feature=FeatureType.INTERPOLATE,
    model=MODEL_NAME,
    provider="modal",
    implementation=Wan21VaceInterpolate14B
)
registry.register(
    feature=FeatureType.POSE_GUIDANCE,
    model=MODEL_NAME,
    provider="modal",
    implementation=Wan21VacePoseGuidance14B,
)
registry.register(
    feature=FeatureType.TEXT_TO_VIDEO,
    model="wan2.1-vace-14b-phantom-fusionx",
    provider="modal",
    implementation=Wan21VaceTextToVideo14BPhantomFusionX
)
registry.register(
    feature=FeatureType.IMAGE_TO_VIDEO,
    model="wan2.1-vace-14b-i2v-fusionx",
    provider="modal",
    implementation=Wan21VaceImageToVideo14BFusionX
)

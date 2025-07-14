import modal
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core.utils import prepare_video_and_mask, load_image_robust, is_local_path, local_image_to_base64, create_timestamp

# --- Configuration ---
APP_NAME = "test-wan-2.1-vace-t2v-14b"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f'{APP_NAME}-outputs'
OUTPUTS_PATH = "/outputs"

VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-14B-diffusers"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.1",
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
        "fastapi"
    ).env({"HF_HUB_CACHE": CACHE_PATH, "TOKENIZERS_PARALLELISM": "false"})
)

cache_volume = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)

app = modal.App(APP_NAME)

@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)
class T2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)
        
        flow_shift = 3.0
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        # for every value in data, pass into pipe
        frames = self.pipe(**data).frames[0]

        timestamp = create_timestamp()
        mp4_name = f"wan-vace-output_{timestamp}.mp4"
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
        t2v_instance = T2V()
        call = t2v_instance.generate.spawn(data)
        
        return JSONResponse({"call_id": call.object_id, "feature_type": "text_to_video"})

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
        payload["feature_type"] = "text_to_video"
        
        # print("payload: ", payload)
        return payload
    
@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)
class I2V:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)
        
        flow_shift = 5.0
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        import torch
        from diffusers.utils import export_to_video
        import io

        print("Starting video generation process...")

        # Define video parameters
        height = data.get('height', 480)
        width = data.get('width', 832)
        num_frames = data.get('num_frames', 81)

        # Prepare the data for the pipeline
        video, mask = prepare_video_and_mask(data.get('image'), height, width, num_frames)
        data.pop('image')

        # Run the diffusion pipeline
        output_frames = self.pipe(
            video=video,
            mask=mask,
            **data,
            generator=torch.Generator("cuda").manual_seed(42),
        ).frames[0]
        print("Pipeline execution finished.")

        timestamp = create_timestamp()
        mp4_name = f"wan21-vace-i2v-output_{timestamp}.mp4"
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
            return {"error": "A 'prompt' is required."}
        if not image:
            return {"error": "An 'image' is required."}
        
        # print(f"image_to_video - prompt: {prompt}")
        # print(f"image_to_video - image: {image}")
        # print(f"image_to_video - negative prompt: {negative_prompt}")
        
        try:
            image = load_image_robust(image)
            data['image'] = image
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}

        # Create I2V instance and call generate
        i2v_instance = I2V()
        call = i2v_instance.generate.spawn(data)
        
        return JSONResponse({"call_id": call.object_id, "feature_type": "image_to_video"})

class Wan21VaceImageToVideo14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "I2V"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Prepare payload specific to Wan2.1 Vace Image-to-Video model."""
        payload = super()._prepare_payload(required_args, **kwargs)
        payload["feature_type"] = "image_to_video"
        
        if is_local_path(payload['image']):
            payload['image'] = local_image_to_base64(payload['image'])
        return payload
   
@app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)
class Interpolate:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)
        
        flow_shift = 5.0
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        import torch
        print("Starting video generation process...")

        height = data.get('height', 512)
        width = data.get('width', 512)
        num_frames = data.get('num_frames', 81)
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
        mp4_name = f"wan21-vace-interpolate-output_{timestamp}.mp4"
        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output_frames, str(mp4_path), fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    @staticmethod
    def handle_web_inference(data: dict):
        """Handle frame interpolation."""  

        prompt = data.get("prompt")
        first_frame = data.get("first_frame")
        last_frame = data.get("last_frame")
        
        if not prompt:
            return {"error": "A 'prompt' is required."}
        if not first_frame or not last_frame:
            return {"error": "Both 'first_frame' and 'last_frame' are required."}
        
        try:
            # load_image_robust can handle both URLs and base64 strings
            first_frame = load_image_robust(first_frame)
            last_frame = load_image_robust(last_frame)   
            data['first_frame'] = first_frame
            data['last_frame'] = last_frame         
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}
        
        # print(f"interpolate - prompt: {prompt}")
        # print(f"interpolate - first frame: {first_frame}")
        # print(f"interpolate - last frame: {last_frame}")
        # print(f"interpolate - negative prompt: {negative_prompt}")
        
        # Create Interpolate instance and call generate
        interpolate_instance = Interpolate()
        call = interpolate_instance.generate.spawn(data)
        
        return JSONResponse({"call_id": call.object_id, "feature_type": "interpolate"})

class Wan21VaceInterpolate14B(ModalProviderBase):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.app_name = APP_NAME
        self.modal_app = app
        self.modal_class_name = "Interpolate"
    def _prepare_payload(self, required_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Prepare payload specific to Wan2.1 Vace Interpolate model.
        Break out required args into payload
        """
        payload = super()._prepare_payload(required_args, **kwargs)
        # payload = {"prompt": required_args['prompt']}
        payload["feature_type"] = "interpolate"
        
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
        "text_to_video": T2V,
        "image_to_video": I2V,
        "interpolate": Interpolate
    }

# Apply Modal decorator
WebAPI = app.cls(
    image=image,
    gpu=GPUType.A100_80GB.value,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: cache_volume, OUTPUTS_PATH: outputs_volume},
    timeout=1200,
    scaledown_window=900,
)(WebAPI)


# Register with the system registry
registry.register(
    feature="text_to_video",
    model="wan2.1-vace-14b",
    provider="modal",
    implementation=Wan21VaceTextToVideo14B
)
registry.register(
    feature="image_to_video",
    model="wan2.1-vace-14b",
    provider="modal",
    implementation=Wan21VaceImageToVideo14B
)
registry.register(
    feature="interpolate",
    model="wan2.1-vace-14b",
    provider="modal",
    implementation=Wan21VaceInterpolate14B
)



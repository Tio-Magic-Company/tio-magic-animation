import modal
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse

from .base import GPUType, GenericWebAPI, ModalProviderBase
from typing import Any, Dict
from ...core.registry import registry
from ...core.utils import prepare_video_and_mask, load_image_robust, is_local_path, local_image_to_base64, create_timestamp, load_video_robust, extract_image_dimensions
from ...core.feature_types import FeatureType
from ...core.schemas import FEATURE_SCHEMAS

# --- Configuration ---
APP_NAME = "test-wan-2.1-vace-t2v-14b"
MODEL_NAME = "wan2.1-vace-14b"
CACHE_NAME = f"{APP_NAME}-cache"
CACHE_PATH = "/cache"
OUTPUTS_NAME = f'{APP_NAME}-outputs'
OUTPUTS_PATH = "/outputs"

VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-14B-diffusers"

GPU_CONFIG: GPUType = GPUType.A100_80GB
TIMEOUT: int = 1800 # 30 minutes
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
        "python-dotenv"
    ).run_commands("pip install easy-dwpose --no-deps")
    .env({"HF_HUB_CACHE": CACHE_PATH, "TOKENIZERS_PARALLELISM": "false"})
)

cache_volume = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)

app = modal.App(APP_NAME)

def app_class_factory(
        base_class,
        *,
        gpu=GPU_CONFIG,
        timeout=TIMEOUT,
        scaledown_window=SCALEDOWN_WINDOW,
        image=image,
        app=app,
        cache_path=CACHE_PATH,
        cache_volume=cache_volume,
        outputs_path=OUTPUTS_PATH,
        outputs_volume=outputs_volume,
):
    return app.cls(
        image=image,
        gpu=gpu,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        volumes={cache_path: cache_volume, outputs_path: outputs_volume},
        timeout=timeout,
        scaledown_window=scaledown_window,
    )(base_class)

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

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
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
        prompt = data.get("prompt")

        if not prompt:
            return {"error": "A 'prompt' is required."}

        print(f"text_to_video - prompt: {prompt}")

        print("handle web inference data: ", data)
        # Create T2V instance and call generate
        t2v_instance = T2VAppClass()
        call = t2v_instance.generate.spawn(data)

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.TEXT_TO_VIDEO})
T2VAppClass = app_class_factory(T2V)

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

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        import torch
        from diffusers.utils import export_to_video
        import io
        print("***MODAL GENERATE METHOD***")
        print("Starting video generation process...")

        i2v_schema = FEATURE_SCHEMAS["image_to_video"][MODEL_NAME]

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
            if 'height' not in data and 'width' not in data:
                data = extract_image_dimensions(image, data)
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}

        # Create I2V instance and call generate
        i2v_instance = I2VAppClass()
        call = i2v_instance.generate.spawn(data)

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.IMAGE_TO_VIDEO})
I2VAppClass = app_class_factory(I2V)

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

        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from diffusers.utils import export_to_video
        import torch
        print("Starting video generation process...")

        interpolate_schema = FEATURE_SCHEMAS["interpolate"][MODEL_NAME]
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

        prompt = data.get("prompt")
        first_frame = data.get("first_frame")
        last_frame = data.get("last_frame")

        if not prompt:
            return {"error": "A 'prompt' is required."}
        if not first_frame or not last_frame:
            return {"error": "Both 'first_frame' and 'last_frame' are required."}

        try:
            # load_image_robust can handle both URLs and base64 strings, return PIL.Image
            first_frame = load_image_robust(first_frame)
            last_frame = load_image_robust(last_frame)   
            data['first_frame'] = first_frame
            data['last_frame'] = last_frame
            if 'height' not in data and 'width' not in data:
                data = extract_image_dimensions(first_frame, data)
        except Exception as e:
            return {"error": f"Error processing images: {str(e)}"}

        # Create Interpolate instance and call generate
        interpolate_instance = InterpolateAppClass()
        call = interpolate_instance.generate.spawn(data)

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.INTERPOLATE})
InterpolateAppClass = app_class_factory(Interpolate)

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

class PoseGuidance:
    @modal.enter()
    def load_models(self):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        # from controlnet_aux import OpenposeDetector
        from easy_dwpose import DWposeDetector
        print("Loading models into GPU memory...")
        self.vae = AutoencoderKLWan.from_pretrained(VACE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(VACE_MODEL_ID, vae=self.vae, torch_dtype=torch.bfloat16)

        flow_shift = 3.0
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        self.pipe.to("cuda")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.open_pose = DWposeDetector(device=device)
        print("✅ Models loaded successfully.")
    @modal.method()
    def generate(self, data: Dict[str, Any]):
        from PIL import Image
        import PIL.Image
        import torch
        from diffusers.utils import export_to_video, load_video
        import io
        import tempfile

        pose_guidance_schema = FEATURE_SCHEMAS["pose_guidance"][MODEL_NAME]

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
            return {"error": f"Error processing images: {str(e)}"}

        try:
            video_fields = ['guiding_video', 'pose_video']
            for field in video_fields:
                if field in data:
                    # convert URL -> bytes, base64 -> bytes
                    video = load_video_robust(data[field])
                    data[field] = video
        except Exception as e:
            return{"error": f"Error processing base64 video: {str(e)}"}

        pose_guidance_instance = PoseGuidanceAppClass()
        call = pose_guidance_instance.generate.spawn(data)

        return JSONResponse({"call_id": call.object_id, "feature_type": FeatureType.POSE_GUIDANCE})
PoseGuidanceAppClass = app_class_factory(PoseGuidance)

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

# Create a subclass with the handlers
class WebAPI(GenericWebAPI):
    feature_handlers = {
        FeatureType.TEXT_TO_VIDEO: T2V,
        FeatureType.IMAGE_TO_VIDEO: I2V,
        FeatureType.INTERPOLATE: Interpolate,
        FeatureType.POSE_GUIDANCE: PoseGuidance,
    }
# Apply Modal decorator
WebAPIClass = app_class_factory(WebAPI)


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



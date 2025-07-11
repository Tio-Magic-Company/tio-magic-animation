# Modal file 

# tm.text_to_video(prompt="", model="wan_2.1")
# t2v accepts: wan2.1, wan2.1_1.3B, wan2.1_14B

import modal
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse
# get user args passed in via wan_2_1 from adapters

VOLUME_NAME = "wan2.1-t2v-14b-cache"
CACHE_DIR = "/cache" 
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

OUTPUTS_NAME = "wan2.1-t2v-14b-outputs"
OUTPUTS_PATH = Path("/outputs")

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
        "fastapi",
    ).env({"HF_HUB_CACHE": CACHE_DIR})
)

cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_NAME, create_if_missing=True)

app = modal.App(
    "wan2.1-t2v-14b",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

@app.cls(
    image=image,
    gpu="H100", 
    volumes={CACHE_DIR: cache_volume, OUTPUTS_PATH: outputs_volume}, 
    timeout=600, #10 minutes
    scaledown_window=900 #stay idle for 15 mins before scaling down
)
class Wan21T2V14B:
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
    def generate(self, prompt: str, negative_prompt: str):
        """
        This method runs the text-to-video generation on an existing container.
        """
        from diffusers.utils import export_to_video
        from datetime import datetime

        print("Generating video...")
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=81,
            guidance_scale=5.0,
        ).frames[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp4_name = f"wan2.1-t2v-14b-pipeline-output_{timestamp}.mp4"

        mp4_path = Path(OUTPUTS_PATH) / mp4_name
        export_to_video(output, mp4_path, fps=16)
        outputs_volume.commit()

        with open(mp4_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    
    @modal.fastapi_endpoint(method="POST")
    def web_inference(self, data: dict = Body(...)):
        """
        FastAPI endpoint that runs on the class instance.
        """

        print("Enter async fastapi call")
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt", "") # Added default

        if not prompt:
            return {"error": "A 'prompt' is required."}
        
        print(f"prompt: {prompt}")
        print(f"negative prompt: {negative_prompt}")

        call = self.generate.spawn(prompt, negative_prompt)

        return JSONResponse({"call_id": call.object_id})

    @modal.fastapi_endpoint(method="GET")
    def get_result(self, call_id: str):
        """
        FastAPI endpoint to poll for the result of a generation call.
        """
        import io
        from modal import FunctionCall

        print(f"Polling for call_id: {call_id}")
        try:
            call = FunctionCall.from_id(call_id)
            video_bytes = call.get(timeout=0) # Use a short timeout to check for completion
        except TimeoutError:
            return JSONResponse({"status": "processing"}, status_code=202)
        except Exception as e:
            print(f"Error fetching result for {call_id}: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

        filename = f"wan2.1-t2v-14b-output_{call_id}.mp4"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4", headers=headers)



# --- Local Test Entrypoint (for `modal run`) ---
# @app.local_entrypoint()
# def main():
#     """
#     This function allows you to test the video generation from your terminal.
#     """
#     from datetime import datetime
#     print("🧪 Starting local test run...")
    
#     prompt = "A dramatic shot of a tortoise winning a race against a hare in a photorealistic style"
#     negative_prompt = "cartoon, drawing, anime, text, watermark"

#     # Create a remote instance of the WanT2V class
#     model = WanT2V()
    
#     print("Starting local generate")
#     # Call the generate method remotely
#     video_bytes = model.generate.remote(prompt, negative_prompt)
#     print("Complete local generate")
  
#     # Save the generated video to a local file
#     timestamp = datetime.now().strftime("%m-%d_%I%M%p")
#     output_filename = f"wan_t2v_output_{timestamp}.mp4"
#     with open(output_filename, "wb") as f:
#         f.write(video_bytes)
#     print(f"✅ Video saved to {output_filename}")
#     return





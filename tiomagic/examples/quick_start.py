import base64
from pathlib import Path
from tiomagic import tm

def text_to_video_example():
    tm.configure(provider="modal")
    prompt = "A cartoon robot dancing to music in a futuristic city"
    required_args = {
        "prompt": prompt,
    } 
    optional_args = {
        "negative_prompt": "blurry, low quality, realism"
    }
    tm.text_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)

def image_to_video_example():
    tm.configure(provider="modal")
    # required_args = {
    #     "prompt" : "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective.",
    #     "image" : "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png",
    # }
    required_args = {
        "prompt" : "Cartoon-styled Bob Ross painting on his canvas",
        "image" : "/Users/karenlin/Documents/tm-animation/tio-magic-animation/start.png"
    }
    optional_args = {
        "negative_prompt": "blurry, low quality, getty"
    }
    tm.image_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)

def interpolate_example():
    tm.configure(provider="modal")
    # first_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
    # last_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
    # prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

    # first_frame = "https://i.imgur.com/OVXR90G.jpeg"
    # last_frame = "https://i.imgur.com/SVL4dPc.png"
    prompt = "Cartoon-styled Bob Ross painting on his canvas, turns towards camera and smiles."
    
    # Convert local images to base64
    # def image_to_base64(image_path):
    #     with open(image_path, "rb") as f:
    #         return base64.b64encode(f.read()).decode('utf-8')
    
    # Use local image files
    first_frame = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/start.png"
    last_frame = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/end.png"
    
    
    required_args = {
        "prompt" : prompt,
        "first_frame": first_frame,
        "last_frame": last_frame
    }
    optional_args = {
        "negative_prompt": "blurry, low quality, getty"
    }

    tm.interpolate(model="wan2.1-vace-14b", required_args=required_args, **optional_args)

def check_status(job_id: str):
    tm.configure(provider="modal")
    tm.check_generation_status(job_id)
    
def get_repo_root() -> Path:
    """Get the repository root directory."""
    # Start from current file's directory
    current_dir = Path(__file__).parent
    
    # Walk up until we find the repository root (look for .git, README.md, etc.)
    while current_dir != current_dir.parent:
        if (current_dir / ".git").exists() or (current_dir / "README.md").exists():
            return current_dir
        current_dir = current_dir.parent
    
    # Fallback: assume current working directory is repo root
    return Path.cwd()

if __name__ == "__main__":
    status = "4ee8e442-aff4-48e5-bacd-9bf77f0f368d"
    # text_to_video_example()
    # check_status(status)

    # image_to_video_example()
    check_status(status)
    # interpolate_example()


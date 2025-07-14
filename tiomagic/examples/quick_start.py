import base64
from pathlib import Path
from tiomagic import tm

def text_to_video_example():
    tm.configure(provider="modal")
    # prompt = "A serene morning in a dense pine forest, soft golden sunlight streaming through the misty trees, gentle rays creating a magical, peaceful atmosphere. Birds flutter between branches, and a small stream glistens as it flows over smooth rocks."
    # prompt = "A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape."
    prompt = "A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer."
    required_args = {
        "prompt": prompt,
    } 
    optional_args = {
        "negative_prompt": "blurry, low quality",
    }
    # tm.text_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
    tm.text_to_video(model="cogvideox-5b", required_args=required_args, **optional_args)

def image_to_video_example():
    tm.configure(provider="modal")
    # required_args = {
    #     "prompt" : "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective.",
    #     "image" : "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png",
    # }
    required_args = {
        "prompt" : "Cartoon-styled Bob Ross painting a tree on his canvas",
        "image" : "/Users/karenlin/Documents/tm-animation/tio-magic-animation/start.png"
    }
    optional_args = {
        "negative_prompt": "blurry, low quality, getty"
    }
    tm.image_to_video(model="cogvideox-5b-image-to-video", required_args=required_args, **optional_args)
    # tm.image_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)

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
        "negative_prompt": "blurry, low quality, getty",
    }

    tm.interpolate(model="wan2.1-vace-14b", required_args=required_args, **optional_args)

def check_status(job_id: str):
    tm.configure(provider="modal")
    tm.check_generation_status(job_id)
    

if __name__ == "__main__":
    status = "ec00f312-b9f4-4888-be9e-ce28e23518be"
    
    # text_to_video_example()
    # image_to_video_example()
    # interpolate_example()

    check_status(status)



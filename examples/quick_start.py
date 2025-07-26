from tiomagic import tm
from dotenv import load_dotenv

def text_to_video_example():
    tm.configure(provider="modal")
    prompt = "A serene morning in a dense pine forest, soft golden sunlight streaming through the misty trees, gentle rays creating a magical, peaceful atmosphere. Birds flutter between branches, and a small stream glistens as it flows over smooth rocks."
    # prompt = "A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape."
    # prompt = "A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer."
    required_args = {
        "prompt": prompt,
    } 
    optional_args = {
        "negative_prompt": "blurry, low quality",
    }
    tm.text_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
    # tm.text_to_video(model="cogvideox-5b", required_args=required_args, **optional_args)

def image_to_video_example():
    tm.configure(provider="modal")
    # tm.configure(provider="local")
    # prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
    # image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
    image = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/test_blob.png"
    prompt = "2D animation of a golden-colored blob wearing a crown moving to the right. The blob is looking around in amazement and moving its arms."

    # prompt = "A penguin dancing in the snow"
    # image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
    # prompt = "Fantastical flying pig flapping its wings and flying across an ancient city, the background moving quickly due to the pig's speed"
    # image = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/flying-pig.png'
    required_args = {
        "prompt" : prompt,
        "image" : image
    }
    optional_args = {
        "negative_prompt": "blurry, low quality, getty",
    }

    # tm.image_to_video(model="framepack-i2v-hy", required_args=required_args, **optional_args)
    # tm.image_to_video(model="cogvideox-5b-image-to-video", required_args=required_args, **optional_args)
    tm.image_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
    # tm.image_to_video(model="wan2.1-i2v-14b-720p", required_args=required_args, **optional_args)
    # tm.image_to_video(model="veo-2.0-generate-001", required_args=required_args, **optional_args)

def interpolate_example():
    tm.configure(provider="modal")
    # Use URL
    # first_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
    # last_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
    # prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

    first_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/start.png'
    last_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/end.png'
    prompt = "Cartoon styled painter Bob ross paints a tree on his canvas, turns towards the camera and smiles to the audience"
    required_args = {
        "prompt" : prompt,
        "first_frame": first_frame,
        "last_frame": last_frame
    }
    optional_args = {
        "negative_prompt": "blurry, low quality, getty",
    }

    # tm.interpolate(model="framepack-i2v-hy", required_args=required_args, **optional_args)
    tm.interpolate(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
    # tm.interpolate(model="wan2.1-flf2v-14b-720p", required_args=required_args, **optional_args)

def pose_guidance_example():
    tm.configure(provider="modal")
    # start image
    image = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/aditya-base-dance.png"

    # guiding video OR pose video
    guiding_video = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/videos/driving-dance.mp4"

    # prompt
    prompt = "3d pixar cartoon animation of a man dancing, empty white background, disney pixar style"

    negative_prompt = "blurry, low quality, worst quality"

    required_args = {
        "image": image,
        "prompt": prompt
    }

    optional_args = {
        "guiding_video": guiding_video,
        "negative_prompt": negative_prompt
    }

    tm.pose_guidance(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
def check_status(job_id: str):
    # tm.configure(provider="modal")
    tm.check_generation_status(job_id)
def cancel_job(job_id: str):
    tm.cancel_job(job_id)

if __name__ == "__main__":
    load_dotenv()
    # check for keys in appropriate provider files

    status = "b8dbbfde-15b9-4788-ad0d-f1d60563f2d0"

    # text_to_video_example()
    # image_to_video_example()
    # interpolate_example()
    # pose_guidance_example()

    # check_status(status)
    cancel_job(status)



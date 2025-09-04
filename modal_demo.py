from tiomagic import tm

def text_to_video_example():
    tm.configure(provider="modal")
    prompt = "A playful cartoon-style penguin with a round belly and flappy wings waddles up to a pair of green sunglasses lying on the ground. The penguin leans forward, carefully picks up the sunglasses with its flipper, and smoothly lifts them up to its face. It tilts its head with a confident smile as the green sunglasses rest perfectly on its beak. The animation is smooth and expressive, with exaggerated, bouncy cartoon motion."
    negative_prompt = "No realism, no photo style, no glitches, no broken limbs, no stiff motion, no blur, no duplicate objects, no missing sunglasses, no frame flickering."

    required_args = {
        "prompt": prompt,
    } 
    optional_args = {
        "negative_prompt": negative_prompt,
    }
    tm.text_to_video(model="wan2.1-t2v-14b", required_args=required_args, **optional_args)

def image_to_video_example():
    tm.configure(provider="modal")
    prompt = "Cartoon-style woman smiling at the camera, waving her right hand as if she was saying hi and greeting someone."
    image = "sample-image.png"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    required_args = {
        "prompt": prompt,
        "image": image,
    } 
    optional_args = {
        "negative_prompt": negative_prompt,
        "num_frames": 81
    }
    tm.image_to_video(model="ltx-video", required_args=required_args, **optional_args)

def wan22_fflf_example():
    tm.configure(provider="modal")
    prompt = "2d animation of a cartoon that goes from day to night"
    first_frame = "/Users/jmportilla/Downloads/start_racoon.png"
    last_frame = "/Users/jmportilla/Downloads/end_racoon.png"

    required_args = {
        "prompt": prompt,
        "first_frame": first_frame,
        "last_frame":last_frame
    } 
    optional_args = {
        
    }
    tm.interpolate(model="wan2.2-flf2v-a14b", required_args=required_args, **optional_args)

def check_status(job_id: str):
    tm.check_generation_status(job_id)

def cancel_job(job_id: str):
    tm.cancel_job(job_id)

def list_implementations():
    tm.list_implementations()


if __name__ == "__main__":
    # image_to_video_example()
    wan22_fflf_example()
    # list_implementations()





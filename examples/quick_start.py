from tiomagic import tm
from dotenv import load_dotenv

def text_to_video_example():
    tm.configure(provider="modal")
    # tm.configure(provider="local")
    # prompt = "the playful cartoon penguin picks up the green sunglasses and puts them on."
    # prompt = "A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape."
    # prompt = "A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer."
    # prompt = "A cheerful cartoon girl in a red dress skips through a sunny meadow, laughing as a chubby orange cat playfully chases a butterfly beside her. The sky is bright blue, flowers sway in the breeze, and everything moves with a soft, animated bounce."
    # prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    # input_prompt = "a cinematic cartoon shot of a hamster eating a tiny burrito"
    # n_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, deformed, disfigured, still picture, messy background, many people in the background, walking backwards"
    # prompt = "A young man wearing a simple t-shirt and jeans begins standing upright with his arms relaxed at his sides. Then, he raises both arms straight up above his head in a clean, symmetrical motion, with a neutral facial expression. His body remains upright, and his posture is steady, showing a smooth upward movement of both arms."
    prompt = "A young man wearing a simple t-shirt and jeans begins in a standing position with a neutral stance and expression. In the final frame, he is seated on an invisible surface with his knees bent at 90 degrees, arms resting naturally on his legs. His body lowers smoothly without twisting, and his upper body remains upright throughout the transition."
    
    required_args = {
        "prompt": prompt,
    } 
    optional_args = {
        "negative_prompt": negative_prompt
    }
    # tm.text_to_video(model="pusa-v1", required_args=required_args, **optional_args)
    # tm.text_to_video(model="wan2.1-t2v-14b", required_args=required_args, **optional_args)
    tm.text_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
    # tm.text_to_video(model="cogvideox-5b", required_args=required_args, **optional_args)
    # tm.text_to_video(model="wan2.1-14b-t2v-fusionx", required_args=required_args, **optional_args)
    # tm.text_to_video(model="wan2.1-vace-14b-phantom-fusionx", required_args=required_args, **optional_args)
    # tm.text_to_video(model="wan2.2-t2v-a14b", required_args=required_args, **optional_args)

def image_to_video_example():
    tm.configure(provider="modal")
    # tm.configure(provider="local")
    # prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
    # image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
    # image = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/test_blob.png"
    # prompt = "2D animation of a golden-colored blob wearing a crown moving to the right. The blob is looking around in amazement and moving its arms."

    # prompt = "A penguin dancing in the snow"
    # image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
    
    # prompt = "Fantastical flying pig flapping its wings and flying across an ancient city, the background moving quickly due to the pig's speed"
    # image = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/flying-pig.png'
    
    # prompt = "Cartoon-style woman smiling at the camera, waving her right hand as if she was saying hi and greeting someone."
    image = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/girl_waving.png'

    # image = "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
    
    # prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background. Flames engulf the structure, with smoke billowing into the air. Firefighters in protective gear rush to the scene, a fire truck labeled '38' visible behind them. The girl's neutral expression contrasts sharply with the chaos of the fire, creating a poignant and emotionally charged scene."
    # negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    # prompt = "Follow the written instructions on the source image for animating the image. A lunar vehicle drives in from the left side of the frame. The astronauts walk up to the lunar vehicle and hop into the lunar vehicle. The Aurora borealis effect glimmers in the sly. In the background, on the right side of the scene, a VTOL craft lands in the background."
    # image = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/image_prompt.png'

    input_prompt = "Disney cartoon-style girl smiling at the camera, waving her right hand as if she was saying hi and greeting someone."
    n_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    required_args = {
        "prompt" : input_prompt,
        "image" : image
    }
    optional_args = {
        "negative_prompt": n_prompt,
        # "height": 512,
        # "width": 704
        # "negative_prompt": "blurry, low quality, getty, sped up",
        # "height": 480,
        # "width": 832
    }

    # tm.image_to_video(model="wan-14b-i2v-fusionx", required_args=required_args, **optional_args)
    # tm.image_to_video(model="wan2.1-i2v-14b-720p", required_args=required_args, **optional_args)
    # tm.image_to_video(model="pusa-v1", required_args=required_args, **optional_args)
    # tm.image_to_video(model="framepack-i2v-hy", required_args=required_args, **optional_args)
    # tm.image_to_video(model="cogvideox-5b-image-to-video", required_args=required_args, **optional_args)
    # tm.image_to_video(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
    # tm.image_to_video(model="wan2.1-vace-14b-i2v-fusionx", required_args=required_args, **optional_args)
    
    # tm.image_to_video(model="ltx-video", required_args=required_args, **optional_args)
    # tm.image_to_video(model="wan2.1-vace-14b-phantomfusionx", required_args=required_args, **optional_args)
    # tm.image_to_video(model="veo-2.0-generate-001", required_args=required_args, **optional_args)
    tm.image_to_video(model="wan2.2-ti2v-a14b", required_args=required_args, **optional_args)

def interpolate_example():
    tm.configure(provider="modal")
    # Use URL
    # first_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
    # last_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
    # prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

    # first_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/start.png'
    # last_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/end.png'
    # prompt = "Cartoon styled painter Bob ross paints a tree on his canvas, turns towards the camera and smiles to the audience"
    
    # first_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/cat_start.png'
    # last_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/cat_last.png'
    # prompt = "Cartoon style, a cute orange cat runs and pounces forward, landing and lowering its body down on the floor, tail gently curing as it relaxes"

    first_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/int_start.png'
    last_frame = '/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/int_end.png'
    prompt = "A cartoon-style young man in a blue t-shirt and jeans starts in a standing position with one hand waving high and a cheerful smile. He bends his knees and jumps into the air. In the final frame, he is mid-jump with both feet off the ground, knees tucked, arms raised in excitement, and his mouth open in a joyful shout. His hair and shirt ripple with the motion, capturing the energy of a spontaneous leap."
    negative_prompt = "No distortion, no extra limbs, no broken joints, no unrealistic body proportions, no missing fingers, no floating artifacts, no glitching clothing, no blurry motion, no stiff or unnatural poses, no duplicate faces, no background inconsistency, no text, no watermarks, no photorealism, no robotic or lifeless expressions."

    required_args = {
        "prompt" : prompt,
        "first_frame": first_frame,
        "last_frame": last_frame
    }
    optional_args = {
        "negative_prompt": negative_prompt,
        # "height": 480,
        # "width": 832,
        "num_frames": 49,
        "guidance_scale": 7.0
    }

    tm.interpolate(model="framepack-i2v-hy", required_args=required_args, **optional_args)
    # tm.interpolate(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
    # tm.interpolate(model="wan2.1-flf2v-14b-720p", required_args=required_args, **optional_args)
    

def pose_guidance_example():
    tm.configure(provider="modal")
    # start image
    image = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/images/aditya-base-dance.png"

    # guiding video OR pose video
    guiding_video = "/Users/karenlin/Documents/tm-animation/tio-magic-animation/videos/driving-dance.mp4"

    # prompt
    prompt = "3d pixar cartoon animation of a man dancing, empty white background, disney pixar style"

    negative_prompt = "No distortion, no extra limbs, no broken joints, no unrealistic body proportions, no missing fingers, no floating artifacts, no glitching clothing, no blurry motion, no stiff or unnatural poses, no duplicate faces, no background inconsistency, no text, no watermarks, no photorealism, no robotic or lifeless expressions."

    required_args = {
        "image": image,
        "prompt": prompt
    }

    optional_args = {
        "guiding_video": guiding_video,
        "negative_prompt": negative_prompt,
    }

    tm.pose_guidance(model="wan2.1-vace-14b", required_args=required_args, **optional_args)
def check_status(job_id: str):
    # tm.configure(provider="modal")
    tm.check_generation_status(job_id)
def cancel_job(job_id: str):
    tm.cancel_job(job_id)

def list_implementations():
    # get a list of all implementations (provider/feature/models) available
    # provider -> feature -> models
    tm.list_implementations()


if __name__ == "__main__":
    load_dotenv()
    # check for keys in appropriate provider files

    status = "2adf62d6-0f5e-42a9-9412-dc2c0575cb4e"

    text_to_video_example()
    # image_to_video_example()
    # interpolate_example()
    # pose_guidance_example()

    # check_status(status)
    # cancel_job(status)
    # list_implementations()



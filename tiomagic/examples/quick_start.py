from tiomagic import tm

def text_to_video_example():
    tm.configure(provider="modal")
    prompt = "A cartoon robot dancing to music in a futuristic city"
    additional_args = {
        "negative_prompt": "blurry, low quality, realism"
    }
    tm.text_to_video(prompt, model="wan2.1-vace-14b", **additional_args)

def image_to_video_example():
    tm.configure(provider="modal")
    image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
    prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
    additional_args = {
        "negative_prompt": "blurry, low quality, getty"
    }
    tm.image_to_video(image=image, prompt=prompt, model="wan2.1-vace-14b", **additional_args)

def interpolate_example():
    tm.configure(provider="modal")
    start_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
    last_frame = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
    prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
    additional_args = {
        "negative_prompt": "blurry, low quality, getty"
    }
    tm.interpolate(start_frame, last_frame, prompt, "wan2.1-vace-14b", **additional_args)

def check_status(job_id: str):
    tm.configure(provider="modal")
    tm.check_generation_status(job_id)
    
if __name__ == "__main__":
    status = "dfbb5dfa-bc22-40a8-8c3b-59be9f07f4f6"
    # text_to_video_example()
    # check_status(status)

    # image_to_video_example()
    check_status(status)
    # interpolate_example()


"""Parameter schemas for pose guidance models."""

# Schema directory organized by model name
SCHEMAS = {
    "wan2.1-vace-14b": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
            "image": {"type": str, "description": "Path or URL to input image"}
            # a guiding_video or pose_video must be included for pose_guidance
        },
        "optional": {
            "guiding_video": {"type": str, "description": "A video to guide the pose of the output video. If provided, a pose_video will be generated for the output video (List[PIL.Image.Image])"},
            "pose_video": {"type": str, "description": "A pose skeleton video to guide the pose of the output video (List[PIL.Image.Image])"},
            "negative_prompt": {"type": str, "default": "", "description": "The prompt or prompts not to guide the image generation. Ignored when guidance_scale is less than 1."},
            "video": {"type": list, "description": "The input video (List[PIL.Image.Image]) to be used as a starting point for the generation."},
            "mask": {"type": list, "description": "The input mask (List[PIL.Image.Image]) that defines which video regions to condition on (black) and which to generate (white)."},
            "reference_images": {"type": list, "description": "A list of one or more reference images (List[PIL.Image.Image]) as extra conditioning for the generation."},
            "conditioning_scale": {"type": float, "default": 1.0, "description": "The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor."},
            "height": {"type": int, "default": 480, "description": "The height in pixels of the generated video."},
            "width": {"type": int, "default": 832, "description": "The width in pixels of the generated video."},
            "num_frames": {"type": int, "default": 81, "description": "The number of frames in the generated video."},
            "num_inference_steps": {"type": int, "default": 50, "description": "The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference."},
            "guidance_scale": {"type": float, "default": 5.0, "description": "Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt."},
            "num_videos_per_prompt": {"type": int, "default": 1, "description": "The number of videos to generate per prompt."},
            "generator": {"type": "torch.Generator", "description": "A torch.Generator or List[torch.Generator] to make generation deterministic."},
            "latents": {"type": "torch.Tensor", "description": "Pre-generated noisy latents to be used as inputs for generation."},
            "prompt_embeds": {"type": "torch.Tensor", "description": "Pre-generated text embeddings. Can be used to easily tweak text inputs."},
            "output_type": {"type": str, "default": "np", "description": "The output format of the generated image. Choose between 'PIL.Image' or 'np.array'."},
            "return_dict": {"type": bool, "default": True, "description": "Whether or not to return a WanPipelineOutput instead of a plain tuple."},
            "attention_kwargs": {"type": dict, "description": "A kwargs dictionary passed along to the AttentionProcessor."},
            "callback_on_step_end": {"type": "Callable", "description": "A function called at the end of each denoising step during inference."},
            "callback_on_step_end_tensor_inputs": {"type": list, "description": "The list of tensor inputs for the callback_on_step_end function."},
            "max_sequence_length": {"type": int, "default": 512, "description": "The maximum sequence length of the text encoder."},
            "flow_shift": {"type": float, "default": 3.0, "description": "a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability."}
        }
    },
}
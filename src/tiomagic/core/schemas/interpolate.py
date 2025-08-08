"""Parameter schemas for interpolation models."""
"""Map model to schema"""

# Schema dictionary organized by model name
SCHEMAS = {
    "wan2.1-flf2v-14b-720p": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
            "first_frame": {"type": str, "description": "Path or URL to first frame"},
            "last_frame": {"type": str, "description": "Path or URL to last frame"}
        },
        "optional": {
            "negative_prompt": {"type": str, "description": "The prompt or prompts not to guide the image generation. Ignored if guidance_scale is less than 1."},
            "height": {"type": int, "default": 480, "description": "The height of the generated video in pixels."},
            "width": {"type": int, "default": 832, "description": "The width of the generated video in pixels."},
            "num_frames": {"type": int, "default": 81, "description": "The number of frames in the generated video."},
            "num_inference_steps": {"type": int, "default": 50, "description": "The number of denoising steps. More steps can improve quality but are slower."},
            "guidance_scale": {"type": float, "default": 5.0, "description": "Classifier-Free Diffusion guidance scale. Higher values align the video more closely with the prompt."},
            "num_videos_per_prompt": {"type": int, "default": 1, "description": "The number of videos to generate for each prompt."},
            "generator": {"type": "torch.Generator", "description": "A torch.Generator or List[torch.Generator] to make generation deterministic."},
            "latents": {"type": "torch.Tensor", "description": "Pre-generated noisy latents to be used as inputs for generation."},
            "prompt_embeds": {"type": "torch.Tensor", "description": "Pre-generated text embeddings, used as an alternative to the 'prompt' argument."},
            "negative_prompt_embeds": {"type": "torch.Tensor", "description": "Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument."},
            "image_embeds": {"type": "torch.Tensor", "description": "Pre-generated image embeddings, used as an alternative to the 'image' argument."},
            "output_type": {"type": str, "default": "np", "description": "The output format of the generated video. Choose between 'PIL.Image' or 'np.array'."},
            "return_dict": {"type": bool, "default": True, "description": "Whether to return a WanPipelineOutput object instead of a plain tuple."},
            "attention_kwargs": {"type": dict, "description": "A kwargs dictionary passed to the AttentionProcessor."},
            "callback_on_step_end": {"type": "Callable", "description": "A function called at the end of each denoising step during inference."},
            "callback_on_step_end_tensor_inputs": {"type": list, "description": "The list of tensor inputs for the callback_on_step_end function."},
            "max_sequence_length": {"type": int, "default": 512, "description": "The maximum sequence length for the text encoder."},
            "flow_shift": {"type": float, "default": 5.0, "description": "a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability."}
        }
    },
    "wan2.1-vace-14b": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
            "first_frame": {"type": str, "description": "Path or URL to first frame"},
            "last_frame": {"type": str, "description": "Path or URL to last frame"}
        },
        "optional": {
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
            "max_sequence_length": {"type": int, "default": 512, "description": "The maximum sequence length of the text encoder."}
        }
    },
    "framepack-i2v-hy": {
        "required": {
            # "image": {"type": "Image", "description": "The starting image for the video generation. Accepts PIL.Image, np.ndarray, or torch.Tensor."}
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
            "first_frame": {"type": str, "description": "Path or URL to first frame"},
            "last_frame": {"type": str, "description": "Path or URL to last frame"}
        },
        "optional": {
            # "last_image": {"type": "Image", "description": "The optional ending image for the video generation, useful for image-to-image transitions."},
            # "prompt": {"type": str, "description": "The prompt to guide video generation."},
            "prompt_2": {"type": str, "description": "A secondary prompt for the second text encoder; defaults to the main prompt if not provided."},
            "negative_prompt": {"type": str, "description": "The prompt to avoid during video generation."},
            "negative_prompt_2": {"type": str, "description": "A secondary negative prompt for the second text encoder."},
            "height": {"type": int, "default": 720, "description": "The height in pixels of the generated video."},
            "width": {"type": int, "default": 1280, "description": "The width in pixels of the generated video."},
            "num_frames": {"type": int, "default": 129, "description": "The number of frames in the generated video."},
            "num_inference_steps": {"type": int, "default": 50, "description": "The number of denoising steps."},
            "sigmas": {"type": list, "description": "Custom sigmas for the denoising scheduler."},
            "true_cfg_scale": {"type": float, "default": 1.0, "description": "Enables true classifier-free guidance when > 1.0."},
            "guidance_scale": {"type": float, "default": 6.0, "description": "Guidance scale to control how closely the video adheres to the prompt."},
            "num_videos_per_prompt": {"type": int, "default": 1, "description": "The number of videos to generate per prompt."},
            "generator": {"type": "torch.Generator", "description": "A torch.Generator to make generation deterministic."},
            "image_latents": {"type": "torch.Tensor", "description": "Pre-encoded image latents, bypassing the VAE for the first image."},
            "last_image_latents": {"type": "torch.Tensor", "description": "Pre-encoded image latents, bypassing the VAE for the last image."},
            "prompt_embeds": {"type": "torch.Tensor", "description": "Pre-generated text embeddings, an alternative to 'prompt'."},
            "pooled_prompt_embeds": {"type": "torch.FloatTensor", "description": "Pre-generated pooled text embeddings."},
            "negative_prompt_embeds": {"type": "torch.FloatTensor", "description": "Pre-generated negative text embeddings, an alternative to 'negative_prompt'."},
            "negative_pooled_prompt_embeds": {"type": "torch.FloatTensor", "description": "Pre-generated negative pooled text embeddings."},
            "output_type": {"type": str, "default": "pil", "description": "The output format of the generated video frames ('pil' or 'np.array')."},
            "return_dict": {"type": bool, "default": True, "description": "Whether to return a HunyuanVideoFramepackPipelineOutput object instead of a plain tuple."},
            "attention_kwargs": {"type": dict, "description": "A kwargs dictionary passed to the AttentionProcessor."},
            "clip_skip": {"type": int, "description": "Number of final layers to skip from the CLIP model."},
            "callback_on_step_end": {"type": "Callable", "description": "A function called at the end of each denoising step."},
            "callback_on_step_end_tensor_inputs": {"type": list, "description": "The list of tensor inputs for the callback_on_step_end function."}
        }
    },
    "luma-ray-2": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
            "first_frame": {"type": str, "description": "URL to first frame"},
            "last_frame": {"type": str, "description": "URL to last frame"}
        },
        "optional": {}
    },

    # Add more models as needed
}
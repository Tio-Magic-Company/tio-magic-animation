"""Parameter schemas for text-to-video models."""

# Schema dictionary organized by model name
SCHEMAS = {
    "wan2.1-t2v-14b": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
        },
        "optional": {
            "negative_prompt": {"type": str, "description": "The prompt or prompts to avoid during image generation. Ignored when guidance_scale is less than 1."},
            "height": {"type": int, "default": 480, "description": "The height in pixels of the generated image."},
            "width": {"type": int, "default": 832, "description": "The width in pixels of the generated image."},
            "num_frames": {"type": int, "default": 81, "description": "The number of frames in the generated video."},
            "num_inference_steps": {"type": int, "default": 50, "description": "The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference."},
            "guidance_scale": {"type": float, "default": 5.0, "description": "Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt."},
            "num_videos_per_prompt": {"type": int, "default": 1, "description": "The number of videos to generate per prompt."},
            "generator": {"type": "torch.Generator", "description": "A torch.Generator or List[torch.Generator] to make generation deterministic."},
            "latents": {"type": "torch.Tensor", "description": "Pre-generated noisy latents to be used as inputs for image generation."},
            "prompt_embeds": {"type": "torch.Tensor", "description": "Pre-generated text embeddings, used as an alternative to the 'prompt' argument."},
            "output_type": {"type": str, "default": "np", "description": "The output format of the generated image. Choose between 'PIL.Image' or 'np.array'."},
            "return_dict": {"type": bool, "default": True, "description": "Whether or not to return a WanPipelineOutput object instead of a plain tuple."},
            "attention_kwargs": {"type": dict, "description": "A kwargs dictionary passed to the AttentionProcessor."},
            "callback_on_step_end": {"type": "Callable", "description": "A function called at the end of each denoising step during inference."},
            "callback_on_step_end_tensor_inputs": {"type": list, "description": "The list of tensor inputs for the callback_on_step_end function."},
            "max_sequence_length": {"type": int, "default": 512, "description": "The maximum sequence length of the text encoder."}
        }
    },
    "wan2.1-14b-t2v-fusionx": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
        },
        "optional": {
            "negative_prompt": {"type": str, "description": "The prompt or prompts to avoid during image generation. Ignored when guidance_scale is less than 1."},
            "height": {"type": int, "default": 480, "description": "The height in pixels of the generated image."},
            "width": {"type": int, "default": 832, "description": "The width in pixels of the generated image."},
            "num_frames": {"type": int, "default": 81, "description": "The number of frames in the generated video."},
            "num_inference_steps": {"type": int, "default": 50, "description": "The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference."},
            "guidance_scale": {"type": float, "default": 5.0, "description": "Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt."},
            "num_videos_per_prompt": {"type": int, "default": 1, "description": "The number of videos to generate per prompt."},
            "generator": {"type": "torch.Generator", "description": "A torch.Generator or List[torch.Generator] to make generation deterministic."},
            "latents": {"type": "torch.Tensor", "description": "Pre-generated noisy latents to be used as inputs for image generation."},
            "prompt_embeds": {"type": "torch.Tensor", "description": "Pre-generated text embeddings, used as an alternative to the 'prompt' argument."},
            "output_type": {"type": str, "default": "np", "description": "The output format of the generated image. Choose between 'PIL.Image' or 'np.array'."},
            "return_dict": {"type": bool, "default": True, "description": "Whether or not to return a WanPipelineOutput object instead of a plain tuple."},
            "attention_kwargs": {"type": dict, "description": "A kwargs dictionary passed to the AttentionProcessor."},
            "callback_on_step_end": {"type": "Callable", "description": "A function called at the end of each denoising step during inference."},
            "callback_on_step_end_tensor_inputs": {"type": list, "description": "The list of tensor inputs for the callback_on_step_end function."},
            "max_sequence_length": {"type": int, "default": 512, "description": "The maximum sequence length of the text encoder."}
        }
    },
    "wan2.1-vace-14b": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
        },
        "optional": {
            "negative_prompt": {"type": str, "default": "", "description": "The prompt or prompts not to guide the image generation. Ignored when guidance_scale is less than 1."},
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
    "cogvideox-5b": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
        },
        "optional": {
            "negative_prompt": {"type": str, "description": "The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1."},
            "height": {"type": int, "default": 480, "description": "The height in pixels of the generated video."},
            "width": {"type": int, "default": 720, "description": "The width in pixels of the generated video."},
            "num_frames": {"type": int, "default": 48, "description": "Number of frames to generate."},
            "num_inference_steps": {"type": int, "default": 50, "description": "The number of denoising steps. More steps can improve quality but are slower."},
            "timesteps": {"type": list, "description": "Custom timesteps to use for the denoising process, must be in descending order."},
            "guidance_scale": {"type": float, "default": 7.0, "description": "Classifier-Free Diffusion guidance scale. Higher values align the video more closely with the prompt."},
            "num_videos_per_prompt": {"type": int, "default": 1, "description": "The number of videos to generate for each prompt."},
            "generator": {"type": "torch.Generator", "description": "A torch.Generator or List[torch.Generator] to make generation deterministic."},
            "latents": {"type": "torch.FloatTensor", "description": "Pre-generated noisy latents to be used as inputs for generation."},
            "prompt_embeds": {"type": "torch.FloatTensor", "description": "Pre-generated text embeddings, used as an alternative to the 'prompt' argument."},
            "negative_prompt_embeds": {"type": "torch.FloatTensor", "description": "Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument."},
            "output_type": {"type": str, "default": "pil", "description": "The output format of the generated video. Choose between 'pil' or 'np.array'."},
            "return_dict": {"type": bool, "default": True, "description": "Whether to return a StableDiffusionXLPipelineOutput object instead of a plain tuple."},
            "attention_kwargs": {"type": dict, "description": "A kwargs dictionary passed to the AttentionProcessor."},
            "callback_on_step_end": {"type": "Callable", "description": "A function called at the end of each denoising step during inference."},
            "callback_on_step_end_tensor_inputs": {"type": list, "description": "The list of tensor inputs for the callback_on_step_end function."},
            "max_sequence_length": {"type": int, "default": 226, "description": "Maximum sequence length in the encoded prompt."}
        }
    },
    "pusa-v1": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
        },
        "optional": {
            "negative_prompt": {"type": str, "description": "The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1."},
        }
    },
    "wan2.1-vace-14b-phantom-fusionx": {
        "required": {
            "prompt": {"type": str, "description": "Text prompt to guide generation"},
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
            "max_sequence_length": {"type": int, "default": 512, "description": "The maximum sequence length of the text encoder."},
            "flow_shift": {"type": float, "default": 3.0, "description": "a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability."}
        }
    },
    # Add more models as needed
}

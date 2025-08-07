---
layout: default
title: Parameter Documentation
---

# Model Parameter Documentation

## Text to Video

### Cogvideox 5b

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| model | string | "cogvideox-5b" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 720 | The width in pixels of the generated video. |
| num_frames | int | 48 | Number of frames to generate. |
| num_inference_steps | int | 50 | The number of denoising steps. More steps can improve quality but are slower. |
| timesteps | list | | Custom timesteps to use for the denoising process, must be in descending order. |
| guidance_scale | float | 7.0 | Classifier-Free Diffusion guidance scale. Higher values align the video more closely with the prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| negative_prompt_embeds | torch.FloatTensor | | Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument. |
| output_type | str | pil | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a StableDiffusionXLPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 226 | Maximum sequence length in the encoded prompt. |

</details>

### Pusa V1

Because PusaV1 is relatively new, there is no central location for its required and optional arguments. Based on its examples, we have added some parameters to the optional arguments section.

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| model | string | "pusa-v1" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. |

</details>

### Wan 2.1 Text to Video 14b

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| model | string | "wan2.1-t2v-14b" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 832 | The width in pixels of the generated video. |
| num_frames | int | 81 | Number of frames in the generated video |
| num_inference_steps | int | 50 | The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference. |
| guidance_scale | float | 5.0 | Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| output_type | str | np | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a WanPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 512 | Maximum sequence length in the encoded prompt. |

</details>

### Wan 2.1 Vace 14b

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| model | string | "wan2.1-vace-14b" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 832 | The width in pixels of the generated video. |
| conditioning_scale | float | 1.0 | The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor. |
| num_frames | int | 81 | Number of frames in the generated video |
| num_inference_steps | int | 50 | The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference. |
| guidance_scale | float | 5.0 | Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| output_type | str | np | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a WanPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 512 | Maximum sequence length in the encoded prompt. |

</details>

### Wan 2.1 Vace 14b T2V FusionX

This is a LoRA applied on top of Wan 2.1 Vace 14b. All of the required arguments and optional arguments are the same as Wan 2.1 Vace 14b, except for the model string.

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| model | string | "wan-14b-vace-t2v-fusionx" |

</details>

### Wan 2.1 Vace 14b Phantom FusionX

This is a LoRA applied on top of Wan 2.1 Vace 14b. All of the required arguments and optional arguments are the same as Wan 2.1 Vace 14b, except for the model string.

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| model | string | "wan2.1-vace-14b-phantom-fusionx" |

</details>

# Model Parameter Documentation

## Image to Video

### Cogvideox 5b Image to Video

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image.<br>**Note:** this model only supports 720 x 480 resolution. Unlike other model implementations, we do not autofix the video to be in the resolution of the given image. |
| model | string | "cogvideox-5b-image-to-video" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 720 | The width in pixels of the generated video. |
| num_frames | int | 48 | Number of frames to generate. |
| num_inference_steps | int | 50 | The number of denoising steps. More steps can improve quality but are slower. |
| timesteps | list | | Custom timesteps to use for the denoising process, must be in descending order. |
| guidance_scale | float | 7.0 | Classifier-Free Diffusion guidance scale. Higher values align the video more closely with the prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| negative_prompt_embeds | torch.FloatTensor | | Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument. |
| output_type | str | pil | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a StableDiffusionXLPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 226 | Maximum sequence length in the encoded prompt. |

</details>

### Framepack I2V HY

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image. |
| model | string | "framepack-i2v-hy" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| prompt_2 | string | "" | A secondary prompt for the second text encoder; defaults to the main prompt if not provided. |
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| negative_prompt2 | string | "" | A secondary negative prompt for the second text encoder. |
| height | int | 720 | The height in pixels of the generated video. |
| width | int | 1280 | The width in pixels of the generated video. |
| num_frames | int | 129 | Number of frames to generate. |
| num_inference_steps | int | 50 | The number of denoising steps. More steps can improve quality but are slower. |
| sigmas | list | | Custom sigmas for the denoising scheduler. |
| true_cfg_scale | float | 1.0 | Enables true classifier-free guidance when > 1.0. |
| guidance_scale | float | 6.0 | Guidance scale to control how closely the video adheres to the prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| image_latents | torch.Tensor | | Pre-encoded image latents, bypassing the VAE for the first image. |
| last_image_latents | torch.Tensor | | Pre-encoded image latents, bypassing the VAE for the last image. |
| prompt_embeds | torch.Tensor | | Pre-generated text embeddings, an alternative to 'prompt'. |
| pooled_prompt_embeds | torch.FloatTensor | | Pre-generated pooled text embeddings. |
| negative_prompt_embeds | torch.FloatTensor | | Pre-generated negative text embeddings, an alternative to 'negative_prompt'. |
| output_type | str | pil | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a HunyuanVideoFramepackPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| clip_skip | int | | Number of final layers to skip from the CLIP model. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |

</details>

### LTX Video

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image. |
| model | string | "ltx-video" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt to avoid during video generation. |
| height | int | 512 | The height in pixels of the generated video. |
| width | int | 704 | The width in pixels of the generated video. |
| num_frames | int | 161 | Number of frames to generate. |
| num_inference_steps | int | 50 | The number of denoising steps. More steps can improve quality but are slower. |
| timesteps | list | | Custom timesteps for the denoising process in descending order. |
| guidance_scale | float | 3.0 | Scale for classifier-free guidance. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator to make generation deterministic. |
| latents | torch.Tensor | | Pre-generated noisy latents. |
| prompt_embeds | torch.Tensor | | Pre-generated text embeddings, an alternative to 'prompt'. |
| promt_attension_mask | torch.Tensor | | Pre-generated attention mask for text embeddings. |
| negative_prompt_embeds | torch.FloatTensor | | Pre-generated negative text embeddings. |
| negative_prompt_attension_mask | torch.FloatTensor | | Pre-generated attention mask for negative text embeddings. |
| decode_timestep | float | 0.0 | The timestep at which the generated video is decoded. |
| decode_noise_scale | float | None | Interpolation factor between random noise and denoised latents at decode time. |
| output_type | str | pil | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a LTXPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 128 | Maximum sequence length for the prompt. |

</details>

### Luma Ray 2

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | URL to input image. **Note that Luma does not accept local files.** |
| model | string | "luma-ray-2" |

</details>

### Pusa V1

Because PusaV1 is relatively new, there is no central location for its required and optional arguments. Based on its examples, we have added some parameters to the optional arguments section.

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image. |
| model | string | "pusa-v1" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| cond_position | str | "0" | Comma-separated list of frame indices for conditioning. You can use any position from 0 to 20. |
| noise_multipliers | str | "0.0" | Comma-separated noise multipliers for conditioning frames. A value of 0 means the condition image is used as totally clean, higher value means adding more noise.<br><br>For I2V, you can use 0.2 or any from 0 to 1.<br><br>For Start-End-Frame, you can use 0.2,0.4, or any from 0 to 1. |
| lora_alpha | float | 1.0 | A bigger alpha would bring more temporal consistency (i.e., make generated frames more like the conditioning part), but may also cause small motion or even collapse. We recommend using a value around 1 to 2. |
| num_inference_steps | int | 30 | The number of denoising steps. More steps can improve quality but are slower. |
| num_frames | int | 81 | |

</details>

### Veo 2.0 Generate 001

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image. |
| model | string | "veo-2.0-generate-002" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negativePrompt | string | "" | Text string that describes anything you want to discourage the model from generating |
| aspectRatio | str | "16:9" | Defines the aspect ratio of the generated videos. Accepts '16:9' (landscape) or '9:16' (portrait). |
| personGeneration | str | "allow_adult" | Controls whether people or face generation is allowed. Accepts 'allow_adult' or 'disallow'. |
| numberOfVideos | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| durationSeconds | int | 8 | Veo 2 only. Length of each output video in seconds, between 5 and 8 |

</details>

### Wan 2.1 I2V 14b 720p

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image. |
| model | string | "wan2.1-i2v-14b-720p" |

</details>

<details>
<summary>Optional Arguments</summary>

Wan vace supports flow shift, which is a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability. The default for Pose Guidance is 3.0. Flow shift is calculated and loaded in the load_models stage. If you want to adjust flow shift, you must change the value in the load_models method, stop the app on Modal, and re-load the model.

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 832 | The width in pixels of the generated video. |
| conditioning_scale | float | 1.0 | The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor. |
| num_frames | int | 81 | Number of frames in the generated video |
| num_inference_steps | int | 50 | The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference. |
| guidance_scale | float | 5.0 | Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| negative_prompt_embeds | torch.Tensor | | Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument. |
| image_embeds | torch.Tensor | | Pre-generated image embeddings, used as an alternative to the 'image' argument. |
| output_type | str | np | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a WanPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 512 | Maximum sequence length in the encoded prompt. |

</details>

### Wan 2.1 Vace 14b

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image. |
| model | string | "wan2.1-vace-14b" |

</details>

<details>
<summary>Optional Arguments</summary>

Wan vace supports flow shift, which is a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability. The default for Pose Guidance is 3.0. Flow shift is calculated and loaded in the load_models stage. If you want to adjust flow shift, you must change the value in the load_models method, stop the app on Modal, and re-load the model.

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| video | list | | The input video (List[PIL.Image.Image]) to be used as a starting point for the generation.<br><br>**Note:** this is created in _process_payload for you. |
| mask | list | | The input mask (List[PIL.Image.Image]) that defines which video regions to condition on (black) and which to generate (white).<br><br>**Note:** this is created in process_payload for you. |
| reference_images | list | | A list of one or more reference images (List[PIL.Image.Image]) as extra conditioning for the generation. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 832 | The width in pixels of the generated video. |
| conditioning_scale | float | 1.0 | The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor. |
| num_frames | int | 81 | Number of frames in the generated video |
| num_inference_steps | int | 50 | The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference. |
| guidance_scale | float | 5.0 | Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| output_type | str | np | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a WanPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 512 | Maximum sequence length in the encoded prompt. |

</details>

### Wan 2.1 Vace 14b I2V FusionX

This is a LoRA applied on top of Wan 2.1 Vace 14b I2V. All of the required arguments and optional arguments are the same as Wan 2.1 Vace 14b I2V, except for the model string.

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Local path or URL to input image. |
| model | string | "wan2.1-vace-14b-i2v-fusionx" |

</details>

## Interpolate

### Wan 2.1 Flf2v 14B 720p

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| first_frame | string | Local path or URL to first frame image. |
| last_frame | string | Local path or URL to last frame image |
| model | string | "wan2.1-flf2v-14b-720p" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 832 | The width in pixels of the generated video. |
| num_frames | int | 81 | Number of frames in the generated video |
| num_inference_steps | int | 50 | The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference. |
| guidance_scale | float | 5.0 | Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| negative_prompt_embeds | torch.Tensor | | Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument. |
| image_embeds | torch.Tensor | | Pre-generated image embeddings, used as an alternative to the 'image' argument. |
| output_type | str | np | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a WanPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 512 | Maximum sequence length in the encoded prompt. |

</details>

### Wan 2.1 Vace 14b

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| first_frame | string | Local path or URL to first frame image. |
| last_frame | string | Local path or URL to last frame image |
| model | string | "wan2.1-vace-14b" |

</details>

<details>
<summary>Optional Arguments</summary>

Wan vace supports flow shift, which is a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability. The default for Pose Guidance is 3.0. Flow shift is calculated and loaded in the load_models stage. If you want to adjust flow shift, you must change the value in the load_models method, stop the app on Modal, and re-load the model.

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| video | list | | The input video (List[PIL.Image.Image]) to be used as a starting point for the generation.<br><br>**Note:** this is created in _process_payload for you. |
| mask | list | | The input mask (List[PIL.Image.Image]) that defines which video regions to condition on (black) and which to generate (white).<br><br>**Note:** this is created in process_payload for you. |
| reference_images | list | | A list of one or more reference images (List[PIL.Image.Image]) as extra conditioning for the generation. |
| conditioning_scale | float | 1.0 | The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 832 | The width in pixels of the generated video. |
| num_frames | int | 81 | Number of frames in the generated video |
| num_inference_steps | int | 50 | The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference. |
| guidance_scale | float | 5.0 | Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| output_type | str | np | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a WanPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 512 | Maximum sequence length in the encoded prompt. |

</details>

### Framepack I2V HY

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| first_frame | string | Local path or URL to first frame image. |
| last_frame | string | Local path or URL to last frame image |
| model | string | "framepack-i2v-hy" |

</details>

<details>
<summary>Optional Arguments</summary>

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| prompt_2 | string | "" | A secondary prompt for the second text encoder; defaults to the main prompt if not provided. |
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| negative_prompt2 | string | "" | A secondary negative prompt for the second text encoder. |
| height | int | 720 | The height in pixels of the generated video. |
| width | int | 1280 | The width in pixels of the generated video. |
| num_frames | int | 129 | Number of frames to generate. |
| num_inference_steps | int | 50 | The number of denoising steps. More steps can improve quality but are slower. |
| sigmas | list | | Custom sigmas for the denoising scheduler. |
| true_cfg_scale | float | 1.0 | Enables true classifier-free guidance when > 1.0. |
| guidance_scale | float | 6.0 | Guidance scale to control how closely the video adheres to the prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| image_latents | torch.Tensor | | Pre-encoded image latents, bypassing the VAE for the first image. |
| last_image_latents | torch.Tensor | | Pre-encoded image latents, bypassing the VAE for the last image. |
| prompt_embeds | torch.Tensor | | Pre-generated text embeddings, an alternative to 'prompt'. |
| pooled_prompt_embeds | torch.FloatTensor | | Pre-generated pooled text embeddings. |
| negative_prompt_embeds | torch.FloatTensor | | Pre-generated negative text embeddings, an alternative to 'negative_prompt'. |
| output_type | str | pil | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a HunyuanVideoFramepackPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| clip_skip | int | | Number of final layers to skip from the CLIP model. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |

</details>

### Luma Ray 2

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| first_frame | string | URL to first frame image. |
| last_frame | string | URL to last frame image |
| model | string | "luma-ray-2" |

</details>

## Pose Guidance

### Wan 2.1 Vace 14b

<details>
<summary>Required Arguments</summary>

| Name | Type | Description |
|------|------|-------------|
| prompt | string | Text prompt to guide generation |
| image | string | Path or URL to input image |
| model | string | "wan2.1-vace-14b" |

</details>

<details>
<summary>Optional Arguments</summary>

Wan vace supports flow shift, which is a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability. The default for Pose Guidance is 3.0. Flow shift is calculated and loaded in the load_models stage. If you want to adjust flow shift, you must change the value in the load_models method, stop the app on Modal, and re-load the model.

| Name | Type | Default Value | Description |
|------|------|---------------|-------------|
| guiding_video | string | | A video to guide the pose of the output video. If provided, a pose_video will be generated for the output video (List[PIL.Image.Image]) |
| pose_video | string | | A pose skeleton video to guide the pose of the output video (List[PIL.Image.Image]) |
| negative_prompt | string | "" | The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1. |
| video | list | | The input video (List[PIL.Image.Image]) to be used as a starting point for the generation.<br><br>**Note:** this is created in _process_payload for you. |
| mask | list | | The input mask (List[PIL.Image.Image]) that defines which video regions to condition on (black) and which to generate (white).<br><br>**Note:** this is created in process_payload for you. |
| reference_images | list | | A list of one or more reference images (List[PIL.Image.Image]) as extra conditioning for the generation. |
| conditioning_scale | float | 1.0 | The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor. |
| height | int | 480 | The height in pixels of the generated video. |
| width | int | 832 | The width in pixels of the generated video. |
| num_frames | int | 81 | Number of frames in the generated video |
| num_inference_steps | int | 50 | The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference. |
| guidance_scale | float | 5.0 | Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt. |
| num_videos_per_prompt | int | 1 | The number of videos to generate for each prompt.<br><br>**Note:** Tio Magic Animation Framework currently only supports 1 video output |
| generator | torch.Generator | | A torch.Generator or List[torch.Generator] to make generation deterministic. |
| latents | torch.FloatTensor | | Pre-generated noisy latents to be used as inputs for generation. |
| prompt_embeds | torch.FloatTensor | | Pre-generated text embeddings, used as an alternative to the 'prompt' argument. |
| output_type | str | np | The output format of the generated video. Choose between 'pil' or 'np.array'. |
| return_dict | bool | True | Whether to return a WanPipelineOutput object instead of a plain tuple. |
| attention_kwargs | dict | | A kwargs dictionary passed to the AttentionProcessor. |
| callback_on_step_end | Callable | | A function called at the end of each denoising step during inference. |
| callback_on_step_end_tensor_inputs | list | | The list of tensor inputs for the callback_on_step_end function. |
| max_sequence_length | int | 512 | Maximum sequence length in the encoded prompt. |

</details>
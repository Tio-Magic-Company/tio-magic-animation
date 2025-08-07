---
layout: default
title: Parameter Documentation
permalink: /parameter-documentation
---

# Model Parameter Documentation

## Text to Video

### Cogvideox 5b

<details>
    <summary>Required Arguments</summary>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Type</th>
            <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <td>prompt</td>
            <td>string</td>
            <td>Text prompt to guide generation</td>
            </tr>
            <tr>
            <td>model</td>
            <td>string</td>
            <td>"cogvideox-5b"</td>
            </tr>
        </tbody>
    </table>
</details>
<details>
    <summary>Optional Arguments</summary>
    <table>
        <thead>
            <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Default Value</th>
            <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <td>negative_prompt</td>
            <td>string</td>
            <td>""</td>
            <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
            </tr>
            <tr>
            <td>height</td>
            <td>int</td>
            <td>480</td>
            <td>The height in pixels of the generated video.</td>
            </tr>
            <tr>
            <td>width</td>
            <td>int</td>
            <td>720</td>
            <td>The width in pixels of the generated video.</td>
            </tr>
            <tr>
            <td>num_frames</td>
            <td>int</td>
            <td>48</td>
            <td>Number of frames to generate.</td>
            </tr>
            <tr>
            <td>num_inference_steps</td>
            <td>int</td>
            <td>50</td>
            <td>The number of denoising steps. More steps can improve quality but are slower.</td>
            </tr>
            <tr>
            <td>timesteps</td>
            <td>list</td>
            <td></td>
            <td>Custom timesteps to use for the denoising process, must be in descending order.</td>
            </tr>
            <tr>
            <td>guidance_scale</td>
            <td>float</td>
            <td>7.0</td>
            <td>Classifier-Free Diffusion guidance scale. Higher values align the video more closely with the prompt.</td>
            </tr>
            <tr>
            <td>num_videos_per_prompt</td>
            <td>int</td>
            <td>1</td>
            <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
            </tr>
            <tr>
            <td>generator</td>
            <td>torch.Generator</td>
            <td></td>
            <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
            </tr>
            <tr>
            <td>latents</td>
            <td>torch.FloatTensor</td>
            <td></td>
            <td>Pre-generated noisy latents to be used as inputs for generation.</td>
            </tr>
            <tr>
            <td>prompt_embeds</td>
            <td>torch.FloatTensor</td>
            <td></td>
            <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
            </tr>
            <tr>
            <td>negative_prompt_embeds</td>
            <td>torch.FloatTensor</td>
            <td></td>
            <td>Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument.</td>
            </tr>
            <tr>
            <td>output_type</td>
            <td>str</td>
            <td>pil</td>
            <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
            </tr>
            <tr>
            <td>return_dict</td>
            <td>bool</td>
            <td>True</td>
            <td>Whether to return a StableDiffusionXLPipelineOutput object instead of a plain tuple.</td>
            </tr>
            <tr>
            <td>attention_kwargs</td>
            <td>dict</td>
            <td></td>
            <td>A kwargs dictionary passed to the AttentionProcessor.</td>
            </tr>
            <tr>
            <td>callback_on_step_end</td>
            <td>Callable</td>
            <td></td>
            <td>A function called at the end of each denoising step during inference.</td>
            </tr>
            <tr>
            <td>callback_on_step_end_tensor_inputs</td>
            <td>list</td>
            <td></td>
            <td>The list of tensor inputs for the callback_on_step_end function.</td>
            </tr>
            <tr>
            <td>max_sequence_length</td>
            <td>int</td>
            <td>226</td>
            <td>Maximum sequence length in the encoded prompt.</td>
            </tr>
        </tbody>
    </table>
</details>
### Pusa V1

Because PusaV1 is relatively new, there is no central location for its required and optional arguments. Based on its examples, we have added some parameters to the optional arguments section.

<details>
    <summary>Required Arguments</summary>
    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"pusa-v1"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>
    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation.</td>
    </tr>
    </tbody>
    </table>
</details>

### Wan 2.1 Text to Video 14b

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-t2v-14b"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>832</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td>Number of frames in the generated video</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>5.0</td>
    <td>Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>np</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a WanPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>512</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>
### Wan 2.1 Vace 14b

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-vace-14b"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>832</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>conditioning_scale</td>
    <td>float</td>
    <td>1.0</td>
    <td>The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td>Number of frames in the generated video</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>5.0</td>
    <td>Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>np</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a WanPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>512</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>
### Wan 2.1 Vace 14b T2V FusionX

This is a LoRA applied on top of Wan 2.1 Vace 14b. All of the required arguments and optional arguments are the same as Wan 2.1 Vace 14b, except for the model string.

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan-14b-vace-t2v-fusionx"</td>
    </tr>
    </tbody>
    </table>
</details>

### Wan 2.1 Vace 14b Phantom FusionX

This is a LoRA applied on top of Wan 2.1 Vace 14b. All of the required arguments and optional arguments are the same as Wan 2.1 Vace 14b, except for the model string.

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-vace-14b-phantom-fusionx"</td>
    </tr>
    </tbody>
    </table>
</details>
# Model Parameter Documentation

## Image to Video

### Cogvideox 5b Image to Video

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.<br><strong>Note:</strong> this model only supports 720 x 480 resolution. Unlike other model implementations, we do not autofix the video to be in the resolution of the given image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"cogvideox-5b-image-to-video"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>720</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>48</td>
    <td>Number of frames to generate.</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps can improve quality but are slower.</td>
    </tr>
    <tr>
    <td>timesteps</td>
    <td>list</td>
    <td></td>
    <td>Custom timesteps to use for the denoising process, must be in descending order.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>7.0</td>
    <td>Classifier-Free Diffusion guidance scale. Higher values align the video more closely with the prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>negative_prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>pil</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a StableDiffusionXLPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>226</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>

### Framepack I2V HY

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"framepack-i2v-hy"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt_2</td>
    <td>string</td>
    <td>""</td>
    <td>A secondary prompt for the second text encoder; defaults to the main prompt if not provided.</td>
    </tr>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>negative_prompt2</td>
    <td>string</td>
    <td>""</td>
    <td>A secondary negative prompt for the second text encoder.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>720</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>1280</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>129</td>
    <td>Number of frames to generate.</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps can improve quality but are slower.</td>
    </tr>
    <tr>
    <td>sigmas</td>
    <td>list</td>
    <td></td>
    <td>Custom sigmas for the denoising scheduler.</td>
    </tr>
    <tr>
    <td>true_cfg_scale</td>
    <td>float</td>
    <td>1.0</td>
    <td>Enables true classifier-free guidance when > 1.0.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>6.0</td>
    <td>Guidance scale to control how closely the video adheres to the prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>image_latents</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-encoded image latents, bypassing the VAE for the first image.</td>
    </tr>
    <tr>
    <td>last_image_latents</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-encoded image latents, bypassing the VAE for the last image.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated text embeddings, an alternative to 'prompt'.</td>
    </tr>
    <tr>
    <td>pooled_prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated pooled text embeddings.</td>
    </tr>
    <tr>
    <td>negative_prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated negative text embeddings, an alternative to 'negative_prompt'.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>pil</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a HunyuanVideoFramepackPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>clip_skip</td>
    <td>int</td>
    <td></td>
    <td>Number of final layers to skip from the CLIP model.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    </tbody>
    </table>
</details>

### LTX Video

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"ltx-video"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt to avoid during video generation.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>512</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>704</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>161</td>
    <td>Number of frames to generate.</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps can improve quality but are slower.</td>
    </tr>
    <tr>
    <td>timesteps</td>
    <td>list</td>
    <td></td>
    <td>Custom timesteps for the denoising process in descending order.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>3.0</td>
    <td>Scale for classifier-free guidance.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated noisy latents.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated text embeddings, an alternative to 'prompt'.</td>
    </tr>
    <tr>
    <td>promt_attension_mask</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated attention mask for text embeddings.</td>
    </tr>
    <tr>
    <td>negative_prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated negative text embeddings.</td>
    </tr>
    <tr>
    <td>negative_prompt_attension_mask</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated attention mask for negative text embeddings.</td>
    </tr>
    <tr>
    <td>decode_timestep</td>
    <td>float</td>
    <td>0.0</td>
    <td>The timestep at which the generated video is decoded.</td>
    </tr>
    <tr>
    <td>decode_noise_scale</td>
    <td>float</td>
    <td>None</td>
    <td>Interpolation factor between random noise and denoised latents at decode time.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>pil</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a LTXPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>128</td>
    <td>Maximum sequence length for the prompt.</td>
    </tr>
    </tbody>
    </table>
</details>
### Luma Ray 2

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>URL to input image. <strong>Note that Luma does not accept local files.</strong></td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"luma-ray-2"</td>
    </tr>
    </tbody>
    </table>
</details>

### Pusa V1

Because PusaV1 is relatively new, there is no central location for its required and optional arguments. Based on its examples, we have added some parameters to the optional arguments section.

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"pusa-v1"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>cond_position</td>
    <td>str</td>
    <td>"0"</td>
    <td>Comma-separated list of frame indices for conditioning. You can use any position from 0 to 20.</td>
    </tr>
    <tr>
    <td>noise_multipliers</td>
    <td>str</td>
    <td>"0.0"</td>
    <td>Comma-separated noise multipliers for conditioning frames. A value of 0 means the condition image is used as totally clean, higher value means adding more noise.<br><br>For I2V, you can use 0.2 or any from 0 to 1.<br><br>For Start-End-Frame, you can use 0.2,0.4, or any from 0 to 1.</td>
    </tr>
    <tr>
    <td>lora_alpha</td>
    <td>float</td>
    <td>1.0</td>
    <td>A bigger alpha would bring more temporal consistency (i.e., make generated frames more like the conditioning part), but may also cause small motion or even collapse. We recommend using a value around 1 to 2.</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>30</td>
    <td>The number of denoising steps. More steps can improve quality but are slower.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td></td>
    </tr>
    </tbody>
    </table>
</details>

### Veo 2.0 Generate 001
<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"veo-2.0-generate-002"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negativePrompt</td>
    <td>string</td>
    <td>""</td>
    <td>Text string that describes anything you want to discourage the model from generating</td>
    </tr>
    <tr>
    <td>aspectRatio</td>
    <td>str</td>
    <td>"16:9"</td>
    <td>Defines the aspect ratio of the generated videos. Accepts '16:9' (landscape) or '9:16' (portrait).</td>
    </tr>
    <tr>
    <td>personGeneration</td>
    <td>str</td>
    <td>"allow_adult"</td>
    <td>Controls whether people or face generation is allowed. Accepts 'allow_adult' or 'disallow'.</td>
    </tr>
    <tr>
    <td>numberOfVideos</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>durationSeconds</td>
    <td>int</td>
    <td>8</td>
    <td>Veo 2 only. Length of each output video in seconds, between 5 and 8</td>
    </tr>
    </tbody>
    </table>
</details>

### Wan 2.1 I2V 14b 720p
<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-i2v-14b-720p"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    Wan vace supports flow shift, which is a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability. The default for Pose Guidance is 3.0. Flow shift is calculated and loaded in the load_models stage. If you want to adjust flow shift, you must change the value in the load_models method, stop the app on Modal, and re-load the model.

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>832</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>conditioning_scale</td>
    <td>float</td>
    <td>1.0</td>
    <td>The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td>Number of frames in the generated video</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>5.0</td>
    <td>Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>negative_prompt_embeds</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument.</td>
    </tr>
    <tr>
    <td>image_embeds</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated image embeddings, used as an alternative to the 'image' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>np</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a WanPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>512</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>

### Wan 2.1 Vace 14b

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-vace-14b"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    Wan vace supports flow shift, which is a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability. The default for Pose Guidance is 3.0. Flow shift is calculated and loaded in the load_models stage. If you want to adjust flow shift, you must change the value in the load_models method, stop the app on Modal, and re-load the model.

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>video</td>
    <td>list</td>
    <td></td>
    <td>The input video (List[PIL.Image.Image]) to be used as a starting point for the generation.<br><br><strong>Note:</strong> this is created in _process_payload for you.</td>
    </tr>
    <tr>
    <td>mask</td>
    <td>list</td>
    <td></td>
    <td>The input mask (List[PIL.Image.Image]) that defines which video regions to condition on (black) and which to generate (white).<br><br><strong>Note:</strong> this is created in process_payload for you.</td>
    </tr>
    <tr>
    <td>reference_images</td>
    <td>list</td>
    <td></td>
    <td>A list of one or more reference images (List[PIL.Image.Image]) as extra conditioning for the generation.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>832</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td>Number of frames in the generated video</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>5.0</td>
    <td>Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>np</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a WanPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>512</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>

### Wan 2.1 Vace 14b I2V FusionX

This is a LoRA applied on top of Wan 2.1 Vace 14b I2V. All of the required arguments and optional arguments are the same as Wan 2.1 Vace 14b I2V, except for the model string.

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Local path or URL to input image.</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-vace-14b-i2v-fusionx"</td>
    </tr>
    </tbody>
    </table>
</details>

## Interpolate

### Wan 2.1 Flf2v 14B 720p
<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>first_frame</td>
    <td>string</td>
    <td>Local path or URL to first frame image.</td>
    </tr>
    <tr>
    <td>last_frame</td>
    <td>string</td>
    <td>Local path or URL to last frame image</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-flf2v-14b-720p"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>832</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td>Number of frames in the generated video</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>5.0</td>
    <td>Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>negative_prompt_embeds</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated negative text embeddings, used as an alternative to the 'negative_prompt' argument.</td>
    </tr>
    <tr>
    <td>image_embeds</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated image embeddings, used as an alternative to the 'image' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>np</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a WanPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>512</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>

### Wan 2.1 Vace 14b

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>first_frame</td>
    <td>string</td>
    <td>Local path or URL to first frame image.</td>
    </tr>
    <tr>
    <td>last_frame</td>
    <td>string</td>
    <td>Local path or URL to last frame image</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-vace-14b"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>video</td>
    <td>list</td>
    <td></td>
    <td>The input video (List[PIL.Image.Image]) to be used as a starting point for the generation.<br><br><strong>Note:</strong> this is created in _process_payload for you.</td>
    </tr>
    <tr>
    <td>mask</td>
    <td>list</td>
    <td></td>
    <td>The input mask (List[PIL.Image.Image]) that defines which video regions to condition on (black) and which to generate (white).<br><br><strong>Note:</strong> this is created in process_payload for you.</td>
    </tr>
    <tr>
    <td>reference_images</td>
    <td>list</td>
    <td></td>
    <td>A list of one or more reference images (List[PIL.Image.Image]) as extra conditioning for the generation.</td>
    </tr>
    <tr>
    <td>conditioning_scale</td>
    <td>float</td>
    <td>1.0</td>
    <td>The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>832</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td>Number of frames in the generated video</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>5.0</td>
    <td>Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>np</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a WanPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>512</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>

### Framepack I2V HY

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>first_frame</td>
    <td>string</td>
    <td>Local path or URL to first frame image.</td>
    </tr>
    <tr>
    <td>last_frame</td>
    <td>string</td>
    <td>Local path or URL to last frame image</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"framepack-i2v-hy"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt_2</td>
    <td>string</td>
    <td>""</td>
    <td>A secondary prompt for the second text encoder; defaults to the main prompt if not provided.</td>
    </tr>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>negative_prompt2</td>
    <td>string</td>
    <td>""</td>
    <td>A secondary negative prompt for the second text encoder.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>720</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>1280</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>129</td>
    <td>Number of frames to generate.</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps can improve quality but are slower.</td>
    </tr>
    <tr>
    <td>sigmas</td>
    <td>list</td>
    <td></td>
    <td>Custom sigmas for the denoising scheduler.</td>
    </tr>
    <tr>
    <td>true_cfg_scale</td>
    <td>float</td>
    <td>1.0</td>
    <td>Enables true classifier-free guidance when > 1.0.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>6.0</td>
    <td>Guidance scale to control how closely the video adheres to the prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>image_latents</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-encoded image latents, bypassing the VAE for the first image.</td>
    </tr>
    <tr>
    <td>last_image_latents</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-encoded image latents, bypassing the VAE for the last image.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.Tensor</td>
    <td></td>
    <td>Pre-generated text embeddings, an alternative to 'prompt'.</td>
    </tr>
    <tr>
    <td>pooled_prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated pooled text embeddings.</td>
    </tr>
    <tr>
    <td>negative_prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated negative text embeddings, an alternative to 'negative_prompt'.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>pil</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a HunyuanVideoFramepackPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>clip_skip</td>
    <td>int</td>
    <td></td>
    <td>Number of final layers to skip from the CLIP model.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    </tbody>
    </table>
</details>

### Luma Ray 2

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>first_frame</td>
    <td>string</td>
    <td>URL to first frame image.</td>
    </tr>
    <tr>
    <td>last_frame</td>
    <td>string</td>
    <td>URL to last frame image</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"luma-ray-2"</td>
    </tr>
    </tbody>
    </table>
</details>
## Pose Guidance

### Wan 2.1 Vace 14b

<details>
    <summary>Required Arguments</summary>

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>prompt</td>
    <td>string</td>
    <td>Text prompt to guide generation</td>
    </tr>
    <tr>
    <td>image</td>
    <td>string</td>
    <td>Path or URL to input image</td>
    </tr>
    <tr>
    <td>model</td>
    <td>string</td>
    <td>"wan2.1-vace-14b"</td>
    </tr>
    </tbody>
    </table>
</details>

<details>
    <summary>Optional Arguments</summary>

    Wan vace supports flow shift, which is a value that estimates motion between two frames. A larger flow shift focuses on high motion or transformation. A smaller flow shift focuses on stability. The default for Pose Guidance is 3.0. Flow shift is calculated and loaded in the load_models stage. If you want to adjust flow shift, you must change the value in the load_models method, stop the app on Modal, and re-load the model.

    <table>
    <thead>
    <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>guiding_video</td>
    <td>string</td>
    <td></td>
    <td>A video to guide the pose of the output video. If provided, a pose_video will be generated for the output video (List[PIL.Image.Image])</td>
    </tr>
    <tr>
    <td>pose_video</td>
    <td>string</td>
    <td></td>
    <td>A pose skeleton video to guide the pose of the output video (List[PIL.Image.Image])</td>
    </tr>
    <tr>
    <td>negative_prompt</td>
    <td>string</td>
    <td>""</td>
    <td>The prompt or prompts not to guide video generation. Ignored if guidance_scale is less than 1.</td>
    </tr>
    <tr>
    <td>video</td>
    <td>list</td>
    <td></td>
    <td>The input video (List[PIL.Image.Image]) to be used as a starting point for the generation.<br><br><strong>Note:</strong> this is created in _process_payload for you.</td>
    </tr>
    <tr>
    <td>mask</td>
    <td>list</td>
    <td></td>
    <td>The input mask (List[PIL.Image.Image]) that defines which video regions to condition on (black) and which to generate (white).<br><br><strong>Note:</strong> this is created in process_payload for you.</td>
    </tr>
    <tr>
    <td>reference_images</td>
    <td>list</td>
    <td></td>
    <td>A list of one or more reference images (List[PIL.Image.Image]) as extra conditioning for the generation.</td>
    </tr>
    <tr>
    <td>conditioning_scale</td>
    <td>float</td>
    <td>1.0</td>
    <td>The scale applied to the control conditioning latent stream. Can be a float, List[float], or torch.Tensor.</td>
    </tr>
    <tr>
    <td>height</td>
    <td>int</td>
    <td>480</td>
    <td>The height in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>width</td>
    <td>int</td>
    <td>832</td>
    <td>The width in pixels of the generated video.</td>
    </tr>
    <tr>
    <td>num_frames</td>
    <td>int</td>
    <td>81</td>
    <td>Number of frames in the generated video</td>
    </tr>
    <tr>
    <td>num_inference_steps</td>
    <td>int</td>
    <td>50</td>
    <td>The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.</td>
    </tr>
    <tr>
    <td>guidance_scale</td>
    <td>float</td>
    <td>5.0</td>
    <td>Guidance scale for classifier-free diffusion. Higher values encourage generation to be closely linked to the text prompt.</td>
    </tr>
    <tr>
    <td>num_videos_per_prompt</td>
    <td>int</td>
    <td>1</td>
    <td>The number of videos to generate for each prompt.<br><br><strong>Note:</strong> Tio Magic Animation Framework currently only supports 1 video output</td>
    </tr>
    <tr>
    <td>generator</td>
    <td>torch.Generator</td>
    <td></td>
    <td>A torch.Generator or List[torch.Generator] to make generation deterministic.</td>
    </tr>
    <tr>
    <td>latents</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated noisy latents to be used as inputs for generation.</td>
    </tr>
    <tr>
    <td>prompt_embeds</td>
    <td>torch.FloatTensor</td>
    <td></td>
    <td>Pre-generated text embeddings, used as an alternative to the 'prompt' argument.</td>
    </tr>
    <tr>
    <td>output_type</td>
    <td>str</td>
    <td>np</td>
    <td>The output format of the generated video. Choose between 'pil' or 'np.array'.</td>
    </tr>
    <tr>
    <td>return_dict</td>
    <td>bool</td>
    <td>True</td>
    <td>Whether to return a WanPipelineOutput object instead of a plain tuple.</td>
    </tr>
    <tr>
    <td>attention_kwargs</td>
    <td>dict</td>
    <td></td>
    <td>A kwargs dictionary passed to the AttentionProcessor.</td>
    </tr>
    <tr>
    <td>callback_on_step_end</td>
    <td>Callable</td>
    <td></td>
    <td>A function called at the end of each denoising step during inference.</td>
    </tr>
    <tr>
    <td>callback_on_step_end_tensor_inputs</td>
    <td>list</td>
    <td></td>
    <td>The list of tensor inputs for the callback_on_step_end function.</td>
    </tr>
    <tr>
    <td>max_sequence_length</td>
    <td>int</td>
    <td>512</td>
    <td>Maximum sequence length in the encoded prompt.</td>
    </tr>
    </tbody>
    </table>
</details>
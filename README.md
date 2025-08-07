
# Tio Magic Animation Toolkit

[![Website](https://img.shields.io/badge/Website-Tio_Magic-181717?logo=google-chrome)](https://tiomagic.com/)
[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-green?logo=github)](https://tio-magic-company.github.io/tio-magic-animation/)

Tio Magic Animation Toolkit is designed to simplify the use of video AI models for animation. The Animation Toolkit empowers animators, developers, and AI enthusiasts to easily generate animated videos without the pain of complex technical setup, local hardware limitations, and haphazard documentation.

This toolkit leverages <a href="https://modal.com/" target="_blank">Modal</a> for cloud computing and runs various open/closed-source video generation models.

## Supported Features

## Image to Video
*Prompt: Woman smiling at the camera, waving her right hand as if she was saying hi and greeting someone.*

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/i2v/disney2_wan_vace.gif" alt="I2V Wan 2.1 Vace 14b"><br>
      <b>Wan 2.1 Vace 14b</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/i2v/disney2_i2v_framepack.gif" alt="I2V Framepack I2V HY"><br>
      <b>Framepack I2V HY</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/i2v/disney2_i2v_fusionx.gif" alt="Wan 2.1 I2V FusionX (LoRA)"><br>
      <b>Wan 2.1 I2V FusionX (LoRA)</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/i2v/disney2_i2v_ltx.gif" alt="I2V LTX Video"><br>
      <b>LTX Video</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/i2v/disney2_i2v_pusa.gif" alt="I2V Pusa V1"><br>
      <b>Pusa V1</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/i2v/disney2_i2v_veo.gif" alt="I2V Veo 2"><br>
      <b>Veo 2</b>
    </td>
  </tr>
</table>

## Interpolate
*Prompt: An anime-style young man in a blue t-shirt starts in a standing position. He lifts his right hand and waves to the camera.*

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/interpolate/interpolate_framepack.gif" alt="Interpolate Framepack I2V HY"><br>
      <b>Framepack I2V HY</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/interpolate/interpolate_wan_flfv2.gif" alt="Interpolate Wan FLFV 14b"><br>
      <b>Wan FLFV 14b</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/interpolate/interpolate_wan_vace.gif" alt="Interpolate Wan 2.1 Vace 14b"><br>
      <b>Wan 2.1 Vace 14b</b>
    </td>
  </tr>
</table>

## Pose Guidance
*Prompt: Anime-style cartoon animation of a man waving, empty white background. Skin tone, shading, lighting should be the same as the reference image.*

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/pose_guidance/pg-sample.png"><br>
      <b>Starting Image</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/pose_guidance/driving-wave.gif"><br>
      <b>Pose Video</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/pose_guidance/pose_guidance.gif" alt="Pose Guidance Wan 2.1 Vace 14b"><br>
      <b>Wan 2.1 Vace 14b</b>
    </td>
  </tr>
</table>

## Text to Video
*A playful cartoon-style penguin with a round belly and flappy wings waddles up to a pair of green sunglasses lying on the ground. The penguin leans forward, carefully picks up the sunglasses with its flipper, and smoothly lifts them up to its face. It tilts its head with a confident smile as the green sunglasses rest perfectly on its beak. The animation is smooth and expressive, with exaggerated, bouncy cartoon motion.*

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/t2v/penguin_t2v_phantomfusionx.gif" alt="T2V Wan 2.1 PhantomX (LoRA)"><br>
      <b>Wan 2.1 PhantomX (LoRA)</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/t2v/penguin_t2v_pusav1.gif" alt="T2V Pusa V1"><br>
      <b>Pusa V1</b>
    </td>
    <td align="center" width="33%">
      <img src="https://storage.googleapis.com/tm-animation-public-examples/t2v/penguin_t2v_want2v.gif" alt="T2V Wan 2.1 14b"><br>
      <b>Wan 2.1 14b</b>
    </td>
  </tr>
</table>

## Usage
Go to <a href="https://tio-magic-company.github.io/tio-magic-animation/getting-started" target="_blank">Tio Magic Animation Toolkit Docs</a> for detailed information on usage.
#### Environment Preparation
First, create a virtual environment and activate it.
```
python3 -m venv venv
source venv/bin/activate # on MacOS/Linux
venv/Scripts/activate # on Windows Command Prompt
```
Then, install Python packages:
```
pip install -e .
```
Then, create a .env file. Depending on what provider(s) you are using, copy/pase appropriate access keys to the .env file. For starters, we recommend registering for a Modal account and create an access token:
```
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```
Create a Hugging Face account and add the token to your Modal account (this is needed to access open source models)


## How to Run Modal Demo 
```
python quick_start.py
```

## Supported Providers and Models

## Modal

### Image to Video
- [Cogvideox 5b I2V](https://huggingface.co/zai-org/CogVideoX-5b-I2V)
- [Framepack I2V HY](https://github.com/lllyasviel/FramePack)
- [LTX video](https://huggingface.co/Lightricks/LTX-Video)
- [Pusa V1](https://huggingface.co/RaphaelLiu/PusaV1)
- [Wan 2.1 I2V 14b 720p](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)
- [Wan 2.1 Vace 14b](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)
- [Wan 2.1 I2V FusionX (LoRA)](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX)

### Interpolate
- [Framepack I2V HY](https://github.com/lllyasviel/FramePack)
- [Wan 2.1 FLFV 14b](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P)
- [Wan 2.1 Vace 14b](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)

### Pose Guidance
- [Wan 2.1 Vace 14b](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)

### Text to Video
- [Cogvideox 5b](https://huggingface.co/zai-org/CogVideoX-5b)
- [Pusa V1](https://huggingface.co/RaphaelLiu/PusaV1)
- [Wan 2.1 T2V FantomX (LoRA)](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX)
- [Wan 2.1 14b](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)
- [Wan 2.1 Vace 14b](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)
- [Wan 2.1 PhantomX (LoRA)](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX)

## Local

### Image to Video
- [Luma AI Ray 2](https://lumalabs.ai/ray)
- [Google Deepmind Veo 2](https://deepmind.google/models/veo/)

### Interpolate
- [Luma AI Ray 2](https://lumalabs.ai/ray)

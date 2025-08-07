---
layout: default
title: Intro
permalink: /
---

# Tio Magic Animation Toolkit

Tio Magic Animation Toolkit is designed to simplify the use of video AI models for animation. The Animation Toolkit empowers animators, developers, and AI enthusiasts to easily generate animated videos without the pain of complex technical setup, local hardware limitations, and haphazard documentation.

## Supported Features
- Text to Video
- Image to Video
- Interpolate (in-betweens)
- Pose Guidance

## Supported Providers and Models
## Local
### Image to Video
- [Luma AI Ray 2](https://lumalabs.ai/ray)
- [Google Deepmind Veo 2](https://deepmind.google/models/veo/)

### Interpolate
- [Luma AI Ray 2](https://lumalabs.ai/ray)

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

# Sample Outputs
<style>
.gif-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  padding: 20px;
}

.gif-grid img {
  width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>

<div class="gif-grid">
  <img src="https://storage.googleapis.com/tm-animation-public-examples/t2v/penguin_t2v_phantomfusionx.gif" alt="Video 1">
  <img src="video2.gif" alt="Video 2">
  <img src="video3.gif" alt="Video 3">
  <img src="video4.gif" alt="Video 4">
  <img src="video5.gif" alt="Video 5">
  <img src="video6.gif" alt="Video 6">
  <img src="video7.gif" alt="Video 7">
  <img src="video8.gif" alt="Video 8">
</div>

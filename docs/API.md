---
layout: default
title: API Reference
permalink: /api
---

# API Reference
This is the api REFERENCE FOR THE `tiomagic` Python package, which allows you to run the <a href="https://github.com/Tio-Magic-Company/tio-magic-animation" target="_blank">Tio Magic Animation Toolkit</a> 

After following setup instructions, at the top of your file add `from tiomagic import tm`

## Configuration
`tm.configure(provider=str, gpu=Optional[str], timeout=Optional[str], scaledown_window=Optional[str])`

Establishes which provider you will be using. This is required to run any calls.
- provider: To view a list of existing providers, features, and models, run `tm.list_implementations`

Note that gpu, timeout, and scaledown_window are options for the Modal provider. 
- gpu: To view gpu options, go to the <a href="https://modal.com/docs/reference/modal.gpu" target="_blank">Modal gpu docs</a>.
- timeout: Execution timeout for Modal calls. All are set to a default of 1800 (30 minutes)
- scaledown_window: Keeps Modal container alive for x minutes after the last request. All are set to a default of 900 (15 minutes)

## Features
### text_to_video
`tm.text_to_video(model=str, required_args=Dict[str: Any], **optional_args=Dict[str: Any])`

Runs text to video calls on a model with given arguments.
- model: To view a list of existing models on a provider `tm.get_models(feature="text_to_video", provider="...")`
- required_args: A dictionary with arguments needed to run the specified model. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a>to see the accepted required_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.
    - For text to video, a prompt (str) is required.
- optional_args: A dictionary with arguments that can be added to the generation call. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a> to see the accepted optional_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.

### image_to_video
`tm.image_to_video(model=str, required_args=Dict[str: Any], **optional_args=Dict[str: Any])`

Runs image to video calls on a model with given arguments.
- model: To view a list of existing models on a provider `tm.get_models(feature="image_to_video", provider="...")`
- required_args: A dictionary with arguments needed to run the specified model. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a>to see the accepted required_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.
    - For image to video, a prompt (str) and an image (local or URL) are required.
- optional_args: A dictionary with arguments that can be added to the generation call. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a> to see the accepted optional_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.

### interpolate
`tm.interpolate(model=str, required_args=Dict[str: Any], **optional_args=Dict[str: Any])`

Runs interpolate (in-between) calls on a model with given arguments.
- model: To view a list of existing models on a provider `tm.get_models(feature="interpolate", provider="...")`
- required_args: A dictionary with arguments needed to run the specified model. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a>to see the accepted required_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.
    - For interpolate, a prompt (str), first_frame (local or URL), and last_frame (local or URL) are required.
- optional_args: A dictionary with arguments that can be added to the generation call. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a> to see the accepted optional_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.

### pose_guidance
`tm.pose_guidance(model=str, required_args=Dict[str: Any], **optional_args=Dict[str: Any])`

Runs pose guidance calls on a model with given arguments.
- model: To view a list of existing models on a provider `tm.get_models(feature="pose_guidance", provider="...")`
- required_args: A dictionary with arguments needed to run the specified model. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a>to see the accepted required_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.
    - For pose guidance, a prompt (str), image (local or URL), and either a guiding video (a video to guide the pose of the output video) or a pose video (a pose skeleton video) is required.
- optional_args: A dictionary with arguments that can be added to the generation call. Refer to <a href="https://tio-magic-company.github.io/tio-magic-animation/parameter-docs" target="_blank">Parameter Documentation</a> to see the accepted optional_args of a particular model, or run `tm.get_schema(feature=str, model=str)`.

### check_generation_status
`tm.check_generation_status(job_id=str)`

Checks and updates the current status of a video generation job. You can find the job_id in `generation_log.json`. THis method can be called periodically to monitor long-running generation tasks. If you are running on Modal provider, you can also go onto your Modal dashboard to track generation.

If a generation is completed, calling `check_generation_status` will download the resulting video into `output_videos` directory.

At the moment, this will only run on the Modal provider. Local providers are all synchronous and will download the resulting video once generation is complete

### cancel_job
`tm.cancel_job(job_id=str)`

Attempts to cancel a running job with the provider. The success of cancellation depends on the provider's capabilities and the current state of the job. The result can be found in `generation_log.json`. 

At the moment, this will only run on the Modal provider. Local providers are all synchronous.

### get_providers
`tm.get_providers`

List all providers available.

### get_models
`tm.get_models(feature: str, provider: str)`

List all models available for a provider and feature.

### get_schema
`tm.get_schema(feature: str, model: str)`

List schema for particular implementation.

### list_implementations
`tm.list_implementations()`

List all implementatoins available. Models listed and sorted by provider and feature.

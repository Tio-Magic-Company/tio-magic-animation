---
layout: default
title: Contributing
permalink: /contributing
---

# Contributing New Models/Features

## Contributing New Models/Features on Modal

### How to add a new model in Modal

<img src="https://storage.googleapis.com/tm-animation-public-examples/repo-assets/modal_diagram_highlighted.png">

Above is the flow of a feature call. The class and methods outlined in red represent the interface between your local machine and Modal.

Contributing a new model only requires you to create the file that runs code on Modal and returns the user data.

1. Create a new `.py` file in `src/tiomagic/providers/modal`. This is where your modal code goes.
   - Refer to **Structure of a Modal Provider file** for a breakdown of an implementation
2. Add model to `src/tiomagic/providers/modal/__init__.py`
3. Add requirements schema to proper features file under `src/tiomagic/core/schemas`

And that's it!

### Structure of a Modal Provider file

If you are unfamiliar with deploying code to Modal, familiarize yourself with [Modal documentation](https://modal.com/docs).

You can refer to `src/tiomagic/providers/modal/wan_2_1_14b.py` and its comments for a detailed description of a Modal Provider file. The docs are a high level description of a file:

**Feature class** (i.e. `class I2V`) consists of methods that run on modal:
- `load_models` - once the Modal container starts, method downloads and initializes the models and pipeline
- `generate` - this method executes the actual video generation using the loaded pipeline, saves the generated video to Modal's volume storage, and returns the video as bytes
- `handle_web_inference` - when Modal receives a POST request to generate a video, this method makes the actual call to generate the video and returns to us a `call_id`, which is stored in `generation_log.json`

**Modal Provider class** (i.e. `Wan21TextToVideo14B`) prepares user data to send to Modal. This class serves as the local client that will communicate with Modal:
- `__init__` - initializes the local client and sets references to the Modal app (name of the app, the app instance, the feature it is managing)
- `_prepare_payload` - prepares the data to send to Modal. If the model you are implementing requires a specific format or naming of variables for generation, here is a good place to make those adjustments.
  - Note that `feature_type` is set in the payload. This is used for routing and is removed from the payload before calling to generate.

**Class WebAPI** is the API that receives the POST request from the user and directs the request to the appropriate feature.
- For example, if a model is capable of 3 features, this one API reads what the `feature_type` is being requested and calls the appropriate Modal Provider class to spawn the correct generation

`registry.register` is a **CRUCIAL** function that links the feature/model/provider combination. If you don't register each feature/model/provider combination, you will not be able to access this model and will run into errors.

### How to add a new feature into an existing model

1. Read "Structure of a Modal Provider"
2. Create a new Feature Class
3. Create a new Modal Provider Class
4. Add the new feature into the WebAPI
5. Add your new feature/model/provider combination using `registry.register`

## Contributing New Models/Features on Local

### How to add a new model on local

1. Obtain the API key of the closed-source model you want to contribute
2. If there is a library you must pip install for the model, add it to the dependencies list under `pyproject.toml`
   - When testing locally, run `pip install -e .` again to ensure that the dependencies are updated
3. Add a new file under `src/tiomagic/providers/local`. This is where your implementation goes.
   - Refer to **Structure of a Local Provider file** for a breakdown of an implementation
4. Add model to `src/tiomagic/providers/local/__init__.py`  
5. Add requirements schema to proper features file under `src/tiomagic/core/schemas`

### Structure of a Local Provider file

**Class ModelFeature** (i.e. `LumaRay2I2V`):
- `__init__` - collects API key from your `.env` file and initializes basic information about the implementation
- `_prepare_payload` - prepares the data to send to the model. If the model you are implementing requires a specific format or naming of variables for generation, here is a good place to make those adjustments.
  - Note that `feature_type` is set in the payload. This is used for routing and is removed from the payload before calling to generate.
- `generate` - the method that makes the generation request to the closed source model
  - Polling to wait for the generation to complete occurs here
  - Once the generation completes, saves video to `output_videos` folder

`registry.register` is a **CRUCIAL** function that links the feature/model/provider combination. If you don't register each feature/model/provider combination, you will not be able to access this model and will run into errors.

### How to add a new feature into an existing model

1. Read "Structure of a Local Provider"
2. Create a new ModelFeature Class
3. Add your new feature/model/provider combination using `registry.register`
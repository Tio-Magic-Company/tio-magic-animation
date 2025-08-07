---
layout: default
title: Getting Started
---

# Getting Started - Installation

*(In the future will be `pip install`)*

1. Open your Command Line Interface (CLI) - terminal on Mac or command prompt on Windows
2. Download Python: `sudo apt install python3`
3. Install Pip (Python's package manager): `sudo apt install python3-pip`
4. Clone this repository to a local directory 
5. Change directories to the location in which you cloned your repository using `cd`
6. Create a virtual environment: `python3 -m venv venv`
7. Activate the virtual environment - (you should see `(venv)` on your command line):
   - **On MacOS/Linux**: `source venv/bin/activate`
   - **On Windows Command Prompt**: `venv/Scripts/activate`
8. Run `pip install -e .`
   - **Don't forget the DOT!**
9. Create a `.env` file at the root of your repository. Depending on what provider(s) you are using, copy/paste the appropriate access keys to the `.env` file in the repository:
   - **Veo 2**: `GOOGLE_API_KEY`
   - **Luma AI**: `LUMAAI_API_KEY`
   - **Modal**: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
10. `cd` to `quick_start.py`, configure the file to your needs (see examples below), and run `python3 quick_start.py`
11. When you are done developing, run `deactivate` to exit the virtual environment

# Getting Started - Quick Start

Move to the directory where `quick_start.py` is located and run `python3 quick_start.py`

```python
from tiomagic import tm
from dotenv import load_dotenv

def run_image_to_video():
    tm.configure(provider="provider")
    
    image = "/local/path/to/image"
    prompt = "this is the text prompt used in video generation"
    negative_prompt = "blurry, low quality, getty"
    
    required_args = {
        "prompt": prompt,
        "image": image
    }
    optional_args = {
        "negative_prompt": negative_prompt
    }
    
    tm.image_to_video(model="model-name", required_args=required_args, **optional_args)

def check_status(job_id: str):
    # updates generation_log.json
    tm.check_generation_status(job_id)

if __name__ == "__main__":
    load_dotenv()
    
    run_image_to_video()
    
    # job_id = "..."
    # check_status(job_id)
```

# Running Local Models

## How it works

Currently, local implementations consist of calling APIs to close-sourced models. The benefit of implementing in Tio Magic Animation Toolkit is that you can simultaneously compare multiple video models against each other, making informed decisions on which models you want to use.

Running a feature call (`tm.image_to_video(model="...", required_params="...")`) makes an API call to a closed-source model. These APIs do not have asynchronous capabilities, so you must wait for your image output. Luckily, it does not take too long (1-2 minutes) to generate.

## Development Setup

1. Ensure you have followed the **Getting Started - Installation** instructions above to prepare your local environment
2. Get the API key to the closed source model you want to use and paste your key into `.env`

## Veo 2.0 Generate 001

1. Ensure you have a `GOOGLE_API_KEY` and you have sufficient credits for the number of videos you want to create
2. Paste your key into `.env`

```python
from tiomagic import tm
from dotenv import load_dotenv

def veo_image_to_video():
    tm.configure(provider="local")
    
    required_args = {
        "prompt": "prompt for video",
        "image": "https:// url or local path to image",
    }
    optional_args = {
        ...
    }
    
    tm.image_to_video(model="veo-2.0-generate-001", required_args=required_args, **optional_args)
    
    return
```

**NOTE**: Veo2.0 does NOT have an asynchronous method of running. You must wait for the function to complete. Once the function finishes, your video will be under directory `output_videos`

## Luma AI Ray 2

1. Ensure you have a `LUMAAI_API_KEY` and you have sufficient credits for the number of videos you want to create
2. Paste your key into `.env`

```python
from tiomagic import tm
from dotenv import load_dotenv

def luma_image_to_video():
    tm.configure(provider="local")
    
    required_args = {
        "prompt": "prompt for video",
        "image": "https:// url path to image",
    }
    # no optional args for luma
    
    tm.image_to_video(model="luma-ray-2", required_args=required_args, **optional_args)
    
    return
```

**NOTES**:
- Images all must be URLs, they cannot be local images
- Luma AI API does NOT have an asynchronous method of running. You must wait for the function to complete. Once the function finishes, your video will be under directory `output_videos`

# Running Modal Models

## How it works

We use Modal because we anticipate that most people do not have the local computing power to boot up multiple video AI models. Modal solves this issue by running everything in their cloud. It does cost money, but Modal gives you a few credits upon registration which you can use to demo this toolkit.

Running a feature call (`tm.text_to_video(model="...", required_params="...")`) makes an API call to Modal, which starts an "app", loads a "container" that includes all of the files to make an inference call to the requested model, and runs a video generation with the given data.

You can visit your Modal dashboard to track the models you've launched and the requests you've made.

On the most basic plan, Modal only allows you to have 8 web endpoints deployed at a time. Every time you launch a new model, you are deploying 2 web endpoints (POST to run a generation, GET to check its status). If you deploy more than 4 models, you will run into errors running your new models. Please go onto your Modal dashboard, select the model you want to take down, and click **stop app** to deploy new models.

## Development Setup

1. Ensure you have followed the **Getting Started - Installation** instructions above to prepare your local environment
2. Create a Modal account. You will be given $3.00 of free credits to use, which can run a few video generations, depending on the GPU you select and the complexity of the feature you request to run
3. Create a Modal key and paste a `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in your `.env` file. You can create one on your Modal dashboard or via Modal CLI

## Example

Depending on the model and feature you use, ensure that your required and optional arguments match the name and type that the model is looking for. You can find these requirements under `/core/schemas`

```python
from tiomagic import tm
from dotenv import load_dotenv

def interpolate_example():
    tm.configure(provider="modal")
    
    required_args = {
        'first_frame': 'URL or Local path to first frame image',
        'last_frame': 'URL or Local path to last frame image',
        'prompt': "Cartoon styled painter Bob ross painting a tree on his canvas, then turns towards the camera and smiles to the audience."
    }
    
    optional_args = {...}
    
    tm.interpolate(model="wan2.1-vace-14b", required_args=required_args, **optional_args)

def check_status(job_id: str):
    # updates generation_log.json
    tm.check_generation_status(job_id)

if __name__ == "__main__":
    load_dotenv()
    interpolate_example() # run interpolate function once, output in generation_log.json
    
    # job_id = "..."
    # check_status(job_id) # run check status on job

# in directory of file, run python3 'file_name.py'
```

Once you successfully run a feature call on Modal, the job will show up in `generation_log.json`. You can check the job's status by running `check_status`.

Occasionally check on the status of the job. Once it is completed on Modal, running `check_status` will download the resulting video into `output_videos` directory in your local repository. You can also find the output video on your Modal dashboard under "Volumes".
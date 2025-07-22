# Tio Magic Animation Toolkit

Tio Magic Animation Toolkit is designed to simplify the use of video AI models for animation. The Animation Toolkit empowers animators, developers, and AI enthusiasts to easily generate animated videos without the pain of complex technical setup, local hardware limitations, and haphazard documentation.

## Supported Features
- Text to Video
- Image to Video
- Interpolate (in-betweens)
- Pose Guidance

## Supported Providers and Models

### Local
- Veo 2.0: veo-2.0-generate-001

### Modal
- Wan 2.1 14b: wan2.1-t2v-14b
- Wan 2.1 FLFV 14b: wan2.1-flf2v-14b-720p
- Wan 2.1 I2V 14b 720p: wan2.1-i2v-14b-720p
- Wan 2.1 Vace 14b: wan2.1-vace-14b
- LTX video: ltx-video
- Framepack I2V HY: framepack-i2v-hy
- Cogvideox 5b: cogvideox-5b
- Cogvideox 5b I2V: cogvideox-5b-image-to-video

## Getting Started - Installation

> **Note:** In the future, installation will be available via `pip install`

1. Open your Command Line Interface (CLI) - terminal on Mac or command prompt on Windows

2. Download Python:
   ```bash
   sudo apt install python3
   ```

3. Install Pip (Python's package manager):
   ```bash
   sudo apt install python3-pip
   ```

4. Clone this repository to a local directory

5. Change directories to the location where you cloned your repository:
   ```bash
   cd /path/to/repository
   ```

6. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

7. Activate the virtual environment (you should see `(venv)` on your command line):
   - **On MacOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows Command Prompt:**
     ```cmd
     venv\Scripts\activate
     ```

8. Install the package:
   ```bash
   pip install -e .
   ```
   > **Important:** Don't forget the DOT!

9. Depending on what provider(s) you are using, copy/paste the appropriate access keys to the `.env` file in the repository:
   - **Veo 2:** `GOOGLE_API_KEY`
   - **Modal:** `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`

10. Navigate to `quick_start.py`, configure the file to your needs (see examples below), and run:
    ```bash
    python3 quick_start.py
    ```

11. When you are done developing, run `deactivate` to exit the virtual environment

## Getting Started - Usage

Move to the directory where `quick_start.py` is located and run:
```bash
python3 quick_start.py
```

### Example Usage

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

## Local Models

### Veo 2.0 Generate 001

Ensure you have a Gemini/Google API key and you have sufficient credits for the number of videos you want to create. Paste your key into `.env`.

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
        # ...
    }

    tm.image_to_video(model="veo-2.0-generate-001", required_args=required_args, **optional_args)

    return
```

> **NOTE:** Veo 2.0 does NOT have an asynchronous method of running. You must wait for the function to complete. Once the function finishes, your video will be under directory `output_videos`

## Modal Models

*(Content for Modal models section to be added)*

## How to Contribute

*(Content for How to Contribute section to be added)*
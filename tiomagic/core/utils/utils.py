
import os
from pathlib import Path
import base64
from typing import Any, Dict


def is_local_path(path: str) -> bool:
    """Check if the path is a local file path."""
    if not isinstance(path, str):
        return False

    # Check if it's a URL
    if path.startswith(('http://', 'https://')):
        return False

    # Check if it's a local file path
    return os.path.exists(path) or Path(path).exists()

def local_image_to_base64(image_path: str) -> str:
    """Convert local image to base64 string."""
    import base64

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Convert to base64
        base64_str = base64.b64encode(image_data).decode('utf-8')

        # Add data URL prefix for image type detection
        # You might want to detect the actual image type here
        return f"data:image/png;base64,{base64_str}"

    except Exception as e:
        raise Exception(f"Failed to convert local image {image_path} to base64: {e}")

def extract_image_dimensions(image, data: Dict[str, Any]) -> Dict[str, Any]:
    """get PIL image dimensions and set them in data"""

    width, height = image.size
    data['width'] = width
    data['height'] = height
    print(f"Extracted image dimensions: {width}x{height}")
    return data



def url_video_to_bytes(url: str) -> bytes:
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def load_video_robust(video_source):
    try:
        # check if web url
        if isinstance(video_source, str):
            if video_source.startswith(('http://', 'https://')):
                print(f"Loading video from URL: {video_source}")
                return url_video_to_bytes(video_source)
            elif video_source.startswith('data:video'):
                print("Loading video from base64 data URL")
                # Extract base64 part from data URL
                if "," in video_source:
                    base64_str = video_source.split(",")[1]
                else:
                    base64_str = video_source
                return base64.b64decode(base64_str)
            # Check if it's a raw base64 string (without data URL prefix)
            elif len(video_source) > 100:  # Base64 strings are typically long
                try:
                    print("Loading video from base64 string")
                    return base64.b64decode(video_source)
                except Exception:
                    # If base64 decode fails, treat as local path
                    pass
        # If it's already bytes, return as-is
        elif isinstance(video_source, bytes):
            return video_source
        else:
            raise ValueError(f"Unsupported video source format: {type(video_source)}")
    except Exception as e:
        print(f"Error loading video from {video_source}: {e}")
        raise





def create_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%m%d_%H%M%S")
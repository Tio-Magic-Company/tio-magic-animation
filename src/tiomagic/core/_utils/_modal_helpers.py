"""Modal API helper functions for checking and deploying Modal applications.
These utilities help manage the lifecycle of Modal apps across your project.
"""
import os
from typing import Dict, Any
import modal
from modal.exception import NotFoundError
import PIL.Image


"""MODAY DEPLOYMENT"""
def check_status(app_name: str) -> Dict[str, Any]:
    """Check if a Modal app is already deployed and responsive.
    
    Parameters:
    - app_name: The name of your Modal app (str)
    
    Returns:
    - dict: Contains 'deployed' (bool), 'endpoints' (dict or None), and 'message' (str)
    """
    result = {
        'deployed': False,
        'endpoints': None,
        'message': ''
    }

    try:
        modal.App.lookup(app_name)
        result['deployed'] = True
        result['message'] = f"App '{app_name} is deployed"
        return result

    except NotFoundError:
        # runs if lookup fails
        result['message'] = f"App '{app_name}' is not currently deployed."
    except Exception as e:
        result['message'] = f"Error checking app deployment: {str(e)}"
    return result

def get_endpoints(app) -> list[str]:
    # maybe check responsiveness of endpoints here
    endpoints = app.registered_web_endpoints
    print('--> ENDPOINTS: ', endpoints)
    return endpoints

def check_modal_app_deployment(app, app_name) -> Dict[str, Any]:
    """Check whether Modal app is deployed, and if endpoints are available.
    Deploys Modal app if not yet deployed
    
    Parameters:
    - app: Modal app object
    - app_name: name of Modal app
    
    Returns:
    - dict: Contains 'success' (bool), 'endpoints' (list[str] or None), and 'message' (str)
    """
    result = {
    'success': False,
    'endpoints': None,
    'message': 'default'
    }

    try:
        # check if app is already deployed
        deployment_status = check_status(app_name)
        print("deployment status: f", deployment_status)
        if not deployment_status['deployed']:
            print("--> app is NOT deployed, deploying now. This may take a few minutes if it's the first time you are loading this model.")
            app.deploy()
        else:
            print("--> app is already deployed")
        result['success'] = True
        result['endpoints'] = get_endpoints(app)
        return result

    except Exception as e:
        result['message'] = f"Error deploying Modal app: {str(e)}"
    return result

def prepare_video_and_mask(img: PIL.Image.Image, height: int, width: int, num_frames: int, last_frame: PIL.Image.Image=None):
    img = img.resize((width, height))
    frames = [img]
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    if last_frame is None:
        frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 1))
    else:
        frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
        last_img = last_frame.resize((width, height))
        frames.append(last_img)

    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    if last_frame is None:
        mask = [mask_black, *[mask_white] * (num_frames - 1)]
    else:
        mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask

def load_image_robust(image_source):
    """Load image from URL, local path, or base64 string."""
    from diffusers.utils import load_image
    import base64
    from io import BytesIO
    import PIL.Image
    import numpy as np

    try:
        # Check if it's a web URL
        if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            print(f"Loading image from URL: {image_source}")
            image = load_image(image_source)

        # Check if it's a base64 string
        elif isinstance(image_source, str) and image_source.startswith('data:image'):
            print("Loading image from base64 data URL")
            # Extract base64 part
            if "," in image_source:
                base64_str = image_source.split(",")[1]
            else:
                base64_str = image_source

            image_data = base64.b64decode(base64_str)
            image = PIL.Image.open(BytesIO(image_data)).convert('RGB')
            # return image

        # Check if it's a local file path (for Modal container)
        elif isinstance(image_source, str) and os.path.exists(image_source):
            print(f"Loading image from local path: {image_source}")
            image = load_image(image_source)

        else:
            raise ValueError(f"Unsupported image source format: {type(image_source)}")
        
        # Add validation after loading
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        # Convert to numpy array for validation
        img_array = np.array(image)
        
        # Check for invalid values
        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
            print("Warning: Image contains invalid values (NaN/Inf), attempting to clean...")
            # Replace NaN with 0 and Inf with max value
            img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
            image = PIL.Image.fromarray(img_array.astype(np.uint8))
        
        return image
    except Exception as e:
        print(f"Error loading image from {image_source}: {e}")
        raise
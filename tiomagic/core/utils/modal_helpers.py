"""
Modal API helper functions for checking and deploying Modal applications.
These utilities help manage the lifecycle of Modal apps across your project.
"""
import sys
import multiprocessing
import subprocess
import time
import logging
from fastapi import Body
from fastapi.responses import JSONResponse
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import modal
from modal.exception import NotFoundError
import PIL.Image

"""MODAL VIDEO GENERATION TRACKING"""
@dataclass
class Generation:
    call_id: Optional[str] = None
    status: Optional[str] = None
    message: str = ''
    timestamp: Optional[str] = None
    prompt: Optional[str] = None
    optional_parameters: Optional[dict] = None
    result_video: Optional[str] = None

    def to_dict(self):
        return asdict(self)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

"""MODAY DEPLOYMENT"""
def check_status(app_name: str) -> Dict[str, Any]:
    """
    Check if a Modal app is already deployed and responsive.
    
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
        app_info = modal.App.lookup(app_name)
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
    print('ENDPOINTS: ', endpoints)
    return endpoints

def check_modal_app_deployment(app, app_name) -> Dict[str, Any]:
    """
    Check whether Modal app is deployed, and if endpoints are available.
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
            print("app is NOT deployed, deploying now")
            app.deploy()
        else:
            print("app is already deployed")
        result['success'] = True
        result['endpoints'] = get_endpoints(app)
        return result

    except Exception as e:
        result['message'] = f"Error deploying Modal app: {str(e)}"
    return result

"""MODAL ENDPOINTS"""
def create_web_inference_endpoint(generate_method, required_params=None, image_params=None):
    """
    Create a generalized web_inference endpoint for Modal classes.
    
    Args:
        generate_method: The generate method to call
        required_params: List of required parameter names (excluding prompt)
        image_params: Dict mapping parameter names to their image loading functions
    """
    def web_inference(data: dict=Body(...)):
        """
        FastAPI endpoint that runs on the class instance
        """
        from diffusers.utils import load_image

        prompt = data.get("prompt") #prompt always required
        if not prompt:
            return{
                "error": "A 'prompt' is required"
            }
        
        negative_prompt = data.get("negative_prompt", "")

        args = []

        if image_params:
            for param_name, load_func in image_params.items():
                param_value = data.get(param_name)
                if param_value:
                    if load_func == "load_image":
                        param_value = load_image(param_value)
                    elif callable(load_func):
                        param_value = load_func(param_value)
                args.append(param_value)

                #check if required
                if required_params and param_name in required_params and not param_value:
                    return {"error": f"A '{param_name}' is required."}
        
        args.extend([prompt, negative_prompt])
        
        # Log the parameters
        print(f"prompt: {prompt}")
        if image_params:
            for param_name in image_params.keys():
                param_value = data.get(param_name)
                if param_value:
                    print(f"{param_name}: {param_value}")
        print(f"negative prompt: {negative_prompt}")
        
        # Call the generate method
        call = generate_method.spawn(*args)
        
        return JSONResponse({"call_id": call.object_id})
    
    return web_inference

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

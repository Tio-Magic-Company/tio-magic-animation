"""
Modal API helper functions for checking and deploying Modal applications.
These utilities help manage the lifecycle of Modal apps across your project.
"""
import time
import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import modal
from modal.exception import NotFoundError

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

# def save_to_log(result):
#     import json
#     import os
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     repo_root = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
#     log_file_path = os.path.join(repo_root, 'generation_log.txt')

#     try:
#         # append result to the log file
#         with open(log_file_path, 'a', encoding='utf-8') as f:
#             f.write(json.dumps(result, indent=2) + '\n---\n')
#         print(f"Result saved to: {log_file_path}")
#     except Exception as e:
#         print(f"Error saving to log file: {e}")


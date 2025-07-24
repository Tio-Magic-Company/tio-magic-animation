from typing import Dict, Any, Tuple, List, Union
import os
from urllib.parse import urlparse

from .schemas import FEATURE_SCHEMAS

def get_schema(feature: str, model: str) -> Dict:
    """Get parameter schema for a feature/model combination"""
    if feature not in FEATURE_SCHEMAS:
        raise ValueError(f"Unknown feature: {feature}")
    feature_schemas = FEATURE_SCHEMAS[feature]

    if model not in feature_schemas:
        raise ValueError(f"Unknown model '{model}' for feature '{feature}'. Available models: {list(feature_schemas.keys())}")
    return feature_schemas[model]

def validate_parameters(feature: str, model: str, required_args: Dict, optional_args: Dict) -> Tuple[bool, Dict, str]:
    """Validate parameters for a specific feature/model.
    
    Returns:
        Tuple containing (success, validated_params, error_message)
    """
    try:
        # get schema
        schema = get_schema(feature, model)

        # validate required parameters
        for param_name, param_info in schema["required"].items():
            if param_name not in required_args:
                return False, {}, f"Missing required parameter: {param_name}"

            # type checking
            param_type = param_info["type"]
            if not is_valid_type(required_args[param_name], param_type):
                return False, {}, f"Parameter '{param_name}' should be of type {get_type_name(param_type)}"

            # image validation
            if param_name == 'image' or param_name == 'first_frame' or param_name == 'last_frame':
                valid, msg = validate_image_parameter(required_args[param_name])
                if not valid:
                    return False, {}, msg

        # process all required params
        validated_params = {}
        for param_name in schema["required"]:
            validated_params[param_name] = required_args[param_name]

        # add optional_args if provided by user
        for param_name, param_value in optional_args.items():
            if param_name in schema["optional"]:
                # verify type if provided
                schema_field = schema["optional"][param_name]
                param_type = schema_field["type"]
                if not is_valid_type(param_value, param_type):
                    return False, {}, f"Optional parameter '{param_name}' should be of type {get_type_name(param_type)}"
                validated_params[param_name] = param_value
            else:
                # param passed in by user is not in optional schema
                return False, {}, f"Parameter '{param_name}' is not a valid optional parameter"

        if feature == "pose_guidance":
            if not (validated_params["guiding_video"] or validated_params["pose_video"]):
                return False, {}, f"Parameter 'guiding_video' or 'pose_video' must be provided for f{feature}"

        return True, validated_params, ""
    except Exception as e:
        return False, {}, f"Validation error: {str(e)}"

def is_valid_type(value: Any, expected_type: Union[type, List[type]]) -> bool:
    """Check if value is of expected type, handling lists and unions"""
    if isinstance(expected_type, list):
        # for Union types (str, list)
        return any(isinstance(value, t) for t in expected_type)
    return isinstance(value, expected_type)

def get_type_name(type_info: Union[type, List[type]]) -> str:
    """Get readable name for type or union of types"""
    if isinstance(type_info, list):
        return " or ".join(t.__name__ for t in type_info)
    return type_info.__name__

def validate_image_parameter(image_param: str) -> Tuple[bool, str]:
    """Validate image path (local path or URL)"""
    # Check if it's a URL
    if image_param.startswith(('http://', 'https://')):
        try:
            result = urlparse(image_param)
            if all([result.scheme, result.netloc]):
                return True, ""
        except:
            return False, f"Invalid image URL: {image_param}"
    # Check if it's a local file
    elif os.path.exists(image_param):
        return True, ""
    else:
        return False, f"Image file not found: {image_param}"


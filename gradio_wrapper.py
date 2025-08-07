import gradio as gr
import subprocess
import os
import sys
from pathlib import Path
import json
import tempfile
from dotenv import load_dotenv
from tiomagic import tm

# load environment variables
load_dotenv()

class ToolkitWrapper:
    def __init__(self, repo_path=None):  # Fixed typo: __innit__ -> __init__
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()  # Fixed: Path.cwd -> Path.cwd()
        
    def validate_environment(self, providers = []):
        """Check if required environment variables are set.
        If a particular provider is selected, check if correct
        environment variables are set.
        """
        
    def get_providers(self):
        providers = tm.get_providers()
        providers.reverse()
        return providers
        
    def get_models(self, feature, provider):
        if provider:
            models = tm.get_models(feature, provider)
            return models if models else []
        return []
        
    def get_imp_schema(self, feature, provider, model):
        print("get schema, ", feature, provider, model)
        if feature and provider and model:
            schema = tm.get_schema(feature, model)
            return schema
        return {"required": {}, "optional": {}}  # Return empty schema if conditions not met
        
    def run_feature(self, func):
        try:
            print("run feature")
            job = func()[-1]
            print("JOB: ", job)
            return job
        except Exception as e:
            return str(e)
    def check_status(self, job_id):
        return tm.check_generation_status(job_id, returnJob=True)
            
    def text_to_video(self, provider, model, required_args, **kwargs):
        call = lambda: (
            tm.configure(provider=provider),
            tm.text_to_video(model=model, required_args=required_args, **kwargs)
        )
        result = self.run_feature(call)
        print(f"text to video returning: {result}")
        return result
    def image_to_video(self, provider, model, required_args, **kwargs):
        call = lambda: (
                tm.configure(provider=provider),
                tm.image_to_video(model=model, required_args=required_args, **kwargs)
            )
        result = self.run_feature(call)
        print(f"image to video returning: {result}")
        return result
    def interpolate(self, provider, model, required_args, **kwargs):
        call = lambda: (
                tm.configure(provider=provider),
                tm.interpolate(model=model, required_args=required_args, **kwargs)
            )
        result = self.run_feature(call)
        print(f"interpolate to video returning: {result}")
        return result
    def pose_guidance(self, provider, model, required_args, **kwargs):
        call = lambda: (
                tm.configure(provider=provider),
                tm.pose_guidance(model=model, required_args=required_args, **kwargs)
            )
        result = self.run_feature(call)
        print(f"pose guidance to video returning: {result}")
        return result

wrapper = ToolkitWrapper()

def create_input_component(arg_config, label):
    """Create appropriate Gradio component based on argument type
    arg_config: a dictionary of a particular argument
    label: name of argument
    """
    image_container = ["image", "first_frame", "last_frame"]
    arg_type = arg_config.get("type", None)
    description = arg_config.get("description", "")
    if arg_type:
        if arg_type == str and label in image_container:
            with gr.Row():
                image_comp = gr.Image(
                    type="filepath",
                    label=f"{label} File - choose one",
                )
                text_comp = gr.Textbox(
                    label=f"{label} URL - choose one",
                )
            return [image_comp, text_comp]
            # return gr.Row([
            #     gr.Image(type="filepath", label=f"{label} File or URL - choose one",),
            #     gr.Textbox(label=f"{label} File or URL - choose one",)  # Changed from Image to Textbox for URL
            # ])
        if arg_type == str:
            return gr.Textbox(label=label, value=arg_config.get("default", ""), info = description)
        elif arg_type == int or arg_type == float:
            return gr.Number(label=label, value=arg_config.get("default", None), info = description)
        elif arg_type == bool:
            return gr.Checkbox(label=label, value=arg_config.get("default", False), info=description)
        else:
            return gr.Textbox(label=label, value=arg_config.get("default", ""), info = description)           

def cast_value(value, target_type, default=None):
    """
    Safely cast a value to the target type.
    Returns the default if casting fails or value is empty.
    """
    # Handle empty values
    if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
        return default
    
    try:
        if target_type == int:
            # Handle decimal strings like "1.0" -> 1
            return int(float(str(value).strip()))
        
        elif target_type == float:
            return float(str(value).strip())
        
        elif target_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower().strip() in ('true', '1', 'yes', 'on', 't', 'y')
            return bool(value)
        
        elif target_type == str:
            return str(value).strip() if value is not None else ""
        
        elif target_type == list:
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                # Try to parse JSON array
                import json
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, list) else [value]
                except:
                    # Split comma-separated values
                    return [v.strip() for v in value.split(',') if v.strip()]
            return [value] if value else []
        
        elif target_type == dict:
            if isinstance(value, dict):
                return value
            elif isinstance(value, str):
                import json
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, dict) else {"value": value}
                except:
                    return {"value": value}
            return {"value": value}
        
        else:
            # For unknown types, return as-is
            return value
            
    except Exception as e:
        print(f"Cast error for {value} to {target_type}: {e}")
        return default

def values_are_different(value, default, value_type):
    """
    Compare two values after casting them to the same type.
    Returns True if they are different, False if they are the same.
    """
    # Cast both to the same type
    typed_value = cast_value(value, value_type)
    typed_default = cast_value(default, value_type)
    
    # Special handling for None/empty comparisons
    if typed_value is None and typed_default is None:
        return False
    
    # Special handling for lists (compare contents, not reference)
    if value_type == list:
        if typed_value is None or typed_default is None:
            return typed_value != typed_default
        return sorted(typed_value) != sorted(typed_default)
    
    # Special handling for dicts
    if value_type == dict:
        if typed_value is None or typed_default is None:
            return typed_value != typed_default
        return typed_value != typed_default
    
    # Standard comparison
    return typed_value != typed_default

def process_optional_arguments(args, args_config, optional_inputs, optional_defaults, start_idx=0):
    """
    Process optional arguments and return only those that differ from defaults.
    
    Args:
        args: The arguments passed from Gradio components
        args_config: The schema configuration for the arguments
        optional_inputs: Dictionary of optional input components
        optional_defaults: Dictionary of default values
        start_idx: Starting index in args array
    
    Returns:
        tuple: (optional_args dict, next_arg_idx)
    """
    optional_args = {}
    arg_idx = start_idx
    
    for label, component in optional_inputs.items():
        if arg_idx >= len(args):
            break
            
        # Get argument configuration
        arg_config = args_config.get("optional", {}).get(label, {})
        expected_type = arg_config.get("type", str)
        default_value = optional_defaults.get(label)
        
        # Handle special component types
        if isinstance(component, gr.Row):
            # Image/file inputs with URL alternative
            file_input = args[arg_idx] if arg_idx < len(args) else None
            url_input = args[arg_idx + 1] if arg_idx + 1 < len(args) else None
            value = file_input if file_input else url_input
            arg_idx += 2
            
            # Simple comparison for file/URL inputs
            if value and value != default_value:
                optional_args[label] = value
        else:
            # Regular components
            value = args[arg_idx]
            arg_idx += 1
            
            # Check if value differs from default
            if values_are_different(value, default_value, expected_type):
                # Store the properly typed value
                optional_args[label] = cast_value(value, expected_type, default_value)
    
    return optional_args, arg_idx

def process_required_arguments(args, args_config, required_inputs, start_idx=0):
    """
    Process required arguments.
    
    Args:
        args: The arguments passed from Gradio components
        args_config: The schema configuration for the arguments
        required_inputs: Dictionary of required input components
        start_idx: Starting index in args array
    
    Returns:
        tuple: (required_args dict, next_arg_idx)
    """
    required_args = {}
    arg_idx = start_idx
    
    for label, component in required_inputs.items():
        if arg_idx >= len(args):
            raise ValueError(f"Missing required argument: {label}")
        
        # Get argument configuration
        arg_config = args_config.get("required", {}).get(label, {})
        expected_type = arg_config.get("type", str)
        
        # Handle special component types
        # if isinstance(component, gr.Row):
        if isinstance(component, list):
            # Image/file inputs with URL alternative
            file_input = args[arg_idx] if arg_idx < len(args) else None
            url_input = args[arg_idx + 1] if arg_idx + 1 < len(args) else None
            value = file_input if file_input else url_input
            arg_idx += 2
            
            # Use file if provided, otherwise use URL
            value = file_input if file_input else url_input
            
            if not value:
                raise ValueError(f"Required argument '{label}' is missing")
            required_args[label] = value
            print(f"Image argument '{label}': file={file_input}, url={url_input}, chosen={value}")

        else:
            # Regular components
            value = args[arg_idx]
            arg_idx += 1
            
            # Cast to expected type
            typed_value = cast_value(value, expected_type)
            if typed_value is None and expected_type != type(None):
                raise ValueError(f"Required argument '{label}' is empty or invalid")
            
            required_args[label] = typed_value
    
    return required_args, arg_idx
    
with gr.Blocks(title="Tio Magic Animation Toolkit") as demo:
    gr.Markdown("# Tio Magic Animation Toolkit")
    gr.Markdown("Tio Magic Animation Toolkit is designed to simplify the use of video AI models for animation. The Animation Toolkit empowers animators, developers, and AI enthusiasts to easily generate animated videos without the pain of complex technical setup, local hardware limitations, and haphazard documentation.")
    gr.Markdown("[Github](https://github.com/Tio-Magic-Company/tio-magic-animation) [Animation Toolkit Documentation]")

    def load_generation_log():
        try:
            with open("generation_log.json", "r") as f:
                jobs = json.load(f)  # This is already a list
            
            # Get last 5 jobs if there are more than 5
            if isinstance(jobs, object):
                jobs = jobs["jobs"][-1:] if len(jobs) > 5 else jobs
                print(f"Loaded {len(jobs)} jobs, ")
                return jobs
            
            return []
            
        except Exception as e:
            print(f"Error loading log: {e}")
            return []
    
    # JSON Generation Log
    with gr.Accordion(label = "Generation Log", open = False):
        generation_log_component = gr.JSON(
            value=load_generation_log(), 
            label="Generation Log"
        )
        
        refresh_btn = gr.Button("Refresh Log")
        
        refresh_btn.click(
            fn=load_generation_log,
            outputs=[generation_log_component]
        )
    with gr.Tab("Text to Video"):
        # T2V
        # Create shared references
        t2v_output_components = {
            'status_text': None,
            'output_video': None
        }
        
        with gr.Row():
            with gr.Column(scale=1):
                provider_dropdown = gr.Dropdown(
                    choices = wrapper.get_providers(),
                    label = "Provider",
                    interactive = True
                )
                model_dropdown = gr.Dropdown(
                    choices = wrapper.get_models(feature="text_to_video", provider=provider_dropdown.value) if provider_dropdown.value else [],
                    label = "Model",
                    interactive = True
                )

                # Dynamic arguments section using @gr.render
                @gr.render(inputs=[provider_dropdown, model_dropdown])
                def render_arguments(provider, model):
                    components = []                    
                    if provider and model:
                        args_config = wrapper.get_imp_schema(
                            feature="text_to_video", 
                            provider=provider, 
                            model=model
                        )
                        
                        # Store references to input components AND their default values
                        required_inputs = {}
                        optional_inputs = {}
                        optional_defaults = {}  # Store default values
                        
                        # Required arguments
                        if args_config.get("required"):
                            components.append(gr.Markdown("### Required Arguments"))
                            for label, arg_config in args_config.get("required", {}).items():
                                component = create_input_component(arg_config, label)
                                required_inputs[label] = component
                                components.append(component)
                        
                        # Optional arguments
                        if args_config.get("optional"):
                            with gr.Accordion(label = "Optional Arguments", open = False):
                                for label, arg_config in args_config.get("optional", {}).items():
                                    component = create_input_component(arg_config, label)
                                    optional_inputs[label] = component
                                    # Store the default value
                                    optional_defaults[label] = arg_config.get("default", None)
                                    components.append(component)
                        
                        # Generate button
                        generate_btn = gr.Button("Generate Video", variant="primary")
                        components.append(generate_btn)
                        
                        # Create handler function
                        def generate_video_handler(*args):
                            try:
                                # Build required_args dictionary
                                required_args, next_idx = process_required_arguments(
                                    args,
                                    args_config,
                                    required_inputs,
                                    start_idx = 0
                                )
                                
                                # Build optional arguments dictionary
                                optional_args, _ = process_optional_arguments(
                                    args,
                                    args_config,
                                    optional_inputs,
                                    optional_defaults,
                                    start_idx = next_idx
                                )
                            
                                
                                # Log what we're sending for debugging
                                print(f"Required args: {required_args}")
                                print(f"Optional args (changed from defaults): {optional_args}")
                                
                                # Call text_to_video with the collected arguments
                                result = wrapper.text_to_video(provider, model, required_args, **optional_args)
                                print("JOB RESULT: ", result)
                                print("check if has attribute: ", hasattr(result, 'job_id'))
                                # Return the result
                                if hasattr(result, 'job_id'):
                                    # It's a Job object
                                    job_info = {
                                        "job_id": result.job_id,
                                        "feature": result.feature,
                                        "model": result.model,
                                        "provider": result.provider,
                                        "creation_time": result.creation_time,
                                        "status": "Job submitted successfully"
                                    }
                                    print("in has attr. job info saved: ", job_info)
                                    
                                    # Check if there's already a result video
                                    video_path = None
                                    if result.generation and result.generation.get('result_video'):
                                        print("video downloaded")
                                        video_path = result.generation['result_video']
                                        if os.path.exists(video_path):
                                            # return video_path, f"✅ Video generated successfully! Job ID: {result.job_id}"
                                            return gr.update(value=video_path), gr.update(value=f"✅ Video ready! Job ID: {result.job_id}")
                                    print("video NOT downloaded, returning job id")
                                    # return None, f"📋 Job submitted successfully! Job ID: {result.job_id}\nUse the 'Check Status' feature to monitor progress.\n{job_info}"
                                    return gr.update(value=None), gr.update(value=job_info)
                                else:
                                    return gr.update(value=None), gr.update(value=f"📋 Job submitted: {result}")
                                    # return None, f"📋 Job submitted: {result}"
                            except ValueError as ve:
                                # return None, f"❌ Validation Error: {str(ve)}"
                                return gr.update(value=None), gr.update(value=f"❌ Error: {str(ve)}")


                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                # return None, f"❌ Error: {str(e)}"
                                return gr.update(value=None), gr.update(value=f"❌ Error: {str(e)}")


                        
                        # Collect all input components (excluding Markdown and Button)
                        all_inputs = []
                        for comp in components:
                            if isinstance(comp, gr.Row):
                                # If it's a Row (image input), add both components
                                all_inputs.extend(comp.children)
                            elif not isinstance(comp, (gr.Markdown, gr.Button)):
                                all_inputs.append(comp)
                        # Connect the handler to the button
                        generate_btn.click(
                            fn=generate_video_handler,
                            inputs=all_inputs,
                            outputs=[
                                t2v_output_components['output_video'],
                                t2v_output_components['status_text']
                            ]
                        )
                    
                    return components
            with gr.Column(scale=1):
                # Create and store references
                check_status_input = gr.Textbox(label="Check Status, paste job_id", lines=1)
                check_status_btn = gr.Button("Check Status")
                t2v_output_components['status_text'] = gr.Textbox(label="Status", lines=3)
                t2v_output_components['output_video'] = gr.Video(label="Generated Video")
                
                def check_job_status(job_id):
                    # return json with status info and video path
                    try:
                        job = wrapper.check_status(job_id)
                        job_dict = job.to_dict()

                        video_path = None
                        generation = job_dict.get("generation", {})
                        if generation and generation.get("result_video"):
                            video_path = generation["result_video"]
                        return json.dumps(job_dict, indent=2), video_path
                    except Exception as e:
                        error_info = {"error": str(e)}
                        return json.dumps(error_info, indent=2), None


                check_status_btn.click(
                    fn=check_job_status,
                    inputs=[check_status_input],
                    outputs=[t2v_output_components['status_text'], t2v_output_components['output_video']]
                )
            
            
        with gr.Row():
            with gr.Column(scale=1):
                def update_models(provider):
                    return gr.update(
                        choices=wrapper.get_models(feature="text_to_video", provider=provider) if provider else [],
                        value=None
                    )
                # when provider changes, update model_dropdown
                provider_dropdown.change(
                    fn=update_models,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown]
                )
            

    with gr.Tab("Image to Video"):
        # I2V
        i2v_output_components = {
            'status_text': None,
            'output_video': None
        }

        with gr.Row():
            with gr.Column(scale=1):
                provider_dropdown = gr.Dropdown(
                    choices = wrapper.get_providers(),
                    label = "Provider",
                    interactive = True
                )
                model_dropdown = gr.Dropdown(
                    choices = wrapper.get_models(feature="image_to_video", provider=provider_dropdown.value) if provider_dropdown.value else [],
                    label = "Model",
                    interactive = True
                )
                
                # Dynamic arguments section using @gr.render
                @gr.render(inputs=[provider_dropdown, model_dropdown])
                def render_arguments(provider, model):
                    components = []
                    
                    if provider and model:
                        args_config = wrapper.get_imp_schema(
                            feature="image_to_video", 
                            provider=provider, 
                            model=model
                        )
                        
                        # Store references to input components AND their default values
                        required_inputs = {}
                        optional_inputs = {}
                        optional_defaults = {}  # Store default values
                        
                        # Required arguments
                        if args_config.get("required"):
                            components.append(gr.Markdown("### Required Arguments"))
                            for label, arg_config in args_config.get("required", {}).items():
                                component = create_input_component(arg_config, label)
                                if isinstance(component, list):
                                    required_inputs[label] = component
                                    components.extend(component) #TODO: only add one component
                                else: 
                                    required_inputs[label] = component
                                    components.append(component)
                        
                        # Optional arguments
                        if args_config.get("optional"):
                            with gr.Accordion(label = "Optional Arguments", open = False):
                                for label, arg_config in args_config.get("optional", {}).items():
                                    component = create_input_component(arg_config, label)
                                    optional_inputs[label] = component
                                    # Store the default value
                                    optional_defaults[label] = arg_config.get("default", None)
                                    components.append(component)
                        
                        # Generate button
                        generate_btn = gr.Button("Generate Video", variant="primary")
                        components.append(generate_btn)
                        
                        # Create handler function
                        def generate_video_handler(*args):
                            try:
                               # Build required_args dictionary
                                required_args, next_idx = process_required_arguments(
                                    args,
                                    args_config,
                                    required_inputs,
                                    start_idx = 0
                                )
                                
                                # Build optional arguments dictionary
                                optional_args, _ = process_optional_arguments(
                                    args,
                                    args_config,
                                    optional_inputs,
                                    optional_defaults,
                                    start_idx = next_idx
                                )
                                
                                # Log what we're sending for debugging
                                print(f"Required args: {required_args}")
                                print(f"Optional args (changed from defaults): {optional_args}")
                                
                                # Call text_to_video with the collected arguments
                                result = wrapper.image_to_video(provider, model, required_args, **optional_args)
                                
                                # Return the result
                                if isinstance(result, str) and os.path.exists(result):
                                    return result, f"✅ Video generated successfully!"
                                else:
                                    return None, f"📋 Job submitted: {result}"
                            except ValueError as ve:
                                return None, f"❌ Validation Error: {str(ve)}"     
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                return None, f"❌ Error: {str(e)}"
                        
                        # Collect all input components (excluding Markdown and Button)
                        all_inputs = []
                        for comp in components:
                            # if isinstance(comp, gr.Row):
                                # If it's a Row (image input), add both components
                                # all_inputs.extend(comp.children)
                            if isinstance(comp, (gr.Markdown, gr.Button)):
                                continue
                            all_inputs.append(comp)
                        # Connect the handler to the button
                        generate_btn.click(
                            fn=generate_video_handler,
                            inputs=all_inputs,
                            outputs=[
                                i2v_output_components['output_video'],
                                i2v_output_components['status_text']
                            ]
                        )
                    
                    return components
                
            with gr.Column(scale=1):
                # Create and store references
                check_status_input = gr.Textbox(label="Check Status, paste job_id", lines=1)
                check_status_btn = gr.Button("Check Status")
                i2v_output_components['status_text'] = gr.Textbox(label="Status", lines=3)
                i2v_output_components['output_video'] = gr.Video(label="Generated Video")
                
                def check_job_status(job_id):
                    # return json with status info and video path
                    try:
                        job = wrapper.check_status(job_id)
                        job_dict = job.to_dict()

                        video_path = None
                        generation = job_dict.get("generation", {})
                        if generation and generation.get("result_video"):
                            video_path = generation["result_video"]
                        return json.dumps(job_dict, indent=2), video_path
                    except Exception as e:
                        error_info = {"error": str(e)}
                        return json.dumps(error_info, indent=2), None


                check_status_btn.click(
                    fn=check_job_status,
                    inputs=[check_status_input],
                    outputs=[i2v_output_components['status_text'], i2v_output_components['output_video']]
                )
            
            
        with gr.Row():
            with gr.Column(scale=1):
                def update_models(provider):
                    return gr.update(
                        choices=wrapper.get_models(feature="image_to_video", provider=provider) if provider else [],
                        value=None
                    )
                # when provider changes, update model_dropdown
                provider_dropdown.change(
                    fn=update_models,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown]
                )
    with gr.Tab("Interpolate to Video"):
        # interpolate
        interpolate_output_components = {
            'status_text': None,
            'output_video': None
        }
        with gr.Row():
            with gr.Column(scale=1):
                provider_dropdown = gr.Dropdown(
                    choices = wrapper.get_providers(),
                    label = "Provider",
                    interactive = True
                )
                model_dropdown = gr.Dropdown(
                    choices = wrapper.get_models(feature="interpolate", provider=provider_dropdown.value) if provider_dropdown.value else [],
                    label = "Model",
                    interactive = True
                )
                
                # Dynamic arguments section using @gr.render
                @gr.render(inputs=[provider_dropdown, model_dropdown])
                def render_arguments(provider, model):
                    components = []
                    
                    if provider and model:
                        args_config = wrapper.get_imp_schema(
                            feature="interpolate", 
                            provider=provider, 
                            model=model
                        )
                        
                        # Store references to input components AND their default values
                        required_inputs = {}
                        optional_inputs = {}
                        optional_defaults = {}  # Store default values
                        
                        # Required arguments
                        if args_config.get("required"):
                            components.append(gr.Markdown("### Required Arguments"))
                            for label, arg_config in args_config.get("required", {}).items():
                                component = create_input_component(arg_config, label)
                                
                                if isinstance(component, list):
                                    required_inputs[label] = component  # Store the list
                                    components.extend(component)  # Add both components
                                else:
                                    required_inputs[label] = component
                                    components.append(component)
                        
                        # Optional arguments
                        if args_config.get("optional"):
                            with gr.Accordion(label = "Optional Arguments", open = False):
                                for label, arg_config in args_config.get("optional", {}).items():
                                    component = create_input_component(arg_config, label)
                                    optional_inputs[label] = component
                                    # Store the default value
                                    optional_defaults[label] = arg_config.get("default", None)
                                    components.append(component)
                        
                        # Generate button
                        generate_btn = gr.Button("Generate Video", variant="primary")
                        components.append(generate_btn)
                        
                        # Create handler function
                        def generate_video_handler(*args):
                            try:
                                # Build required_args dictionary
                                required_args, next_idx = process_required_arguments(
                                    args,
                                    args_config,
                                    required_inputs,
                                    start_idx = 0
                                )
                                
                                # Build optional arguments dictionary
                                optional_args, _ = process_optional_arguments(
                                    args,
                                    args_config,
                                    optional_inputs,
                                    optional_defaults,
                                    start_idx = next_idx
                                )
                                
                                # Log what we're sending for debugging
                                print(f"Required args: {required_args}")
                                print(f"Optional args (changed from defaults): {optional_args}")
                                
                                # Call text_to_video with the collected arguments
                                result = wrapper.interpolate(provider, model, required_args, **optional_args)
                                
                                # Return the result
                                if isinstance(result, str) and os.path.exists(result):
                                    return result, "Video generated successfully!"
                                else:
                                    return None, f"Job submitted: {result}"
                            except ValueError as ve:
                                return None, f"❌ Validation Error: {str(ve)}"  
                            except Exception as e:
                                return None, f"Error: {str(e)}"
                        
                        # Collect all input components (excluding Markdown and Button)
                        all_inputs = []
                        for comp in components:
                            if not isinstance(comp, (gr.Markdown, gr.Button)):
                                continue
                            all_inputs.append(comp)
                        # Connect the handler to the button
                        generate_btn.click(
                            fn=generate_video_handler,
                            inputs=all_inputs,
                            outputs=[
                                interpolate_output_components['output_video'],
                                interpolate_output_components['status_text']
                            ]
                        )
                    
                    return components
                
            with gr.Column(scale=1):
                # Create and store references
                check_status_input = gr.Textbox(label="Check Status, paste job_id", lines=1)
                check_status_btn = gr.Button("Check Status")
                interpolate_output_components['status_text'] = gr.Textbox(label="Status", lines=3)
                interpolate_output_components['output_video'] = gr.Video(label="Generated Video")
                
                def check_job_status(job_id):
                    # return json with status info and video path
                    try:
                        job = wrapper.check_status(job_id)
                        job_dict = job.to_dict()

                        video_path = None
                        generation = job_dict.get("generation", {})
                        if generation and generation.get("result_video"):
                            video_path = generation["result_video"]
                        return json.dumps(job_dict, indent=2), video_path
                    except Exception as e:
                        error_info = {"error": str(e)}
                        return json.dumps(error_info, indent=2), None


                check_status_btn.click(
                    fn=check_job_status,
                    inputs=[check_status_input],
                    outputs=[interpolate_output_components['status_text'], interpolate_output_components['output_video']]
                )
            
            
        with gr.Row():
            with gr.Column(scale=1):
                def update_models(provider):
                    return gr.update(
                        choices=wrapper.get_models(feature="interpolate", provider=provider) if provider else [],
                        value=None
                    )
                # when provider changes, update model_dropdown
                provider_dropdown.change(
                    fn=update_models,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown]
                )
    with gr.Tab("Pose Guidance To Video"):
        # pose guidance
        pg_output_components = {
            'status_text': None,
            'output_video': None
        }
        with gr.Row():
            with gr.Column(scale=1):
                provider_dropdown = gr.Dropdown(
                    choices = wrapper.get_providers(),
                    label = "Provider",
                    interactive = True
                )
                model_dropdown = gr.Dropdown(
                    choices = wrapper.get_models(feature="pose_guidance", provider=provider_dropdown.value) if provider_dropdown.value else [],
                    label = "Model",
                    interactive = True
                )
                
                # Dynamic arguments section using @gr.render
                @gr.render(inputs=[provider_dropdown, model_dropdown])
                def render_arguments(provider, model):
                    components = []
                    
                    if provider and model:
                        args_config = wrapper.get_imp_schema(
                            feature="pose_guidance", 
                            provider=provider, 
                            model=model
                        )
                        
                        # Store references to input components AND their default values
                        required_inputs = {}
                        optional_inputs = {}
                        optional_defaults = {}  # Store default values
                        
                        # Required arguments
                        if args_config.get("required"):
                            components.append(gr.Markdown("### Required Arguments"))
                            for label, arg_config in args_config.get("required", {}).items():
                                component = create_input_component(arg_config, label)

                                # Check if it's a list (image + URL inputs)
                                if isinstance(component, list):
                                    required_inputs[label] = component  # Store the list
                                    components.extend(component)  # Add both components
                                else:
                                    required_inputs[label] = component
                                    components.append(component)
                        
                        # Optional arguments
                        if args_config.get("optional"):
                            with gr.Accordion(label = "Optional Arguments", open = False):
                                for label, arg_config in args_config.get("optional", {}).items():
                                    component = create_input_component(arg_config, label)
                                    optional_inputs[label] = component
                                    # Store the default value
                                    optional_defaults[label] = arg_config.get("default", None)
                                    components.append(component)
                        
                        # Generate button
                        generate_btn = gr.Button("Generate Video", variant="primary")
                        components.append(generate_btn)
                        
                        # Create handler function
                        def generate_video_handler(*args):
                            try:
                                # Build required_args dictionary
                                required_args, next_idx = process_required_arguments(
                                    args,
                                    args_config,
                                    required_inputs,
                                    start_idx = 0
                                )
                                
                                # Build optional arguments dictionary
                                optional_args, _ = process_optional_arguments(
                                    args,
                                    args_config,
                                    optional_inputs,
                                    optional_defaults,
                                    start_idx = next_idx
                                )
                                
                                # Log what we're sending for debugging
                                print(f"Required args: {required_args}")
                                print(f"Optional args (changed from defaults): {optional_args}")
                                
                                # Call text_to_video with the collected arguments
                                result = wrapper.pose_guidance(provider, model, required_args, **optional_args)
                                
                                # Return the result
                                if isinstance(result, str) and os.path.exists(result):
                                    return result, "Video generated successfully!"
                                else:
                                    return None, f"Job submitted: {result}"
                            except ValueError as ve:
                                return None, f"❌ Validation Error: {str(ve)}"  
                            except Exception as e:
                                return None, f"Error: {str(e)}"
                        
                        # Collect all input components (excluding Markdown and Button)
                        all_inputs = []
                        for comp in components:
                            if not isinstance(comp, (gr.Markdown, gr.Button)):
                                continue
                            all_inputs.append(comp)
                        # Connect the handler to the button
                        generate_btn.click(
                            fn=generate_video_handler,
                            inputs=all_inputs,
                            outputs=[
                                pg_output_components['output_video'],
                                pg_output_components['status_text']
                            ]
                        )
                    
                    return components
                
            with gr.Column(scale=1):
                # Create and store references
                check_status_input = gr.Textbox(label="Check Status, paste job_id", lines=1)
                check_status_btn = gr.Button("Check Status")
                pg_output_components['status_text'] = gr.Textbox(label="Status", lines=3)
                pg_output_components['output_video'] = gr.Video(label="Generated Video")
                
                def check_job_status(job_id):
                    # return json with status info and video path
                    try:
                        job = wrapper.check_status(job_id)
                        job_dict = job.to_dict()

                        video_path = None
                        generation = job_dict.get("generation", {})
                        if generation and generation.get("result_video"):
                            video_path = generation["result_video"]
                        return json.dumps(job_dict, indent=2), video_path
                    except Exception as e:
                        error_info = {"error": str(e)}
                        return json.dumps(error_info, indent=2), None


                check_status_btn.click(
                    fn=check_job_status,
                    inputs=[check_status_input],
                    outputs=[pg_output_components['status_text'], pg_output_components['output_video']]
                )
            
            
        with gr.Row():
            with gr.Column(scale=1):
                def update_models(provider):
                    return gr.update(
                        choices=wrapper.get_models(feature="pose_guidance", provider=provider) if provider else [],
                        value=None
                    )
                # when provider changes, update model_dropdown
                provider_dropdown.change(
                    fn=update_models,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown]
                )
# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
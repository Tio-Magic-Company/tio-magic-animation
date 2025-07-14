from . import image_to_video, text_to_video, interpolate

FEATURE_SCHEMAS = {
    "text_to_video": text_to_video.SCHEMAS,
    "image_to_video": image_to_video.SCHEMAS,
    "interpolate": interpolate.SCHEMAS,
}
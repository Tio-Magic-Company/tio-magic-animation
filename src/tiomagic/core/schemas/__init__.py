from . import image_to_video, interpolate, pose_guidance, text_to_video

FEATURE_SCHEMAS = {
    "text_to_video": text_to_video.SCHEMAS,
    "image_to_video": image_to_video.SCHEMAS,
    "interpolate": interpolate.SCHEMAS,
    "pose_guidance": pose_guidance.SCHEMAS,
}
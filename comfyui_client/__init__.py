"""
ComfyUI Client - A Python library for interacting with ComfyUI API.

This library provides a simple interface to submit workflows,
monitor progress, and retrieve generated images/videos from ComfyUI.
"""

from comfyui_client.client import ComfyUIClient
from comfyui_client.workflow import Workflow, WorkflowBuilder
from comfyui_client.exceptions import (
    ComfyUIError,
    ComfyUIConnectionError,
    WorkflowError,
    ComfyUITimeoutError,
    QueueError,
    ImageError,
)
from comfyui_client.utils import (
    save_media,
    save_medias,
    save_image,
    save_images,
    save_video,
    save_videos,
    load_image,
    random_seed,
    create_masked_image,
)

__version__ = '0.1.0'
__all__ = [
    'ComfyUIClient',
    'Workflow',
    'WorkflowBuilder',
    'ComfyUIError',
    'ComfyUIConnectionError',
    'WorkflowError',
    'ComfyUITimeoutError',
    'QueueError',
    'ImageError',
    'save_media',
    'save_medias',
    'save_image',
    'save_images',
    'save_video',
    'save_videos',
    'load_image',
    'random_seed',
    'create_masked_image',
]

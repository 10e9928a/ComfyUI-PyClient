"""
Utility functions for ComfyUI client.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, override
from collections.abc import Mapping


def random_seed() -> int:
    """
    Generate a random seed value.

    :return: Random integer suitable for use as a seed
    """
    return random.randint(0, 2**32 - 1)


def save_media(
    data: bytes,
    filepath: str | Path,
) -> Path:
    """
    Save media data (image/video) to a file.

    :param data: Media bytes (image or video)
    :param filepath: Target file path
    :return: Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        _ = f.write(data)
    return filepath


def save_medias(
    medias: list[bytes],
    output_dir: str | Path,
    prefix: str = 'media',
    extension: str = 'png',
) -> list[Path]:
    """
    Save multiple media files (images/videos) to a directory.

    :param medias: List of media bytes
    :param output_dir: Output directory
    :param prefix: Filename prefix
    :param extension: File extension (png, jpg, mp4, webm, gif, etc.)
    :return: List of paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for i, media_data in enumerate(medias):
        filepath = output_dir / f'{prefix}_{i:04d}.{extension}'
        _ = save_media(media_data, filepath)
        saved_paths.append(filepath)

    return saved_paths


# Aliases for backward compatibility
def save_image(image_data: bytes, filepath: str | Path) -> Path:
    """Alias for save_media (backward compatibility)."""
    return save_media(image_data, filepath)


def save_images(
    images: list[bytes],
    output_dir: str | Path,
    prefix: str = 'image',
    extension: str = 'png',
) -> list[Path]:
    """Alias for save_medias (backward compatibility)."""
    return save_medias(images, output_dir, prefix, extension)


def save_video(video_data: bytes, filepath: str | Path) -> Path:
    """Alias for save_media (backward compatibility)."""
    return save_media(video_data, filepath)


def save_videos(
    videos: list[bytes],
    output_dir: str | Path,
    prefix: str = 'video',
    extension: str = 'mp4',
) -> list[Path]:
    """Alias for save_medias (backward compatibility)."""
    return save_medias(videos, output_dir, prefix, extension)


def load_image(filepath: str | Path) -> bytes:
    """
    Load an image file as bytes.

    :param filepath: Image file path
    :return: Image data as bytes
    """
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        return f.read()


def create_masked_image(
    image: str | Path | bytes,
    mask: str | Path | bytes,
    output: str | Path | None = None,
    invert_mask: bool = True,
    normalize_mask: bool = True,
) -> bytes:
    """
    Merge an image and mask into an RGBA image with alpha channel.

    The alpha channel is derived from the mask, which can be used by
    ComfyUI's LoadImage node to provide both IMAGE and MASK outputs.

    :param image: Source image path or bytes
    :param mask: Mask image path or bytes (non-black areas = area to repaint)
    :param output: Optional output file path to save the result
    :param invert_mask: If True, invert the mask (black becomes white)
    :param normalize_mask: If True, normalize gray masks to full black/white
                          (handles ComfyUI exported masks which use gray instead of white)
    :return: RGBA image as PNG bytes
    """
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        raise ImportError('No module named Pillow')

    # Load image
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = Image.open(BytesIO(image))
    img = img.convert('RGB')

    # Load mask
    if isinstance(mask, (str, Path)):
        mask_img = Image.open(mask)
    else:
        mask_img = Image.open(BytesIO(mask))
    mask_img = mask_img.convert('L')  # Convert to grayscale

    # Resize mask to match image if needed
    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, Image.Resampling.LANCZOS)

    # Normalize mask: convert gray values to full white
    # This handles ComfyUI exported masks which use gray (~128-180) instead of white (255)
    if normalize_mask:
        import numpy as np
        mask_array = np.array(mask_img)
        # Any non-zero pixel becomes 255 (white)
        mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_array, mode='L')

    # Invert mask if requested
    if invert_mask:
        from PIL import ImageOps
        mask_img = ImageOps.invert(mask_img)

    # Create RGBA image with alpha channel from mask
    rgba = Image.new('RGBA', img.size)
    rgba.paste(img, (0, 0))
    rgba.putalpha(mask_img)

    # Save to bytes
    buffer = BytesIO()
    rgba.save(buffer, format='PNG')
    result = buffer.getvalue()

    # Optionally save to file
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'wb') as f:
            _ = f.write(result)

    return result


def parse_server_address(address: str) -> tuple[str, int]:
    """
    Parse a server address string into host and port.

    :param address: Server address (e.g., '127.0.0.1:8188' or 'localhost')
    :return: Tuple of (host, port)
    """
    if ':' in address:
        host, port_str = address.rsplit(':', 1)
        try:
            port = int(port_str)
        except ValueError:
            port = 8188
    else:
        host = address
        port = 8188

    return host, port


def format_workflow_json(workflow: Mapping[str, object], indent: int = 2) -> str:
    """
    Format a workflow dictionary as pretty JSON.

    :param workflow: Workflow dictionary
    :param indent: Indentation level
    :return: Formatted JSON string
    """
    return json.dumps(workflow, indent=indent, ensure_ascii=False)


def get_node_output_type(node_class: str) -> list[str]:
    """
    Get the output types for common node classes.

    This is a helper for workflow building. Returns a list of output names.

    :param node_class: Node class type
    :return: List of output names
    """
    output_types: dict[str, list[str]] = {
        'CheckpointLoaderSimple': ['MODEL', 'CLIP', 'VAE'],
        'CLIPTextEncode': ['CONDITIONING'],
        'EmptyLatentImage': ['LATENT'],
        'KSampler': ['LATENT'],
        'VAEDecode': ['IMAGE'],
        'VAEEncode': ['LATENT'],
        'SaveImage': [],
        'PreviewImage': [],
        'LoadImage': ['IMAGE', 'MASK'],
        'LoraLoader': ['MODEL', 'CLIP'],
        'ControlNetLoader': ['CONTROL_NET'],
        'ControlNetApply': ['CONDITIONING'],
    }
    return output_types.get(node_class, [])


class ProgressTracker:
    """
    A simple progress tracker for monitoring workflow execution.
    """

    def __init__(
        self,
        callback: Callable[[ProgressTracker], None] | None = None,
    ) -> None:
        """
        Initialize the progress tracker.

        :param callback: Optional callback function for progress updates
        """
        self.callback: Callable[[ProgressTracker], None] | None = callback
        self.current_node: str | None = None
        self.current_step: int = 0
        self.total_steps: int = 0
        self.progress: float = 0.0

    def update(self, data: dict[str, object]) -> None:
        """
        Update progress from WebSocket message data.

        :param data: Progress data from WebSocket
        """
        if 'node' in data:
            node_value = data['node']
            if isinstance(node_value, str):
                self.current_node = node_value

        if 'value' in data and 'max' in data:
            value = data['value']
            max_value = data['max']
            if isinstance(value, int) and isinstance(max_value, int):
                self.current_step = value
                self.total_steps = max_value
                self.progress = (
                    self.current_step / self.total_steps if self.total_steps > 0 else 0.0
                )

        if self.callback:
            self.callback(self)

    @override
    def __str__(self) -> str:
        """Get progress as string."""
        if self.total_steps > 0:
            return f'{self.current_step}/{self.total_steps} ({self.progress:.1%})'
        return 'Waiting...'

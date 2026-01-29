"""
ComfyUI Client exceptions.
"""


class ComfyUIError(Exception):
    """Base exception for ComfyUI client errors."""

    pass


class ComfyUIConnectionError(ComfyUIError):
    """Raised when connection to ComfyUI server fails."""

    pass


class WorkflowError(ComfyUIError):
    """Raised when workflow execution fails."""

    pass


class ComfyUITimeoutError(ComfyUIError):
    """Raised when operation times out."""

    pass


class QueueError(ComfyUIError):
    """Raised when queue operation fails."""

    pass


class ImageError(ComfyUIError):
    """Raised when image retrieval fails."""

    pass

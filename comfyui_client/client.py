"""
ComfyUI Client - Main client class for interacting with ComfyUI API.
"""

from __future__ import annotations

import json
import uuid
import urllib.request
import urllib.parse
import urllib.error
from http.client import HTTPResponse
from types import TracebackType
from typing import Callable, cast
from io import BytesIO

from comfyui_client.exceptions import (
    ComfyUIConnectionError,
    QueueError,
    ImageError,
)
from comfyui_client.websocket_client import WebSocketClient


class ComfyUIClient:
    """
    A client for interacting with ComfyUI server.

    This client provides methods to:
    - Submit workflows for execution
    - Monitor execution progress via WebSocket
    - Retrieve generated images
    - Manage the execution queue

    Example:
        client = ComfyUIClient('127.0.0.1:8188')
        prompt_id = client.queue_prompt(workflow)
        images = client.wait_for_completion(prompt_id)
    """

    def __init__(
        self,
        server_address: str = '127.0.0.1:8188',
        client_id: str | None = None,
        use_ssl: bool = False,
    ) -> None:
        """
        Initialize the ComfyUI client.

        :param server_address: ComfyUI server address (host:port)
        :param client_id: Optional client ID for WebSocket connection
        :param use_ssl: Whether to use HTTPS/WSS
        """
        self.server_address: str = server_address
        self.client_id: str = client_id or str(uuid.uuid4())
        self.use_ssl: bool = use_ssl
        self._ws_client: WebSocketClient | None = None

        self._http_protocol: str = 'https' if use_ssl else 'http'
        self._ws_protocol: str = 'wss' if use_ssl else 'ws'

    @property
    def base_url(self) -> str:
        """Get the base HTTP URL for the server."""
        return f'{self._http_protocol}://{self.server_address}'

    @property
    def ws_url(self) -> str:
        """Get the WebSocket URL for the server."""
        return f'{self._ws_protocol}://{self.server_address}/ws?clientId={self.client_id}'

    def _make_request(
        self,
        endpoint: str,
        method: str = 'GET',
        data: dict[str, object] | None = None,
    ) -> object:
        """
        Make an HTTP request to the ComfyUI server.

        :param endpoint: API endpoint
        :param method: HTTP method
        :param data: Request data (for POST requests)
        :return: Response data
        """
        url = f'{self.base_url}{endpoint}'

        try:
            if method == 'POST' and data is not None:
                json_data = json.dumps(data).encode('utf-8')
                request = urllib.request.Request(
                    url,
                    data=json_data,
                    headers={'Content-Type': 'application/json'},
                    method=method,
                )
            else:
                request = urllib.request.Request(url, method=method)

            response: HTTPResponse = cast(HTTPResponse, urllib.request.urlopen(request))
            try:
                response_text: str = response.read().decode('utf-8')
                result: object = cast(object, json.loads(response_text))
                return result
            finally:
                response.close()

        except urllib.error.HTTPError as e:
            # ComfyUI returns 400 with JSON body for validation errors
            # Try to parse the error response
            try:
                error_body: str = e.read().decode('utf-8')
                error_data: object = cast(object, json.loads(error_body))
                return error_data
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ComfyUIConnectionError(f'HTTP Error {e.code}: {e.reason}')
        except urllib.error.URLError as e:
            raise ComfyUIConnectionError(f'Failed to connect to ComfyUI server: {e}')
        except json.JSONDecodeError as e:
            raise ComfyUIConnectionError(f'Failed to parse server response: {e}')

    def get_system_stats(self) -> dict[str, object]:
        """
        Get system statistics from ComfyUI server.

        :return: System stats including GPU info, memory usage, etc.
        """
        result = self._make_request('/system_stats')
        if isinstance(result, dict):
            return cast(dict[str, object], result)
        return {}

    def get_queue(self) -> dict[str, object]:
        """
        Get the current execution queue.

        :return: Queue information including pending and running prompts
        """
        result = self._make_request('/queue')
        if isinstance(result, dict):
            return cast(dict[str, object], result)
        return {}

    def get_history(self, prompt_id: str | None = None) -> dict[str, object]:
        """
        Get execution history.

        :param prompt_id: Optional prompt ID to get specific history
        :return: Execution history
        """
        if prompt_id:
            result = self._make_request(f'/history/{prompt_id}')
        else:
            result = self._make_request('/history')
        if isinstance(result, dict):
            return cast(dict[str, object], result)
        return {}

    def queue_prompt(self, workflow: dict[str, object]) -> str:
        """
        Submit a workflow to the execution queue.

        :param workflow: The workflow (prompt) to execute
        :return: The prompt ID for tracking execution
        """
        data: dict[str, object] = {
            'prompt': workflow,
            'client_id': self.client_id,
        }

        try:
            response = self._make_request('/prompt', method='POST', data=data)
            if not isinstance(response, dict):
                raise QueueError(f'Failed to queue prompt: {response}')

            typed_response: dict[str, object] = cast(dict[str, object], response)

            # Check for error response from ComfyUI
            if 'error' in typed_response:
                error_obj: object = typed_response.get('error')
                if isinstance(error_obj, dict):
                    error_dict: dict[str, object] = cast(dict[str, object], error_obj)
                    error_msg: object = error_dict.get('message', 'Unknown error')
                    raise QueueError(f'ComfyUI error: {error_msg}')
                raise QueueError(f'ComfyUI error: {error_obj}')

            if 'prompt_id' not in typed_response:
                raise QueueError(f'Failed to queue prompt: {response}')

            return str(typed_response['prompt_id'])
        except QueueError:
            raise
        except Exception as e:
            raise QueueError(f'Failed to queue prompt: {e}')

    def get_image(
        self,
        filename: str,
        subfolder: str = '',
        folder_type: str = 'output',
    ) -> bytes:
        """
        Retrieve an image from the server.

        :param filename: Image filename
        :param subfolder: Subfolder containing the image
        :param folder_type: Folder type ('output', 'input', 'temp')
        :return: Image data as bytes
        """
        params = urllib.parse.urlencode({
            'filename': filename,
            'subfolder': subfolder,
            'type': folder_type,
        })
        url = f'{self.base_url}/view?{params}'

        try:
            response: HTTPResponse = cast(HTTPResponse, urllib.request.urlopen(url))
            try:
                img_bytes: bytes = response.read()
                return img_bytes
            finally:
                response.close()
        except urllib.error.URLError as e:
            raise ImageError(f'Failed to retrieve image: {e}')

    def upload_image(
        self,
        image_data: bytes | BytesIO,
        filename: str,
        subfolder: str = '',
        folder_type: str = 'input',
        overwrite: bool = True,
    ) -> dict[str, object]:
        """
        Upload an image to the server.

        :param image_data: Image data as bytes or BytesIO
        :param filename: Target filename
        :param subfolder: Target subfolder
        :param folder_type: Folder type ('input', 'temp')
        :param overwrite: Whether to overwrite existing file
        :return: Upload response with filename info
        """
        if isinstance(image_data, BytesIO):
            image_bytes = image_data.read()
        else:
            image_bytes = image_data

        boundary = uuid.uuid4().hex
        content_type = f'multipart/form-data; boundary={boundary}'

        body: list[bytes] = []
        body.append(f'--{boundary}'.encode())
        body.append(
            f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode()
        )
        body.append(b'Content-Type: image/png')
        body.append(b'')
        body.append(image_bytes)

        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="subfolder"')
        body.append(b'')
        body.append(subfolder.encode())

        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="type"')
        body.append(b'')
        body.append(folder_type.encode())

        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="overwrite"')
        body.append(b'')
        body.append(str(overwrite).lower().encode())

        body.append(f'--{boundary}--'.encode())

        body_data = b'\r\n'.join(body)

        url = f'{self.base_url}/upload/image'
        request = urllib.request.Request(
            url,
            data=body_data,
            headers={'Content-Type': content_type},
            method='POST',
        )

        try:
            response: HTTPResponse = cast(HTTPResponse, urllib.request.urlopen(request))
            try:
                response_text: str = response.read().decode('utf-8')
                result: object = cast(object, json.loads(response_text))
                if isinstance(result, dict):
                    return cast(dict[str, object], result)
                return {}
            finally:
                response.close()
        except urllib.error.URLError as e:
            raise ImageError(f'Failed to upload image: {e}')

    def connect_websocket(self) -> WebSocketClient:
        """
        Connect to the WebSocket for receiving execution updates.

        :return: WebSocket client instance
        """
        if self._ws_client is None or not self._ws_client.is_connected:
            self._ws_client = WebSocketClient(self.ws_url)
            self._ws_client.connect()
        return self._ws_client

    def disconnect_websocket(self) -> None:
        """Disconnect the WebSocket connection."""
        if self._ws_client is not None:
            self._ws_client.disconnect()
            self._ws_client = None

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float | None = None,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ) -> list[dict[str, object]]:
        """
        Wait for a prompt to complete and return the generated images.

        :param prompt_id: The prompt ID to wait for
        :param timeout: Optional timeout in seconds
        :param on_progress: Optional callback for progress updates
        :return: List of generated image information
        """
        ws_client = self.connect_websocket()

        try:
            output_images: list[dict[str, object]] = []
            while True:
                message = ws_client.receive(timeout=timeout)

                if message is None:
                    continue

                msg_type: object = message.get('type')
                if msg_type == 'executing':
                    data: object = message.get('data')
                    if isinstance(data, dict):
                        typed_data: dict[str, object] = cast(dict[str, object], data)
                        if typed_data.get('prompt_id') == prompt_id:
                            if typed_data.get('node') is None:
                                # Execution completed
                                break

                if msg_type == 'progress' and on_progress:
                    progress_data: object = message.get('data')
                    if isinstance(progress_data, dict):
                        typed_progress: dict[str, object] = cast(dict[str, object], progress_data)
                        on_progress(typed_progress)

                if msg_type == 'executed':
                    exec_data: object = message.get('data')
                    if isinstance(exec_data, dict):
                        typed_exec: dict[str, object] = cast(dict[str, object], exec_data)
                        if typed_exec.get('prompt_id') == prompt_id:
                            output: object = typed_exec.get('output')
                            if isinstance(output, dict) and 'images' in output:
                                typed_output: dict[str, object] = cast(dict[str, object], output)
                                img_list: object = typed_output.get('images')
                                if isinstance(img_list, list):
                                    typed_list: list[object] = cast(list[object], img_list)
                                    for img_item in typed_list:
                                        if isinstance(img_item, dict):
                                            typed_img: dict[str, object] = cast(dict[str, object], img_item)
                                            output_images.append(typed_img)

            # Get complete history for the prompt
            history = self.get_history(prompt_id)
            if prompt_id in history:
                prompt_history: object = history[prompt_id]
                if isinstance(prompt_history, dict):
                    typed_hist_data: dict[str, object] = cast(dict[str, object], prompt_history)
                    outputs: object = typed_hist_data.get('outputs')
                    if isinstance(outputs, dict):
                        typed_outputs: dict[str, object] = cast(dict[str, object], outputs)
                        for node_val in typed_outputs.values():
                            if isinstance(node_val, dict) and 'images' in node_val:
                                typed_node: dict[str, object] = cast(dict[str, object], node_val)
                                node_images: object = typed_node.get('images')
                                if isinstance(node_images, list):
                                    typed_imgs: list[object] = cast(list[object], node_images)
                                    for hist_img in typed_imgs:
                                        if isinstance(hist_img, dict):
                                            typed_hist: dict[str, object] = cast(dict[str, object], hist_img)
                                            if typed_hist not in output_images:
                                                output_images.append(typed_hist)

            return output_images

        finally:
            pass  # Keep connection open for reuse

    def generate(
        self,
        workflow: dict[str, object],
        timeout: float | None = None,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ) -> list[bytes]:
        """
        Execute a workflow and return the generated images.

        This is a convenience method that combines queue_prompt,
        wait_for_completion, and get_image.

        :param workflow: The workflow to execute
        :param timeout: Optional timeout in seconds
        :param on_progress: Optional callback for progress updates
        :return: List of generated images as bytes
        """
        prompt_id = self.queue_prompt(workflow)
        image_info_list = self.wait_for_completion(
            prompt_id,
            timeout=timeout,
            on_progress=on_progress,
        )

        images: list[bytes] = []
        for img_info in image_info_list:
            filename: object | None = img_info.get('filename')
            subfolder = img_info.get('subfolder', '')
            folder_type = img_info.get('type', 'output')
            if isinstance(filename, str):
                img_data = self.get_image(
                    filename=filename,
                    subfolder=str(subfolder) if subfolder else '',
                    folder_type=str(folder_type) if folder_type else 'output',
                )
                images.append(img_data)

        return images

    def interrupt(self) -> dict[str, object]:
        """
        Interrupt the current execution.

        :return: Server response
        """
        result = self._make_request('/interrupt', method='POST', data={})
        if isinstance(result, dict):
            return cast(dict[str, object], result)
        return {}

    def clear_queue(self) -> dict[str, object]:
        """
        Clear all pending items from the queue.

        :return: Server response
        """
        result = self._make_request('/queue', method='POST', data={'clear': True})
        if isinstance(result, dict):
            return cast(dict[str, object], result)
        return {}

    def delete_queue_item(self, prompt_id: str) -> dict[str, object]:
        """
        Delete a specific item from the queue.

        :param prompt_id: The prompt ID to delete
        :return: Server response
        """
        result = self._make_request(
            '/queue',
            method='POST',
            data={'delete': [prompt_id]},
        )
        if isinstance(result, dict):
            return cast(dict[str, object], result)
        return {}

    def get_object_info(self, node_class: str | None = None) -> dict[str, object]:
        """
        Get information about available nodes.

        :param node_class: Optional specific node class to query
        :return: Node information
        """
        if node_class:
            result = self._make_request(f'/object_info/{node_class}')
        else:
            result = self._make_request('/object_info')
        if isinstance(result, dict):
            return cast(dict[str, object], result)
        return {}

    def get_embeddings(self) -> list[str]:
        """
        Get list of available embeddings.

        :return: List of embedding names
        """
        result = self._make_request('/embeddings')
        if isinstance(result, list):
            typed_result: list[object] = cast(list[object], result)
            return [str(item) for item in typed_result if isinstance(item, str)]
        return []

    def get_extensions(self) -> list[str]:
        """
        Get list of installed extensions.

        :return: List of extension paths
        """
        result = self._make_request('/extensions')
        if isinstance(result, list):
            typed_result: list[object] = cast(list[object], result)
            return [str(item) for item in typed_result if isinstance(item, str)]
        return []

    def __enter__(self) -> ComfyUIClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Context manager exit."""
        self.disconnect_websocket()
        return False

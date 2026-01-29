"""
WebSocket client for ComfyUI real-time communication.
"""

from __future__ import annotations

import json
import struct
import socket
import ssl
import hashlib
import base64
import os
from types import TracebackType
from typing import cast
from urllib.parse import urlparse

from comfyui_client.exceptions import ComfyUIConnectionError, ComfyUITimeoutError


class WebSocketClient:
    """
    A simple WebSocket client for ComfyUI.

    This client handles the WebSocket protocol for receiving
    execution updates from ComfyUI server.
    """

    def __init__(self, url: str) -> None:
        """
        Initialize the WebSocket client.

        :param url: WebSocket URL (ws:// or wss://)
        """
        self.url: str = url
        self._socket: socket.socket | ssl.SSLSocket | None = None
        self._is_connected: bool = False

        parsed = urlparse(url)
        self._host: str = parsed.hostname or 'localhost'
        self._port: int = parsed.port or (443 if parsed.scheme == 'wss' else 80)
        self._path: str = parsed.path or '/'
        if parsed.query:
            self._path += f'?{parsed.query}'
        self._use_ssl: bool = parsed.scheme == 'wss'

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_connected

    def connect(self) -> None:
        """
        Establish WebSocket connection.

        :raises ComfyUIConnectionError: If connection fails
        """
        try:
            raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw_socket.settimeout(30)

            if self._use_ssl:
                context = ssl.create_default_context()
                self._socket = context.wrap_socket(
                    raw_socket,
                    server_hostname=self._host,
                )
            else:
                self._socket = raw_socket

            self._socket.connect((self._host, self._port))
            self._handshake()
            self._is_connected = True

        except socket.error as e:
            self._is_connected = False
            raise ComfyUIConnectionError(f'Failed to connect WebSocket: {e}')

    def _handshake(self) -> None:
        """Perform WebSocket handshake."""
        if self._socket is None:
            raise ComfyUIConnectionError('Socket not initialized')

        key = base64.b64encode(os.urandom(16)).decode('utf-8')

        request = (
            f'GET {self._path} HTTP/1.1\r\n'
            f'Host: {self._host}:{self._port}\r\n'
            f'Upgrade: websocket\r\n'
            f'Connection: Upgrade\r\n'
            f'Sec-WebSocket-Key: {key}\r\n'
            f'Sec-WebSocket-Version: 13\r\n'
            f'\r\n'
        )

        self._socket.sendall(request.encode('utf-8'))

        response = b''
        while b'\r\n\r\n' not in response:
            chunk = self._socket.recv(1024)
            if not chunk:
                raise ComfyUIConnectionError('Connection closed during handshake')
            response += chunk

        if b'101' not in response:
            raise ComfyUIConnectionError(
                f'WebSocket handshake failed: {response.decode()}'
            )

        # Verify accept key
        expected_key = key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        expected_accept = base64.b64encode(
            hashlib.sha1(expected_key.encode()).digest()
        ).decode('utf-8')

        if expected_accept.encode() not in response:
            raise ComfyUIConnectionError('Invalid WebSocket accept key')

    def _recv_exact(self, size: int) -> bytes:
        """Receive exact number of bytes."""
        if self._socket is None:
            raise ComfyUIConnectionError('Socket not initialized')

        data = b''
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise ComfyUIConnectionError('Connection closed')
            data += chunk
        return data

    def receive(self, timeout: float | None = None) -> dict[str, object] | None:
        """
        Receive a message from WebSocket.

        :param timeout: Optional timeout in seconds
        :return: Parsed JSON message or None for binary/ping frames
        :raises ComfyUITimeoutError: If timeout is reached
        :raises ComfyUIConnectionError: If connection is lost
        """
        if not self._is_connected or self._socket is None:
            raise ComfyUIConnectionError('WebSocket not connected')

        try:
            if timeout is not None:
                self._socket.settimeout(timeout)

            # Read frame header
            header = self._recv_exact(2)
            opcode = header[0] & 0x0F
            masked = (header[1] & 0x80) != 0
            payload_len = header[1] & 0x7F

            # Extended payload length
            if payload_len == 126:
                ext_len = self._recv_exact(2)
                unpack_tuple: tuple[int] = struct.unpack('>H', ext_len)
                payload_len = unpack_tuple[0]
            elif payload_len == 127:
                ext_len = self._recv_exact(8)
                unpack_tuple_q: tuple[int] = struct.unpack('>Q', ext_len)
                payload_len = unpack_tuple_q[0]

            # Masking key
            mask_key: bytes | None = None
            if masked:
                mask_key = self._recv_exact(4)

            # Payload
            payload = self._recv_exact(payload_len)

            # Unmask if needed
            if mask_key is not None:
                payload = bytes(
                    payload[i] ^ mask_key[i % 4] for i in range(len(payload))
                )

            # Handle opcodes
            if opcode == 0x08:  # Close
                self._is_connected = False
                raise ComfyUIConnectionError('WebSocket closed by server')
            elif opcode == 0x09:  # Ping
                self._send_pong(payload)
                return None
            elif opcode == 0x0A:  # Pong
                return None
            elif opcode == 0x02:  # Binary
                # Binary frame - usually preview images
                return None
            elif opcode == 0x01:  # Text
                try:
                    parsed: object = cast(object, json.loads(payload.decode('utf-8')))
                    if isinstance(parsed, dict):
                        result: dict[str, object] = cast(dict[str, object], parsed)
                        return result
                    return None
                except json.JSONDecodeError:
                    return None

            return None

        except socket.timeout as e:
            raise ComfyUITimeoutError(f'WebSocket receive timeout: {e}')
        except socket.error as e:
            self._is_connected = False
            raise ComfyUIConnectionError(f'WebSocket error: {e}')

    def _send_pong(self, payload: bytes) -> None:
        """Send pong frame in response to ping."""
        if self._socket is None:
            return
        frame = bytes([0x8A, len(payload)]) + payload
        self._socket.sendall(frame)

    def send(self, message: dict[str, object]) -> None:
        """
        Send a JSON message through WebSocket.

        :param message: Message to send
        """
        if not self._is_connected or self._socket is None:
            raise ComfyUIConnectionError('WebSocket not connected')

        payload = json.dumps(message).encode('utf-8')

        # Build frame with masking
        mask_key = os.urandom(4)
        masked_payload = bytes(
            payload[i] ^ mask_key[i % 4] for i in range(len(payload))
        )

        frame = bytearray()
        frame.append(0x81)  # FIN + text opcode

        if len(payload) < 126:
            frame.append(0x80 | len(payload))
        elif len(payload) < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack('>H', len(payload)))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack('>Q', len(payload)))

        frame.extend(mask_key)
        frame.extend(masked_payload)

        try:
            self._socket.sendall(frame)
        except socket.error as e:
            self._is_connected = False
            raise ComfyUIConnectionError(f'Failed to send message: {e}')

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._socket is not None:
            try:
                # Send close frame
                self._socket.sendall(bytes([0x88, 0x00]))
            except socket.error:
                pass

            try:
                self._socket.close()
            except socket.error:
                pass

            self._socket = None
            self._is_connected = False

    def __enter__(self) -> WebSocketClient:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Context manager exit."""
        self.disconnect()
        return False

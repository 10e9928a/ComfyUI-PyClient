"""
Tests for utility functions.
"""

import tempfile
from pathlib import Path

from comfyui_client.utils import (
    random_seed,
    save_image,
    save_images,
    load_image,
    parse_server_address,
    format_workflow_json,
    get_node_output_type,
    ProgressTracker,
)


class TestRandomSeed:
    """Tests for random_seed function."""

    def test_returns_int(self) -> None:
        """Test that random_seed returns an integer."""
        seed = random_seed()
        assert isinstance(seed, int)

    def test_within_range(self) -> None:
        """Test seed is within valid range."""
        for _ in range(100):
            seed = random_seed()
            assert 0 <= seed < 2**32

    def test_varies(self) -> None:
        """Test that seeds vary (not all the same)."""
        seeds = [random_seed() for _ in range(10)]
        assert len(set(seeds)) > 1


class TestImageFunctions:
    """Tests for image save/load functions."""

    def test_save_and_load_image(self) -> None:
        """Test saving and loading an image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.png'
            test_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100

            _ = save_image(test_data, filepath)
            assert filepath.exists()

            loaded = load_image(filepath)
            assert loaded == test_data

    def test_save_images_multiple(self) -> None:
        """Test saving multiple images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [
                b'\x89PNG\r\n\x1a\n' + b'\x00' * 50,
                b'\x89PNG\r\n\x1a\n' + b'\x01' * 50,
                b'\x89PNG\r\n\x1a\n' + b'\x02' * 50,
            ]

            paths = save_images(images, tmpdir, prefix='test')
            assert len(paths) == 3
            assert all(p.exists() for p in paths)
            assert all('test' in p.name for p in paths)

    def test_save_image_creates_directory(self) -> None:
        """Test that save_image creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'subdir' / 'deep' / 'test.png'
            test_data = b'\x89PNG\r\n\x1a\n'

            _ = save_image(test_data, filepath)
            assert filepath.exists()


class TestParseServerAddress:
    """Tests for parse_server_address function."""

    def test_with_port(self) -> None:
        """Test parsing address with port."""
        host, port = parse_server_address('127.0.0.1:8188')
        assert host == '127.0.0.1'
        assert port == 8188

    def test_without_port(self) -> None:
        """Test parsing address without port."""
        host, port = parse_server_address('localhost')
        assert host == 'localhost'
        assert port == 8188

    def test_custom_port(self) -> None:
        """Test parsing with custom port."""
        host, port = parse_server_address('example.com:9999')
        assert host == 'example.com'
        assert port == 9999


class TestFormatWorkflowJson:
    """Tests for format_workflow_json function."""

    def test_formats_workflow(self) -> None:
        """Test formatting workflow to JSON."""
        workflow: dict[str, object] = {'1': {'class_type': 'Test', 'inputs': {}}}
        result = format_workflow_json(workflow)
        assert isinstance(result, str)
        assert '"class_type"' in result
        assert '"Test"' in result


class TestGetNodeOutputType:
    """Tests for get_node_output_type function."""

    def test_known_node(self) -> None:
        """Test getting outputs for known node."""
        outputs = get_node_output_type('CheckpointLoaderSimple')
        assert outputs == ['MODEL', 'CLIP', 'VAE']

    def test_ksampler(self) -> None:
        """Test KSampler outputs."""
        outputs = get_node_output_type('KSampler')
        assert outputs == ['LATENT']

    def test_unknown_node(self) -> None:
        """Test unknown node returns empty list."""
        outputs = get_node_output_type('UnknownNode')
        assert outputs == []


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_init(self) -> None:
        """Test initialization."""
        tracker = ProgressTracker()
        assert tracker.current_step == 0
        assert tracker.total_steps == 0
        assert tracker.progress == 0.0

    def test_update(self) -> None:
        """Test updating progress."""
        tracker = ProgressTracker()
        tracker.update({'value': 5, 'max': 20})
        assert tracker.current_step == 5
        assert tracker.total_steps == 20
        assert tracker.progress == 0.25

    def test_update_with_node(self) -> None:
        """Test updating with node info."""
        tracker = ProgressTracker()
        tracker.update({'node': 'test_node'})
        assert tracker.current_node == 'test_node'

    def test_callback(self) -> None:
        """Test callback is called on update."""
        called: list[float] = []

        def callback(t: ProgressTracker) -> None:
            called.append(t.progress)

        tracker = ProgressTracker(callback=callback)
        tracker.update({'value': 1, 'max': 2})
        assert len(called) == 1
        assert called[0] == 0.5

    def test_str(self) -> None:
        """Test string representation."""
        tracker = ProgressTracker()
        tracker.update({'value': 10, 'max': 20})
        result = str(tracker)
        assert '10/20' in result
        assert '50' in result

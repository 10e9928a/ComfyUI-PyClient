# ComfyUI Client

A Python client library for interacting with ComfyUI API, providing a simple interface to submit workflows, monitor progress, and retrieve generated images/videos.

## Features

- Connect to ComfyUI server via HTTP/WebSocket
- Submit and execute workflows
- Real-time progress monitoring
- Retrieve generated images and videos
- Workflow management utilities (load, modify, save)
- Programmatic workflow creation with `WorkflowBuilder`
- Support for image upload and img2img workflows
- Support for inpainting with mask images

## Requirements

- Python >= 3.12
- ComfyUI server running locally or remotely

## Installation

```bash
pip install ComfyUI-PyClient
```

Or install from source:

```bash
git clone https://github.com/10e9928a/ComfyUI-PyClient.git
cd ComfyUI-PyClient
pip install -e .
```

### Optional Dependencies

For image processing features (e.g., `create_masked_image`):

```bash
pip install 'ComfyUI-PyClient[image]'
```

Install all optional dependencies:

```bash
pip install 'ComfyUI-PyClient[all]'
```

## Quick Start

### Basic Usage

```python
from comfyui_client import ComfyUIClient, WorkflowBuilder
from comfyui_client.utils import save_images, random_seed

# Connect to ComfyUI server
with ComfyUIClient('127.0.0.1:8188') as client:
    # Create a simple text-to-image workflow
    workflow = WorkflowBuilder.create_simple_txt2img(
        model_name='v1-5-pruned-emaonly-fp16.safetensors',
        positive_prompt='a beautiful sunset over mountains, highly detailed',
        negative_prompt='blurry, low quality',
        width=512,
        height=512,
        steps=20,
        cfg=7.0,
        seed=random_seed(),
    )

    # Generate images
    images = client.generate(workflow.to_dict())

    # Save generated images
    save_images(images, './output', prefix='generated')
```

### Load Workflow from File

```python
from comfyui_client import ComfyUIClient, Workflow
from comfyui_client.utils import save_images, random_seed

with ComfyUIClient('127.0.0.1:8188') as client:
    # Load workflow from JSON file
    workflow = Workflow.from_file('my_workflow.json')

    # Modify parameters
    workflow.set_seed(random_seed())
    workflow.set_prompt(positive='new prompt text')
    workflow.set_steps(25)
    workflow.set_cfg(8.0)

    # Execute and save
    images = client.generate(workflow.to_dict())
    save_images(images, './output')
```

### Progress Monitoring

```python
def on_progress(data: dict) -> None:
    value = data.get('value', 0)
    max_val = data.get('max', 1)
    print(f'Progress: {value}/{max_val} ({value/max_val*100:.1f}%)')

images = client.generate(
    workflow.to_dict(),
    on_progress=on_progress,
)
```

### Upload Image for img2img

```python
from comfyui_client import ComfyUIClient
from comfyui_client.utils import load_image

with ComfyUIClient('127.0.0.1:8188') as client:
    # Upload image
    image_data = load_image('input.png')
    result = client.upload_image(image_data, 'input.png')

    # Use uploaded image in workflow
    # ...
```

### Inpainting with Mask

```python
from comfyui_client import ComfyUIClient
from comfyui_client.utils import create_masked_image

# Create RGBA image with mask as alpha channel
masked_image = create_masked_image(
    image='source.png',
    mask='mask.png',
    invert_mask=True,
)

with ComfyUIClient('127.0.0.1:8188') as client:
    result = client.upload_image(masked_image, 'masked_input.png')
    # Use in inpainting workflow...
```

## API Reference

### ComfyUIClient

Main client class for interacting with ComfyUI server.

```python
client = ComfyUIClient(
    server_address='127.0.0.1:8188',  # Server address
    client_id=None,                    # Optional client ID
    use_ssl=False,                     # Use HTTPS/WSS
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `queue_prompt(workflow)` | Submit workflow to execution queue |
| `wait_for_completion(prompt_id)` | Wait for execution to complete |
| `generate(workflow)` | Execute workflow and return images |
| `get_image(filename)` | Retrieve image from server |
| `upload_image(data, filename)` | Upload image to server |
| `get_system_stats()` | Get server system statistics |
| `get_queue()` | Get current execution queue |
| `get_history(prompt_id)` | Get execution history |
| `interrupt()` | Interrupt current execution |
| `clear_queue()` | Clear all pending queue items |
| `get_object_info()` | Get available node information |

### Workflow

Class for loading and modifying workflows.

```python
# Load from file
workflow = Workflow.from_file('workflow.json')

# Load from JSON string
workflow = Workflow.from_json(json_str)

# Modify parameters
workflow.set_seed(12345)
workflow.set_prompt(positive='prompt', negative='neg')
workflow.set_image_size(width=768, height=768)
workflow.set_steps(30)
workflow.set_cfg(7.5)
workflow.set_sampler('dpmpp_2m', scheduler='karras')
workflow.set_model('model.safetensors')
workflow.set_batch_size(2)

# Access nodes
node = workflow.get_node('5')
workflow.set_node_input('5', 'seed', 12345)

# Find nodes by type
samplers = workflow.find_nodes_by_class('KSampler')

# Export
workflow.to_dict()
workflow.to_json()
workflow.save('output.json')
```

### WorkflowBuilder

Builder class for creating workflows programmatically.

```python
builder = WorkflowBuilder()

# Add nodes
loader_id = builder.add_checkpoint_loader('model.safetensors')
positive_id = builder.add_clip_text_encode('prompt', (loader_id, 1))
negative_id = builder.add_clip_text_encode('', (loader_id, 1))
latent_id = builder.add_empty_latent(512, 512)
sampler_id = builder.add_ksampler(
    model_ref=(loader_id, 0),
    positive_ref=(positive_id, 0),
    negative_ref=(negative_id, 0),
    latent_ref=(latent_id, 0),
)
decode_id = builder.add_vae_decode((sampler_id, 0), (loader_id, 2))
builder.add_save_image((decode_id, 0))

workflow = builder.build()
```

Or use the convenience method:

```python
workflow = WorkflowBuilder.create_simple_txt2img(
    model_name='model.safetensors',
    positive_prompt='a cat',
    negative_prompt='blurry',
)
```

### Utility Functions

```python
from comfyui_client.utils import (
    save_image,          # Save single image
    save_images,         # Save multiple images
    save_video,          # Save single video
    save_videos,         # Save multiple videos
    load_image,          # Load image as bytes
    random_seed,         # Generate random seed
    create_masked_image, # Create RGBA with mask
)
```

## Exceptions

```python
from comfyui_client import (
    ComfyUIError,          # Base exception
    ComfyUIConnectionError, # Connection failed
    ComfyUITimeoutError,   # Operation timeout
    WorkflowError,         # Workflow error
    QueueError,            # Queue operation error
    ImageError,            # Image operation error
)
```

## Examples

See the `examples/` directory for more detailed examples:

- `basic_usage.py` - Basic text-to-image generation
- `workflow_from_file.py` - Load and execute workflow from JSON
- `modify_node_params.py` - Modify workflow parameters
- `upload_and_img2img.py` - Image-to-image generation
- `inpainting_with_mask.py` - Inpainting with mask
- `video_generation.py` - Video generation workflows
- `batch_generation.py` - Batch image generation

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run linter:

```bash
ruff check .
```

Format code:

```bash
black .
isort .
```

## License

MIT License

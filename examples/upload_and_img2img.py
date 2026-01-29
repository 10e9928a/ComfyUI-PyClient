#!/usr/bin/env python3
"""
Example: Image-to-image generation with uploaded image.

This example demonstrates how to:
1. Upload an image to ComfyUI server
2. Create an img2img workflow
3. Generate variations of the uploaded image
"""

from pathlib import Path

from comfyui_client import ComfyUIClient
from comfyui_client.utils import load_image, save_images, random_seed


def create_img2img_workflow(
    model_name: str,
    image_name: str,
    positive_prompt: str,
    negative_prompt: str = '',
    denoise: float = .7,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = 0,
) -> dict[str, object]:
    """Create an image-to-image workflow."""
    return {
        '1': {
            'class_type': 'CheckpointLoaderSimple',
            'inputs': {
                'ckpt_name': model_name,
            },
        },
        '2': {
            'class_type': 'LoadImage',
            'inputs': {
                'image': image_name,
            },
        },
        '3': {
            'class_type': 'VAEEncode',
            'inputs': {
                'pixels': ['2', 0],
                'vae': ['1', 2],
            },
        },
        '4': {
            'class_type': 'CLIPTextEncode',
            'inputs': {
                'text': positive_prompt,
                'clip': ['1', 1],
            },
        },
        '5': {
            'class_type': 'CLIPTextEncode',
            'inputs': {
                'text': negative_prompt,
                'clip': ['1', 1],
            },
        },
        '6': {
            'class_type': 'KSampler',
            'inputs': {
                'seed': seed,
                'steps': steps,
                'cfg': cfg,
                'sampler_name': 'euler',
                'scheduler': 'normal',
                'denoise': denoise,
                'model': ['1', 0],
                'positive': ['4', 0],
                'negative': ['5', 0],
                'latent_image': ['3', 0],
            },
        },
        '7': {
            'class_type': 'VAEDecode',
            'inputs': {
                'samples': ['6', 0],
                'vae': ['1', 2],
            },
        },
        '8': {
            'class_type': 'SaveImage',
            'inputs': {
                'filename_prefix': 'img2img',
                'images': ['7', 0],
            },
        },
    }


def main() -> None:
    """Run img2img example."""
    # Path to input image
    input_image_path = Path('./examples/input/my_example.png')

    if not input_image_path.exists():
        print(f'Input image not found: {input_image_path}')
        print('Please provide an input image')
        return

    with ComfyUIClient('127.0.0.1:8088', use_ssl=False) as client:
        # Upload the image
        print('Uploading image...')
        image_data = load_image(input_image_path)
        upload_result = client.upload_image(
            image_data,
            filename=input_image_path.name,
        )
        print(f'Upload result: {upload_result}')
        uploaded_name = upload_result.get('name')
        if not isinstance(uploaded_name, str):
            uploaded_name = input_image_path.name
        print(f'Uploaded as: {uploaded_name}')

        # Create img2img workflow
        workflow = create_img2img_workflow(
            model_name='v1-5-pruned-emaonly-fp16.safetensors',
            image_name=uploaded_name,
            positive_prompt='oil painting style, artistic, masterpiece',
            negative_prompt='blurry, low quality',
            denoise=0.65,
            steps=25,
            cfg=7.0,
            seed=random_seed(),
        )

        # Generate
        print('Generating img2img...')
        images = client.generate(workflow)

        if images:
            saved = save_images(images, './output', prefix='img2img')
            print(f'Saved: {saved}')
        else:
            print('No images generated')


if __name__ == '__main__':
    main()

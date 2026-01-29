#!/usr/bin/env python3
"""
Basic usage example for ComfyUI Client.

This example demonstrates how to:
1. Connect to a ComfyUI server
2. Create a simple text-to-image workflow
3. Execute the workflow and save the generated images
"""

from typing import cast

from comfyui_client import ComfyUIClient, WorkflowBuilder
from comfyui_client.utils import save_images, random_seed


def main() -> None:
    """Run basic usage example."""
    # Create a client connected to local ComfyUI server
    server_address = '127.0.0.1:8088'

    # 尝试使用 SSL 连接
    with ComfyUIClient(server_address, use_ssl=False) as client:
        # Check server connection
        try:
            stats = client.get_system_stats()
            print('Connected to ComfyUI server')
            devices: object = stats.get('devices')
            if isinstance(devices, list) and devices:
                typed_devices: list[object] = cast(list[object], devices)
                first_dev: object = typed_devices[0]
                if isinstance(first_dev, dict):
                    dev_dict: dict[str, object] = cast(dict[str, object], first_dev)
                    print(f"GPU: {dev_dict.get('name', 'Unknown')}")
        except Exception as e:
            print(f'Failed to connect: {e}')
            return

        # Create a simple text-to-image workflow
        workflow = WorkflowBuilder.create_simple_txt2img(
            model_name='v1-5-pruned-emaonly-fp16.safetensors',
            positive_prompt='a beautiful sunset over mountains, highly detailed, 8k',
            negative_prompt='blurry, low quality, watermark',
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            seed=random_seed(),
            sampler='euler',
            scheduler='normal',
        )

        print('Generating image...')

        # Progress callback
        def on_progress(data: dict[str, object]) -> None:
            value = data.get('value')
            max_val = data.get('max')
            if isinstance(value, (int, float)) and isinstance(max_val, (int, float)):
                progress = value / max_val * 100
                print(f'Progress: {progress:.1f}%')

        # Generate images
        images = client.generate(
            workflow.to_dict(),
            on_progress=on_progress,
        )

        # Save generated images
        if images:
            saved_paths = save_images(images, './output', prefix='generated')
            print(f'Saved {len(saved_paths)} images:')
            for path in saved_paths:
                print(f'  - {path}')
        else:
            print('No images generated')


if __name__ == '__main__':
    main()

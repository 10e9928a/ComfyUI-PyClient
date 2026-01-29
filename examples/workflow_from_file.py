#!/usr/bin/env python3
"""
Example: Load and modify a workflow from file.

This example shows how to:
1. Load an existing workflow from a JSON file
2. Modify workflow parameters
3. Execute and get results
"""

from pathlib import Path

from comfyui_client import ComfyUIClient, Workflow
from comfyui_client.utils import save_images, random_seed


def main() -> None:
    """Run workflow from file example."""
    # Path to your workflow JSON file
    workflow_path = Path('./examples/input/my_image_workflow.json')

    if not workflow_path.exists():
        print(f'Workflow file not found: {workflow_path}')
        print('Please export a workflow from ComfyUI (Save as API format)')
        return

    # Load workflow from file
    workflow = Workflow.from_file(workflow_path)
    print(f'Loaded workflow with {len(workflow)} nodes')

    # Modify workflow parameters (chain method calls)
    _ = (
        workflow
        .set_seed(random_seed())
        .set_prompt(
            positive='a cyberpunk city at night, neon lights, rain',
            negative='blurry, low quality',
        )
        .set_steps(25)
        .set_cfg(7.5)
        .set_image_size(768, 512)
    )

    # Connect and execute
    with ComfyUIClient('127.0.0.1:8088', use_ssl=False) as client:
        print('Executing workflow...')

        images = client.generate(workflow.to_dict())

        if images:
            saved = save_images(images, './output', prefix='from_file')
            print(f'Saved {len(saved)} images')
        else:
            print('No images generated')


if __name__ == '__main__':
    main()

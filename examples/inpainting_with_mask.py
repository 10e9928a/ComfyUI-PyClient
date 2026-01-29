#!/usr/bin/env python3
"""
Example: Inpainting with mask from workflow file.

This example shows how to:
1. Load an existing inpainting workflow from a JSON file
2. Create masked image from source image and mask using create_masked_image()
3. Upload image with mask (alpha channel) to server
4. Modify workflow parameters (image, prompt, etc.)
5. Execute and get results

Note: The mask is obtained from LoadImage node's second output (index 1).
The uploaded image should have an alpha channel as the mask.
"""

from pathlib import Path

from comfyui_client import ComfyUIClient, Workflow
from comfyui_client.utils import  save_images, random_seed, create_masked_image


def main() -> None:
    """Run inpainting with mask example."""
    # Path to your workflow JSON file
    workflow_path = Path('./examples/input/my_inpainting_workflow.json')

    if not workflow_path.exists():
        print(f'Workflow file not found: {workflow_path}')
        print('Please export an inpainting workflow from ComfyUI (Save as API format)')
        return

    # Load workflow from file
    workflow = Workflow.from_file(workflow_path)
    print(f'Loaded workflow with {len(workflow)} nodes')

    # Input image and mask paths
    image_path = Path('./examples/input/my_example.png')
    mask_path = Path('./examples/input/my_mask.png')

    # Check input files
    if not image_path.exists():
        print(f'Source image not found: {image_path}')
        return

    if not mask_path.exists():
        print(f'Mask image not found: {mask_path}')
        return

    # Create masked image (merge image and mask into RGBA)
    print('Creating masked image...')
    masked_image_path = Path('./output/masked_input.png')
    masked_image_data = create_masked_image(
        image=image_path,
        mask=mask_path,
        output=masked_image_path,
        invert_mask=True,  # White in mask = area to repaint
    )
    print(f'Masked image saved to: {masked_image_path}')



    # Connect and execute
    server = '127.0.0.1:8088'

    with ComfyUIClient(server, use_ssl=False) as client:
        print('Connected to ComfyUI server')

        # Upload masked image
        print('Uploading masked image...')
        result = client.upload_image(masked_image_data, 'inpaint_input.png', overwrite=True)
        uploaded_image = str(result.get('name', 'inpaint_input.png'))
        print(f'Uploaded image: {uploaded_image}')

        # Find LoadImage node and update it
        # LoadImage outputs: [IMAGE (index 0), MASK (index 1)]
        load_image_nodes = workflow.find_nodes_by_class('LoadImage')
        if load_image_nodes:
            _ = workflow.set_node_input(load_image_nodes[0][0], 'image', uploaded_image)

        # Modify workflow parameters
        _ = workflow.set_seed(random_seed())

        # Update prompt (find CLIPTextEncode node)
        clip_nodes = workflow.find_nodes_by_class('CLIPTextEncode')
        if clip_nodes:
            _ = workflow.set_node_input(
                clip_nodes[0][0],
                'text',
                'red eyes',
            )

        print('Executing inpainting workflow...')
        images = client.generate(workflow.to_dict())

        if images:
            saved = save_images(images, './output', prefix='inpaint')
            print(f'Saved {len(saved)} images:')
            for path in saved:
                print(f'  - {path}')
        else:
            print('No images generated')


if __name__ == '__main__':
    main()

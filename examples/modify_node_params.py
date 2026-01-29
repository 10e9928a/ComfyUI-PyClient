#!/usr/bin/env python3
"""
Example: Modify specific node parameters in a workflow.

This example shows how to:
1. Load a workflow from file
2. List all nodes and their types
3. Modify specific node parameters by node ID
4. Find nodes by class type and modify them
"""

from pathlib import Path

from comfyui_client import ComfyUIClient, Workflow
from comfyui_client.utils import save_images, random_seed


def main() -> None:
    """Run modify node params example."""
    workflow_path = Path('./examples/input/my_workflow.json')

    if not workflow_path.exists():
        print(f'Workflow file not found: {workflow_path}')
        print('Please export a workflow from ComfyUI (Save as API format)')
        return

    # Load workflow
    workflow = Workflow.from_file(workflow_path)
    print(f'Loaded workflow with {len(workflow)} nodes')

    # List all nodes
    print('\n--- Node List ---')
    for node_id in workflow.node_ids:
        node = workflow.get_node(node_id)
        if node:
            class_type = node.get('class_type', 'Unknown')
            print(f'  Node {node_id}: {class_type}')

    # Modify specific node by ID
    # Example: Modify KSampler (node 3) parameters
    print('\n--- Modify Specific Node ---')
    _ = workflow.set_node_input('3', 'seed', random_seed())
    _ = workflow.set_node_input('3', 'steps', 30)
    _ = workflow.set_node_input('3', 'cfg', 8.5)
    _ = workflow.set_node_input('3', 'sampler_name', 'dpmpp_2m')
    _ = workflow.set_node_input('3', 'scheduler', 'karras')
    print('  Modified KSampler (node 3): steps=30, cfg=8.5, sampler=dpmpp_2m')

    # Modify EmptyLatentImage (node 5)
    _ = workflow.set_node_input('5', 'width', 768)
    _ = workflow.set_node_input('5', 'height', 512)
    print('  Modified EmptyLatentImage (node 5): 768x512')

    # Modify positive prompt (node 6)
    _ = workflow.set_node_input('6', 'text', 'a beautiful sunset over ocean, golden hour, 8k')
    print('  Modified positive prompt (node 6)')

    # Modify negative prompt (node 7)
    _ = workflow.set_node_input('7', 'text', 'blurry, low quality, watermark')
    print('  Modified negative prompt (node 7)')

    # Modify SaveImage prefix (node 9)
    _ = workflow.set_node_input('9', 'filename_prefix', 'custom_output')
    print('  Modified SaveImage prefix (node 9)')

    # Verify modifications
    print('\n--- Verify Changes ---')
    print(f"  steps: {workflow.get_node_input('3', 'steps')}")
    print(f"  cfg: {workflow.get_node_input('3', 'cfg')}")
    print(f"  size: {workflow.get_node_input('5', 'width')}x{workflow.get_node_input('5', 'height')}")

    # Find and modify nodes by class type
    print('\n--- Find Nodes by Class ---')
    samplers = workflow.find_nodes_by_class('KSampler')
    for node_id, _ in samplers:
        print(f'  Found KSampler: node {node_id}')

    clip_nodes = workflow.find_nodes_by_class('CLIPTextEncode')
    print(f'  Found {len(clip_nodes)} CLIPTextEncode nodes')

    # Execute workflow
    server = '127.0.0.1:8088'
    with ComfyUIClient(server, use_ssl=False) as client:
        print('\nExecuting workflow...')
        images = client.generate(workflow.to_dict())

        if images:
            saved = save_images(images, './output', prefix='modified_node')
            print(f'Saved {len(saved)} images')
        else:
            print('No images generated')


if __name__ == '__main__':
    main()

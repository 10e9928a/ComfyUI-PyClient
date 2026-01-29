#!/usr/bin/env python3
"""
Example: Video generation workflow.

This example shows how to:
1. Load a video generation workflow from file
2. Modify video-related parameters
3. Execute the workflow and save output videos
"""

from pathlib import Path

from comfyui_client import ComfyUIClient, Workflow
from comfyui_client.utils import save_medias, random_seed


def main() -> None:
    """Run video generation example."""
    # Path to your video generation workflow (exported from ComfyUI in API format)
    workflow_path = Path('./examples/input/my_video_workflow.json')

    if not workflow_path.exists():
        print(f'Workflow file not found: {workflow_path}')
        print('Please export a video generation workflow from ComfyUI (Save as API format)')
        print('\nCommon video generation nodes:')
        print('  - AnimateDiff: For animated image generation')
        print('  - SVD (Stable Video Diffusion): For image-to-video')
        print('  - CogVideoX: For text-to-video')
        print('  - Mochi: For video generation')
        return

    # Load workflow
    workflow = Workflow.from_file(workflow_path)
    print(f'Loaded workflow with {len(workflow)} nodes')

    # List all nodes to understand workflow structure
    print('\n--- Node List ---')
    for node_id in workflow.node_ids:
        node = workflow.get_node(node_id)
        if node:
            class_type = node.get('class_type', 'Unknown')
            print(f'  Node {node_id}: {class_type}')

    # Modify video generation parameters
    # Note: Actual node IDs and parameters depend on your workflow
    print('\n--- Modify Video Parameters ---')

    # Example: Modify KSampler for video
    samplers = workflow.find_nodes_by_class('KSampler')
    for node_id, _ in samplers:
        _ = workflow.set_node_input(node_id, 'seed', random_seed())
        _ = workflow.set_node_input(node_id, 'steps', 20)
        _ = workflow.set_node_input(node_id, 'cfg', 7.0)
        print(f'  Modified KSampler (node {node_id})')

    # Example: Modify AnimateDiff parameters (if exists)
    animatediff_nodes = workflow.find_nodes_by_class('ADE_AnimateDiffLoaderWithContext')
    for node_id, _ in animatediff_nodes:
        print(f'  Found AnimateDiff loader: node {node_id}')

    # Example: Modify SVD parameters (if exists)
    svd_nodes = workflow.find_nodes_by_class('SVD_img2vid_Conditioning')
    for node_id, _ in svd_nodes:
        _ = workflow.set_node_input(node_id, 'video_frames', 25)
        _ = workflow.set_node_input(node_id, 'fps', 8)
        print(f'  Modified SVD conditioning (node {node_id}): 25 frames, 8 fps')

    # Example: Modify video combine/save node
    video_combine_nodes = workflow.find_nodes_by_class('VHS_VideoCombine')
    for node_id, _ in video_combine_nodes:
        _ = workflow.set_node_input(node_id, 'frame_rate', 8)
        _ = workflow.set_node_input(node_id, 'filename_prefix', 'generated_video')
        print(f'  Modified VideoCombine (node {node_id})')

    # Modify prompt if exists
    clip_nodes = workflow.find_nodes_by_class('CLIPTextEncode')
    if clip_nodes:
        # Usually first CLIP node is positive prompt
        node_id = clip_nodes[0][0]
        _ = workflow.set_node_input(
            node_id,
            'text',
            'a cat walking on the beach, high quality, 4k video'
        )
        print(f'  Modified prompt (node {node_id})')

    # Execute workflow
    server = '127.0.0.1:8088'
    with ComfyUIClient(server, use_ssl=False) as client:
        print('\nExecuting video workflow...')
        print('(This may take a while depending on video length)')

        # Generate - returns list of output data (images or video bytes)
        outputs = client.generate(workflow.to_dict())

        if outputs:
            saved = save_medias(outputs, './output', prefix='video', extension='mp4')
            print(f'Saved {len(saved)} files:')
            for path in saved:
                print(f'  - {path}')
        else:
            print('No outputs generated')


if __name__ == '__main__':
    main()

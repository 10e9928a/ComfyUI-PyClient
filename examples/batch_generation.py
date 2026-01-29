#!/usr/bin/env python3
"""
Example: Batch image generation with different prompts.

This example demonstrates how to:
1. Generate multiple images with different prompts
2. Use progress tracking
3. Handle errors gracefully
"""

from comfyui_client import ComfyUIClient, WorkflowBuilder
from comfyui_client.utils import save_images, random_seed, ProgressTracker


def main() -> None:
    """Run batch generation example."""
    prompts = [
        ('sunset_beach', 'a peaceful sunset on a tropical beach, palm trees, waves'),
        ('mountain_snow', 'snowy mountain peaks at sunrise, crisp air, alpine'),
        ('city_night', 'bustling city street at night, neon signs, rain reflections'),
        ('forest_path', 'mystical forest path with sunlight filtering through trees'),
    ]

    with ComfyUIClient('127.0.0.1:8088', use_ssl=False) as client:
        for name, positive_prompt in prompts:
            print(f"\n{'='*50}")
            print(f'Generating: {name}')
            print(f"{'='*50}")

            # Create workflow
            workflow = WorkflowBuilder.create_simple_txt2img(
                model_name='v1-5-pruned-emaonly-fp16.safetensors',
                positive_prompt=positive_prompt,
                negative_prompt='blurry, low quality, watermark, text',
                width=512,
                height=512,
                steps=20,
                cfg=7.0,
                seed=random_seed(),
            )

            # Progress tracker
            tracker = ProgressTracker()

            def on_progress(data: dict[str, object]) -> None:
                tracker.update(data)
                print(f'\rProgress: {tracker}', end='', flush=True)

            try:
                images = client.generate(
                    workflow.to_dict(),
                    on_progress=on_progress,
                )
                print()  # New line after progress

                if images:
                    saved = save_images(
                        images,
                        './output/batch',
                        prefix=name.replace(' ', '_'),
                    )
                    print(f'Saved: {saved[0]}')

            except Exception as e:
                print(f'\nError generating {name}: {e}')
                continue

    print('\nBatch generation complete!')


if __name__ == '__main__':
    main()

"""
Workflow management utilities for ComfyUI.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import cast

class Workflow:
    """
    Represents a ComfyUI workflow.

    This class provides methods to load, modify, and save workflows.
    """

    def __init__(self, data: dict[str, object] | None = None) -> None:
        """
        Initialize a Workflow.

        :param data: Optional workflow data dictionary
        """
        self._data: dict[str, object] = data or {}

    @classmethod
    def from_file(cls, filepath: str | Path) -> Workflow:
        """
        Load a workflow from a JSON file.

        :param filepath: Path to the workflow JSON file
        :return: Workflow instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded: object = cast(object, json.load(f))
        if isinstance(loaded, dict):
            result: dict[str, object] = cast(dict[str, object], loaded)
            return cls(result)
        return cls({})

    @classmethod
    def from_json(cls, json_str: str) -> Workflow:
        """
        Load a workflow from a JSON string.

        :param json_str: JSON string containing workflow data
        :return: Workflow instance
        """
        loaded: object = cast(object, json.loads(json_str))
        if isinstance(loaded, dict):
            result: dict[str, object] = cast(dict[str, object], loaded)
            return cls(result)
        return cls({})

    def to_dict(self) -> dict[str, object]:
        """
        Convert workflow to dictionary.

        :return: Workflow as dictionary
        """
        return copy.deepcopy(self._data)

    def to_json(self, indent: int | None = 2) -> str:
        """
        Convert workflow to JSON string.

        :param indent: JSON indentation level
        :return: JSON string
        """
        return json.dumps(self._data, indent=indent, ensure_ascii=False)

    def save(self, filepath: str | Path) -> None:
        """
        Save workflow to a JSON file.

        :param filepath: Target file path
        """
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get_node(self, node_id: str) -> dict[str, object] | None:
        """
        Get a node by its ID.

        :param node_id: Node ID
        :return: Node data or None
        """
        node: object = self._data.get(node_id)
        if isinstance(node, dict):
            result: dict[str, object] = cast(dict[str, object], node)
            return result
        return None

    def set_node_input(
        self,
        node_id: str,
        input_name: str,
        value: object,
    ) -> Workflow:
        """
        Set an input value for a node.

        :param node_id: Target node ID
        :param input_name: Input parameter name
        :param value: New value
        :return: Self for method chaining
        """
        if node_id in self._data:
            node_obj: object = self._data[node_id]
            if isinstance(node_obj, dict):
                typed_node: dict[str, object] = cast(dict[str, object], node_obj)
                if 'inputs' not in typed_node:
                    typed_node['inputs'] = {}
                inputs_obj: object = typed_node.get('inputs')
                if isinstance(inputs_obj, dict):
                    typed_inputs: dict[str, object] = cast(dict[str, object], inputs_obj)
                    typed_inputs[input_name] = value
        return self

    def get_node_input(
        self,
        node_id: str,
        input_name: str,
    ) -> object:
        """
        Get an input value from a node.

        :param node_id: Node ID
        :param input_name: Input parameter name
        :return: Input value or None
        """
        node_obj: object = self._data.get(node_id)
        if isinstance(node_obj, dict):
            typed_node: dict[str, object] = cast(dict[str, object], node_obj)
            inputs_obj: object = typed_node.get('inputs')
            if isinstance(inputs_obj, dict):
                typed_inputs: dict[str, object] = cast(dict[str, object], inputs_obj)
                return typed_inputs.get(input_name)
        return None

    def find_nodes_by_class(self, class_type: str) -> list[tuple[str, dict[str, object]]]:
        """
        Find all nodes of a specific class type.

        :param class_type: Node class type (e.g., 'KSampler', 'CLIPTextEncode')
        :return: List of (node_id, node_data) tuples
        """
        results: list[tuple[str, dict[str, object]]] = []
        for node_id, node_val in self._data.items():
            if isinstance(node_val, dict):
                node_data: dict[str, object] = cast(dict[str, object], node_val)
                if node_data.get('class_type') == class_type:
                    results.append((node_id, node_data))
        return results

    def set_seed(self, seed: int, node_id: str | None = None) -> Workflow:
        """
        Set the seed value for KSampler nodes.

        :param seed: Seed value
        :param node_id: Specific node ID, or None for all KSampler nodes
        :return: Self for method chaining
        """
        if node_id:
            _ = self.set_node_input(node_id, 'seed', seed)
        else:
            for nid, _ in self.find_nodes_by_class('KSampler'):
                _ = self.set_node_input(nid, 'seed', seed)
            for nid, _ in self.find_nodes_by_class('KSamplerAdvanced'):
                _ = self.set_node_input(nid, 'noise_seed', seed)
        return self

    def set_prompt(
        self,
        positive: str | None = None,
        negative: str | None = None,
    ) -> Workflow:
        """
        Set positive and/or negative prompts.

        This method attempts to find CLIPTextEncode nodes and update their text.
        It uses heuristics based on node connections to identify positive vs negative.

        :param positive: Positive prompt text
        :param negative: Negative prompt text
        :return: Self for method chaining
        """
        clip_nodes = self.find_nodes_by_class('CLIPTextEncode')

        if len(clip_nodes) == 2:
            # Try to identify positive/negative by checking connections to sampler
            for node_id, _ in clip_nodes:
                # Check if this node connects to 'positive' or 'negative' input
                for other_val in self._data.values():
                    if isinstance(other_val, dict) and 'inputs' in other_val:
                        other_data: dict[str, object] = cast(dict[str, object], other_val)
                        inputs_val: object = other_data.get('inputs')
                        if isinstance(inputs_val, dict):
                            typed_inputs: dict[str, object] = cast(dict[str, object], inputs_val)
                            for input_name_key in typed_inputs:
                                input_val: object = typed_inputs[input_name_key]
                                if isinstance(input_val, list):
                                    typed_list: list[object] = cast(list[object], input_val)
                                    if len(typed_list) >= 1:
                                        if typed_list[0] == node_id:
                                            if input_name_key == 'positive' and positive:
                                                _ = self.set_node_input(node_id, 'text', positive)
                                            elif input_name_key == 'negative' and negative:
                                                _ = self.set_node_input(node_id, 'text', negative)
        elif len(clip_nodes) == 1 and positive:
            # Only one CLIP node, assume it's positive
            _ = self.set_node_input(clip_nodes[0][0], 'text', positive)

        return self

    def set_image_size(
        self,
        width: int,
        height: int,
        node_id: str | None = None,
    ) -> Workflow:
        """
        Set the image size for EmptyLatentImage nodes.

        :param width: Image width
        :param height: Image height
        :param node_id: Specific node ID, or None for all matching nodes
        :return: Self for method chaining
        """
        if node_id:
            _ = self.set_node_input(node_id, 'width', width)
            _ = self.set_node_input(node_id, 'height', height)
        else:
            for nid, _ in self.find_nodes_by_class('EmptyLatentImage'):
                _ = self.set_node_input(nid, 'width', width)
                _ = self.set_node_input(nid, 'height', height)
        return self

    def set_steps(self, steps: int, node_id: str | None = None) -> Workflow:
        """
        Set the number of sampling steps.

        :param steps: Number of steps
        :param node_id: Specific node ID, or None for all KSampler nodes
        :return: Self for method chaining
        """
        if node_id:
            _ = self.set_node_input(node_id, 'steps', steps)
        else:
            for nid, _ in self.find_nodes_by_class('KSampler'):
                _ = self.set_node_input(nid, 'steps', steps)
        return self

    def set_cfg(self, cfg: float, node_id: str | None = None) -> Workflow:
        """
        Set the CFG scale value.

        :param cfg: CFG scale value
        :param node_id: Specific node ID, or None for all KSampler nodes
        :return: Self for method chaining
        """
        if node_id:
            _ = self.set_node_input(node_id, 'cfg', cfg)
        else:
            for nid, _ in self.find_nodes_by_class('KSampler'):
                _ = self.set_node_input(nid, 'cfg', cfg)
        return self

    def set_sampler(
        self,
        sampler_name: str,
        scheduler: str | None = None,
        node_id: str | None = None,
    ) -> Workflow:
        """
        Set the sampler and scheduler.

        :param sampler_name: Sampler name (e.g., 'euler', 'dpmpp_2m')
        :param scheduler: Scheduler name (e.g., 'normal', 'karras')
        :param node_id: Specific node ID, or None for all KSampler nodes
        :return: Self for method chaining
        """
        if node_id:
            _ = self.set_node_input(node_id, 'sampler_name', sampler_name)
            if scheduler:
                _ = self.set_node_input(node_id, 'scheduler', scheduler)
        else:
            for nid, _ in self.find_nodes_by_class('KSampler'):
                _ = self.set_node_input(nid, 'sampler_name', sampler_name)
                if scheduler:
                    _ = self.set_node_input(nid, 'scheduler', scheduler)
        return self

    def set_model(
        self,
        model_name: str,
        node_id: str | None = None,
    ) -> Workflow:
        """
        Set the checkpoint model.

        :param model_name: Model filename
        :param node_id: Specific node ID, or None for all CheckpointLoader nodes
        :return: Self for method chaining
        """
        if node_id:
            _ = self.set_node_input(node_id, 'ckpt_name', model_name)
        else:
            for nid, _ in self.find_nodes_by_class('CheckpointLoaderSimple'):
                _ = self.set_node_input(nid, 'ckpt_name', model_name)
            for nid, _ in self.find_nodes_by_class('CheckpointLoader'):
                _ = self.set_node_input(nid, 'ckpt_name', model_name)
        return self

    def set_batch_size(
        self,
        batch_size: int,
        node_id: str | None = None,
    ) -> Workflow:
        """
        Set the batch size for image generation.

        :param batch_size: Number of images to generate
        :param node_id: Specific node ID, or None for all EmptyLatentImage nodes
        :return: Self for method chaining
        """
        if node_id:
            _ = self.set_node_input(node_id, 'batch_size', batch_size)
        else:
            for nid, _ in self.find_nodes_by_class('EmptyLatentImage'):
                _ = self.set_node_input(nid, 'batch_size', batch_size)
        return self

    def clone(self) -> Workflow:
        """
        Create a deep copy of this workflow.

        :return: New Workflow instance
        """
        return Workflow(copy.deepcopy(self._data))

    @property
    def node_ids(self) -> list[str]:
        """Get all node IDs in the workflow."""
        return list(self._data.keys())

    def __getitem__(self, node_id: str) -> dict[str, object] | None:
        """Get node by ID using bracket notation."""
        node_val: object = self._data.get(node_id)
        if isinstance(node_val, dict):
            result: dict[str, object] = cast(dict[str, object], node_val)
            return result
        return None

    def __setitem__(self, node_id: str, value: dict[str, object]) -> None:
        """Set node by ID using bracket notation."""
        self._data[node_id] = value

    def __contains__(self, node_id: str) -> bool:
        """Check if node ID exists."""
        return node_id in self._data

    def __len__(self) -> int:
        """Get number of nodes."""
        return len(self._data)


class WorkflowBuilder:
    """
    A builder class for creating ComfyUI workflows programmatically.

    This provides a fluent API for constructing workflows from scratch.
    """

    def __init__(self) -> None:
        """Initialize an empty workflow builder."""
        self._nodes: dict[str, object] = {}
        self._next_id: int = 1

    def _get_next_id(self) -> str:
        """Get the next available node ID."""
        node_id = str(self._next_id)
        self._next_id += 1
        return node_id

    def add_node(
        self,
        class_type: str,
        inputs: dict[str, object] | None = None,
        node_id: str | None = None,
    ) -> str:
        """
        Add a node to the workflow.

        :param class_type: Node class type
        :param inputs: Node inputs
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        if node_id is None:
            node_id = self._get_next_id()
        else:
            # Update next_id if necessary
            try:
                id_num = int(node_id)
                if id_num >= self._next_id:
                    self._next_id = id_num + 1
            except ValueError:
                pass

        self._nodes[node_id] = {
            'class_type': class_type,
            'inputs': inputs or {},
        }
        return node_id

    def add_checkpoint_loader(
        self,
        model_name: str,
        node_id: str | None = None,
    ) -> str:
        """
        Add a CheckpointLoaderSimple node.

        :param model_name: Model filename
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        return self.add_node(
            'CheckpointLoaderSimple',
            {'ckpt_name': model_name},
            node_id,
        )

    def add_clip_text_encode(
        self,
        text: str,
        clip_ref: tuple[str, int],
        node_id: str | None = None,
    ) -> str:
        """
        Add a CLIPTextEncode node.

        :param text: Prompt text
        :param clip_ref: Reference to CLIP output (node_id, output_index)
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        return self.add_node(
            'CLIPTextEncode',
            {
                'text': text,
                'clip': list(clip_ref),
            },
            node_id,
        )

    def add_empty_latent(
        self,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        node_id: str | None = None,
    ) -> str:
        """
        Add an EmptyLatentImage node.

        :param width: Image width
        :param height: Image height
        :param batch_size: Batch size
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        return self.add_node(
            'EmptyLatentImage',
            {
                'width': width,
                'height': height,
                'batch_size': batch_size,
            },
            node_id,
        )

    def add_ksampler(
        self,
        model_ref: tuple[str, int],
        positive_ref: tuple[str, int],
        negative_ref: tuple[str, int],
        latent_ref: tuple[str, int],
        seed: int = 0,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = 'euler',
        scheduler: str = 'normal',
        denoise: float = 1.0,
        node_id: str | None = None,
    ) -> str:
        """
        Add a KSampler node.

        :param model_ref: Reference to model output
        :param positive_ref: Reference to positive conditioning
        :param negative_ref: Reference to negative conditioning
        :param latent_ref: Reference to latent image
        :param seed: Random seed
        :param steps: Number of steps
        :param cfg: CFG scale
        :param sampler_name: Sampler name
        :param scheduler: Scheduler name
        :param denoise: Denoise strength
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        return self.add_node(
            'KSampler',
            {
                'seed': seed,
                'steps': steps,
                'cfg': cfg,
                'sampler_name': sampler_name,
                'scheduler': scheduler,
                'denoise': denoise,
                'model': list(model_ref),
                'positive': list(positive_ref),
                'negative': list(negative_ref),
                'latent_image': list(latent_ref),
            },
            node_id,
        )

    def add_vae_decode(
        self,
        samples_ref: tuple[str, int],
        vae_ref: tuple[str, int],
        node_id: str | None = None,
    ) -> str:
        """
        Add a VAEDecode node.

        :param samples_ref: Reference to latent samples
        :param vae_ref: Reference to VAE
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        return self.add_node(
            'VAEDecode',
            {
                'samples': list(samples_ref),
                'vae': list(vae_ref),
            },
            node_id,
        )

    def add_save_image(
        self,
        images_ref: tuple[str, int],
        filename_prefix: str = 'ComfyUI',
        node_id: str | None = None,
    ) -> str:
        """
        Add a SaveImage node.

        :param images_ref: Reference to images
        :param filename_prefix: Output filename prefix
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        return self.add_node(
            'SaveImage',
            {
                'filename_prefix': filename_prefix,
                'images': list(images_ref),
            },
            node_id,
        )

    def add_preview_image(
        self,
        images_ref: tuple[str, int],
        node_id: str | None = None,
    ) -> str:
        """
        Add a PreviewImage node.

        :param images_ref: Reference to images
        :param node_id: Optional specific node ID
        :return: The node ID
        """
        return self.add_node(
            'PreviewImage',
            {'images': list(images_ref)},
            node_id,
        )

    def build(self) -> Workflow:
        """
        Build and return the workflow.

        :return: Workflow instance
        """
        workflow_data: dict[str, object] = copy.deepcopy(self._nodes)
        return Workflow(workflow_data)

    def build_dict(self) -> dict[str, object]:
        """
        Build and return the workflow as a dictionary.

        :return: Workflow dictionary
        """
        workflow_data: dict[str, object] = copy.deepcopy(self._nodes)
        return workflow_data

    @classmethod
    def create_simple_txt2img(
        cls,
        model_name: str,
        positive_prompt: str,
        negative_prompt: str = '',
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = 0,
        sampler: str = 'euler',
        scheduler: str = 'normal',
    ) -> Workflow:
        """
        Create a simple text-to-image workflow.

        :param model_name: Checkpoint model filename
        :param positive_prompt: Positive prompt
        :param negative_prompt: Negative prompt
        :param width: Image width
        :param height: Image height
        :param steps: Number of sampling steps
        :param cfg: CFG scale
        :param seed: Random seed
        :param sampler: Sampler name
        :param scheduler: Scheduler name
        :return: Complete workflow
        """
        builder = cls()

        # Add nodes
        loader_id = builder.add_checkpoint_loader(model_name)
        positive_id = builder.add_clip_text_encode(
            positive_prompt,
            (loader_id, 1),
        )
        negative_id = builder.add_clip_text_encode(
            negative_prompt,
            (loader_id, 1),
        )
        latent_id = builder.add_empty_latent(width, height)
        sampler_id = builder.add_ksampler(
            model_ref=(loader_id, 0),
            positive_ref=(positive_id, 0),
            negative_ref=(negative_id, 0),
            latent_ref=(latent_id, 0),
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
        )
        decode_id = builder.add_vae_decode(
            samples_ref=(sampler_id, 0),
            vae_ref=(loader_id, 2),
        )
        _ = builder.add_save_image(images_ref=(decode_id, 0))

        return builder.build()

"""
Tests for Workflow class.
"""

import json
from typing import cast

from comfyui_client.workflow import Workflow, WorkflowBuilder


class TestWorkflow:
    """Tests for Workflow class."""

    def test_init_empty(self) -> None:
        """Test creating empty workflow."""
        workflow = Workflow()
        assert len(workflow) == 0
        assert workflow.to_dict() == {}

    def test_init_with_data(self) -> None:
        """Test creating workflow with data."""
        data: dict[str, object] = {
            '1': {
                'class_type': 'CheckpointLoaderSimple',
                'inputs': {'ckpt_name': 'model.safetensors'},
            }
        }
        workflow = Workflow(data)
        assert len(workflow) == 1
        assert '1' in workflow

    def test_from_json(self) -> None:
        """Test loading workflow from JSON string."""
        json_str = '{"1": {"class_type": "Test", "inputs": {}}}'
        workflow = Workflow.from_json(json_str)
        assert len(workflow) == 1
        node = workflow['1']
        assert node is not None
        assert node['class_type'] == 'Test'

    def test_to_json(self) -> None:
        """Test converting workflow to JSON."""
        workflow = Workflow({'1': {'class_type': 'Test', 'inputs': {}}})
        json_str = workflow.to_json()
        loaded: object = cast(object, json.loads(json_str))
        assert isinstance(loaded, dict)
        data: dict[str, object] = cast(dict[str, object], loaded)
        node_data = data.get('1')
        assert isinstance(node_data, dict)
        typed_node: dict[str, object] = cast(dict[str, object], node_data)
        assert typed_node.get('class_type') == 'Test'

    def test_get_node(self) -> None:
        """Test getting a node."""
        workflow = Workflow({
            '1': {'class_type': 'Test', 'inputs': {'value': 42}},
        })
        node = workflow.get_node('1')
        assert node is not None
        assert node['class_type'] == 'Test'

    def test_get_node_not_found(self) -> None:
        """Test getting non-existent node."""
        workflow = Workflow()
        assert workflow.get_node('999') is None

    def test_set_node_input(self) -> None:
        """Test setting node input."""
        workflow = Workflow({
            '1': {'class_type': 'Test', 'inputs': {'value': 1}},
        })
        _ = workflow.set_node_input('1', 'value', 42)
        assert workflow.get_node_input('1', 'value') == 42

    def test_find_nodes_by_class(self) -> None:
        """Test finding nodes by class type."""
        workflow = Workflow({
            '1': {'class_type': 'KSampler', 'inputs': {}},
            '2': {'class_type': 'CLIPTextEncode', 'inputs': {}},
            '3': {'class_type': 'KSampler', 'inputs': {}},
        })
        samplers = workflow.find_nodes_by_class('KSampler')
        assert len(samplers) == 2

    def test_set_seed(self) -> None:
        """Test setting seed on all samplers."""
        workflow = Workflow({
            '1': {'class_type': 'KSampler', 'inputs': {'seed': 0}},
            '2': {'class_type': 'KSampler', 'inputs': {'seed': 0}},
        })
        _ = workflow.set_seed(12345)
        assert workflow.get_node_input('1', 'seed') == 12345
        assert workflow.get_node_input('2', 'seed') == 12345

    def test_set_image_size(self) -> None:
        """Test setting image size."""
        workflow = Workflow({
            '1': {
                'class_type': 'EmptyLatentImage',
                'inputs': {'width': 512, 'height': 512},
            },
        })
        _ = workflow.set_image_size(768, 1024)
        assert workflow.get_node_input('1', 'width') == 768
        assert workflow.get_node_input('1', 'height') == 1024

    def test_clone(self) -> None:
        """Test cloning workflow."""
        original = Workflow({'1': {'class_type': 'Test', 'inputs': {'v': 1}}})
        cloned = original.clone()

        # Modify original
        _ = original.set_node_input('1', 'v', 999)

        # Clone should be unchanged
        assert cloned.get_node_input('1', 'v') == 1

    def test_node_ids(self) -> None:
        """Test getting node IDs."""
        workflow = Workflow({
            '1': {'class_type': 'A', 'inputs': {}},
            '2': {'class_type': 'B', 'inputs': {}},
        })
        ids = workflow.node_ids
        assert '1' in ids
        assert '2' in ids


class TestWorkflowBuilder:
    """Tests for WorkflowBuilder class."""

    def test_add_node(self) -> None:
        """Test adding a node."""
        builder = WorkflowBuilder()
        node_id = builder.add_node('TestNode', {'value': 42})
        assert node_id == '1'
        workflow = builder.build()
        node = workflow['1']
        assert node is not None
        assert node['class_type'] == 'TestNode'

    def test_add_checkpoint_loader(self) -> None:
        """Test adding checkpoint loader."""
        builder = WorkflowBuilder()
        node_id = builder.add_checkpoint_loader('model.safetensors')
        workflow = builder.build()
        node = workflow[node_id]
        assert node is not None
        assert node['class_type'] == 'CheckpointLoaderSimple'
        inputs = node.get('inputs')
        assert isinstance(inputs, dict)
        typed_inputs: dict[str, object] = cast(dict[str, object], inputs)
        assert typed_inputs.get('ckpt_name') == 'model.safetensors'

    def test_add_ksampler(self) -> None:
        """Test adding KSampler."""
        builder = WorkflowBuilder()
        node_id = builder.add_ksampler(
            model_ref=('1', 0),
            positive_ref=('2', 0),
            negative_ref=('3', 0),
            latent_ref=('4', 0),
            seed=42,
            steps=20,
        )
        workflow = builder.build()
        node = workflow[node_id]
        assert node is not None
        assert node['class_type'] == 'KSampler'
        inputs = node.get('inputs')
        assert isinstance(inputs, dict)
        typed_inputs: dict[str, object] = cast(dict[str, object], inputs)
        assert typed_inputs.get('seed') == 42
        assert typed_inputs.get('steps') == 20

    def test_create_simple_txt2img(self) -> None:
        """Test creating simple txt2img workflow."""
        workflow = WorkflowBuilder.create_simple_txt2img(
            model_name='model.safetensors',
            positive_prompt='test positive',
            negative_prompt='test negative',
            width=768,
            height=512,
            steps=25,
        )

        # Verify workflow has expected node types
        workflow_dict = workflow.to_dict()
        class_types: list[str] = []
        for node_val in workflow_dict.values():
            if isinstance(node_val, dict):
                node: dict[str, object] = cast(dict[str, object], node_val)
                ct = node.get('class_type')
                if isinstance(ct, str):
                    class_types.append(ct)
        assert 'CheckpointLoaderSimple' in class_types
        assert 'CLIPTextEncode' in class_types
        assert 'EmptyLatentImage' in class_types
        assert 'KSampler' in class_types
        assert 'VAEDecode' in class_types
        assert 'SaveImage' in class_types

    def test_build_dict(self) -> None:
        """Test building workflow as dict."""
        builder = WorkflowBuilder()
        _ = builder.add_node('Test', {})
        result = builder.build_dict()
        assert isinstance(result, dict)
        assert '1' in result

    def test_inpaint_workflow_with_mask(self) -> None:
        """Test creating an inpaint workflow with mask."""
        builder = WorkflowBuilder()

        # Add checkpoint loader
        loader_id = builder.add_checkpoint_loader('inpaint_model.safetensors')

        # Add LoadImage node for source image
        image_id = builder.add_node(
            'LoadImage',
            {'image': 'source_image.png'},
        )

        # Add LoadImage node for mask
        mask_id = builder.add_node(
            'LoadImage',
            {'image': 'mask_image.png'},
        )

        # Add SetLatentNoiseMask node to apply mask to latent
        set_mask_id = builder.add_node(
            'SetLatentNoiseMask',
            {
                'samples': [image_id, 0],
                'mask': [mask_id, 1],
            },
        )

        # Add CLIP text encode for positive prompt
        positive_id = builder.add_clip_text_encode(
            'beautiful landscape',
            (loader_id, 1),
        )

        # Add CLIP text encode for negative prompt
        negative_id = builder.add_clip_text_encode(
            'blurry, low quality',
            (loader_id, 1),
        )

        # Add VAEEncode for image
        vae_encode_id = builder.add_node(
            'VAEEncode',
            {
                'pixels': [image_id, 0],
                'vae': [loader_id, 2],
            },
        )

        # Add KSampler with masked latent
        sampler_id = builder.add_ksampler(
            model_ref=(loader_id, 0),
            positive_ref=(positive_id, 0),
            negative_ref=(negative_id, 0),
            latent_ref=(set_mask_id, 0),
            seed=12345,
            steps=20,
            denoise=.8,
        )

        # Build workflow
        workflow = builder.build()

        # Verify nodes exist
        assert workflow[loader_id] is not None
        assert workflow[image_id] is not None
        assert workflow[mask_id] is not None
        assert workflow[set_mask_id] is not None

        # Verify SetLatentNoiseMask node
        mask_node = workflow[set_mask_id]
        assert mask_node is not None
        assert mask_node['class_type'] == 'SetLatentNoiseMask'
        mask_inputs = mask_node.get('inputs')
        assert isinstance(mask_inputs, dict)
        typed_mask_inputs: dict[str, object] = cast(dict[str, object], mask_inputs)
        assert 'mask' in typed_mask_inputs

        # Verify sampler references masked latent
        sampler_node = workflow[sampler_id]
        assert sampler_node is not None
        sampler_inputs = sampler_node.get('inputs')
        assert isinstance(sampler_inputs, dict)
        typed_sampler_inputs: dict[str, object] = cast(dict[str, object], sampler_inputs)
        latent_ref = typed_sampler_inputs.get('latent_image')
        assert isinstance(latent_ref, list)
        typed_latent_ref: list[object] = cast(list[object], latent_ref)
        assert typed_latent_ref[0] == set_mask_id

        # Verify VAEEncode node exists
        vae_node = workflow[vae_encode_id]
        assert vae_node is not None
        assert vae_node['class_type'] == 'VAEEncode'

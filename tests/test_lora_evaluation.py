import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from core import sdxl


class LoRAEvaluationTests(unittest.TestCase):
    def test_comparison_reuses_seed_for_every_variant(self):
        prompts = ["portrait", "full body"]
        variants = [
            {"label": "Base", "model_path": "base.safetensors", "steps": 20, "cfg": 7.0},
            {"label": "LoRA 0.8", "model_path": "base.safetensors", "steps": 24, "cfg": 6.0},
        ]

        with patch.object(sdxl, "sdxl_generate", side_effect=lambda **kwargs: kwargs["seed"]) as generate:
            cells, labels = sdxl.comparison_generate(
                prompts,
                variants,
                count=2,
                seed=100,
            )

        self.assertEqual(labels, ["Base", "LoRA 0.8"])
        self.assertEqual([cell["seeds"] for cell in cells], [
            [100, 101],
            [102, 103],
            [100, 101],
            [102, 103],
        ])
        self.assertEqual([call.kwargs["seed"] for call in generate.call_args_list], [
            100, 101, 102, 103, 100, 101, 102, 103,
        ])
        self.assertEqual([cell["steps"] for cell in cells], [20, 20, 24, 24])
        self.assertEqual([cell["cfg"] for cell in cells], [7.0, 7.0, 6.0, 6.0])

    def test_adapter_is_loaded_once_per_variant(self):
        variants = [{
            "model_path": "base.safetensors",
            "lora_paths": ["adapter.safetensors"],
            "lora_weights": [0.8],
            "steps": 20,
            "cfg": 7.0,
        }]

        with patch.object(sdxl, "sdxl_generate", return_value="image") as generate:
            sdxl.comparison_generate(["one", "two"], variants, count=2, seed=42)

        configure_flags = [call.kwargs["configure_loras"] for call in generate.call_args_list]
        self.assertEqual(configure_flags, [True, False, False, False])

    def test_cached_pipeline_lora_is_unloaded_before_baseline(self):
        pipeline = Mock()
        pipeline._ai_toolkit_lora_adapters = ["old_adapter"]

        sdxl._configure_pipeline_loras(pipeline, [], [])

        pipeline.unload_lora_weights.assert_called_once_with()
        pipeline.load_lora_weights.assert_not_called()
        self.assertEqual(pipeline._ai_toolkit_lora_adapters, [])

    def test_requested_lora_defaults_to_weight_one(self):
        pipeline = Mock()
        pipeline._ai_toolkit_lora_adapters = []
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as lora_file:
            names = sdxl._configure_pipeline_loras(pipeline, [lora_file.name], None)

        self.assertEqual(names, ["ai_toolkit_lora_0"])
        pipeline.set_adapters.assert_called_once_with(
            ["ai_toolkit_lora_0"], adapter_weights=[1.0]
        )
        load_call = pipeline.load_lora_weights.call_args
        self.assertEqual(load_call.args[0], os.path.dirname(lora_file.name))
        self.assertEqual(load_call.kwargs["weight_name"], os.path.basename(lora_file.name))


if __name__ == "__main__":
    unittest.main()

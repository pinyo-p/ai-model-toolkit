import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from core.runtimes import (
    Krea2Runtime,
    RuntimeLoadContext,
    detect_model_type,
    get_runtime,
    get_runtime_defaults,
)


class ModelRuntimeTests(unittest.TestCase):
    def test_existing_family_defaults_and_cfg_are_preserved(self):
        self.assertEqual(
            (get_runtime_defaults("FLUX.2-klein-9B.safetensors").steps,
             get_runtime_defaults("FLUX.2-klein-9B.safetensors").cfg),
            (4, 1.0),
        )
        self.assertEqual(
            get_runtime("flux2").effective_cfg("FLUX.2-klein-9B.safetensors", 7.0),
            1.0,
        )
        self.assertEqual(
            get_runtime("flux2").effective_cfg("FLUX.2-dev.safetensors", 4.0),
            4.0,
        )

    def test_krea2_repo_and_paths_are_detected(self):
        self.assertEqual(detect_model_type("krea/Krea-2-Turbo"), "krea2")
        self.assertEqual(detect_model_type("/models/krea2/raw"), "krea2")

    def test_krea2_diffusers_directory_is_detected_from_model_index(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "model_index.json"), "w") as file:
                json.dump({"_class_name": "Krea2Pipeline"}, file)
            self.assertEqual(detect_model_type(model_dir), "krea2")

    def test_krea2_defaults_match_official_release_recipes(self):
        raw = get_runtime_defaults("krea/Krea-2-Raw")
        turbo = get_runtime_defaults("krea/Krea-2-Turbo")

        self.assertEqual((raw.steps, raw.cfg), (52, 3.5))
        self.assertEqual((turbo.steps, turbo.cfg), (8, 0.0))
        self.assertEqual(get_runtime("krea2").effective_cfg("Krea-2-Turbo", 9.0), 0.0)
        self.assertEqual(get_runtime("krea2").effective_cfg("Krea-2-Raw", 3.5), 3.5)

    def test_krea2_runtime_delegates_full_repo_loading(self):
        loader = Mock(return_value="pipeline")
        context = RuntimeLoadContext(
            model_path="krea/Krea-2-Turbo",
            vae_path=None,
            text_encoder_path=None,
            vae=None,
            dtype=torch.bfloat16,
            load_pipeline=loader,
        )
        fake_pipeline_class = type("Krea2Pipeline", (), {})

        with patch("diffusers.Krea2Pipeline", fake_pipeline_class, create=True):
            pipeline = Krea2Runtime().load(context)

        self.assertEqual(pipeline, "pipeline")
        loader.assert_called_once_with(
            fake_pipeline_class,
            "krea/Krea-2-Turbo",
            dtype=torch.bfloat16,
        )

    def test_krea2_single_file_explains_required_format(self):
        with tempfile.NamedTemporaryFile(prefix="krea2-", suffix=".safetensors") as checkpoint:
            context = RuntimeLoadContext(
                model_path=checkpoint.name,
                vae_path=None,
                text_encoder_path=None,
                vae=None,
                dtype=torch.bfloat16,
                load_pipeline=Mock(),
            )
            with self.assertRaisesRegex(RuntimeError, "full Diffusers repository"):
                Krea2Runtime().load(context)


if __name__ == "__main__":
    unittest.main()

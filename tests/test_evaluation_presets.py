import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

import main
from core import datasets, evaluation_presets


def _image_file():
    buffer = io.BytesIO()
    Image.new("RGB", (768, 768), "navy").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _manifest_with_run(lora_path: str, dataset_type: str = "person") -> dict:
    return {
        "id": "subject-a1b2",
        "name": "Subject",
        "type": dataset_type,
        "trigger_word": "subject_x",
        "training_runs": [{
            "job_id": "run-1",
            "status": "done",
            "profile": "balanced",
            "lora_path": lora_path,
            "recipe": {"inference_model": "krea/Krea-2-Turbo", "seed": 77},
        }],
    }


class EvaluationPresetTests(unittest.TestCase):
    def test_completed_run_builds_base_and_two_weight_variants(self):
        with tempfile.TemporaryDirectory() as root:
            lora_path = Path(root) / "trained.safetensors"
            lora_path.write_bytes(b"weights")

            preset = evaluation_presets.build_evaluation_preset(
                _manifest_with_run(str(lora_path))
            )

            self.assertEqual(preset["model"], "krea/Krea-2-Turbo")
            self.assertEqual(preset["seed"], 77)
            self.assertEqual([item["weight"] for item in preset["variants"]], [None, 0.7, 1.0])
            self.assertEqual(len(preset["prompts"]), 6)
            self.assertTrue(all("subject_x" in prompt for prompt in preset["prompts"]))

    def test_missing_lora_is_explained(self):
        with self.assertRaisesRegex(evaluation_presets.EvaluationPresetError, "moved or deleted"):
            evaluation_presets.build_evaluation_preset(
                _manifest_with_run("/missing/trained.safetensors")
            )

    def test_every_dataset_type_has_six_triggered_prompts(self):
        for dataset_type in ("person", "style", "clothing", "environment", "vehicle", "object"):
            with self.subTest(dataset_type=dataset_type):
                prompts = evaluation_presets.evaluation_prompts(
                    _manifest_with_run("unused", dataset_type=dataset_type)
                )
                self.assertEqual(len(prompts), 6)
                self.assertTrue(all("subject_x" in prompt for prompt in prompts))

    def test_authenticated_preset_api_uses_persisted_training_run(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Jacket", "clothing", "jacket_x", [("jacket.png", _image_file())]
            )
            lora_path = Path(root) / "jacket.safetensors"
            lora_path.write_bytes(b"weights")
            datasets.record_training_run(root, created["id"], {
                "job_id": "run-api",
                "status": "done",
                "profile": "fast",
                "lora_path": str(lora_path),
                "seed": 19,
            })
            main.app.dependency_overrides[main.get_current_user] = lambda: "tester"
            client = TestClient(main.app)
            try:
                with patch.object(main, "_datasets_root", return_value=root):
                    response = client.get(
                        f"/api/datasets/{created['id']}/evaluation-preset?run_id=run-api"
                    )
                self.assertEqual(response.status_code, 200)
                preset = response.json()
                self.assertEqual(preset["seed"], 19)
                self.assertIn("wearing jacket_x", preset["prompts"][0])
            finally:
                main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()

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


def _experiment(lora_path: str) -> dict:
    return {
        "prompts": ["portrait", "profile", "full body"],
        "variants": [
            {"label": "Base", "loras": []},
            {"label": "Subject 0.7", "loras": [{"path": lora_path, "weight": 0.7}]},
            {"label": "Subject 1.0", "loras": [{"path": lora_path, "weight": 1.0}]},
        ],
    }


def _evaluation_result(lora_path: str) -> dict:
    experiment = _experiment(lora_path)
    cells = []
    for y in range(3):
        for x in range(3):
            image_id = f"{y:02x}{x:02x}" + ("0" * 28)
            cells.append({
                "x": x,
                "y": y,
                "images": [f"/output/eval_{image_id}.png"],
                "seeds": [42 + y],
                "steps": 8,
                "cfg": 0.0,
                "ignored": "not persisted",
            })
    return {
        "experiment": experiment,
        "x_labels": [item["label"] for item in experiment["variants"]],
        "y_labels": experiment["prompts"],
        "cells": cells,
    }


def _write_evaluation_outputs(root: Path, result: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for cell in result["cells"]:
        for url in cell["images"]:
            (root / Path(url).name).write_bytes(b"png evidence")


class EvaluationPresetTests(unittest.TestCase):
    def test_vote_summary_reports_unique_lora_winner(self):
        summary = evaluation_presets.summarize_votes(
            _experiment("/tmp/subject.safetensors"), {"0": 1, "1": 1, "2": 2},
            require_complete=True,
        )

        self.assertEqual(summary["reviewed_count"], 3)
        self.assertEqual(summary["winner"]["label"], "Subject 0.7")
        self.assertEqual(summary["winner"]["lora_weight"], 0.7)
        self.assertEqual(summary["verdict"], "lora")

    def test_vote_summary_reports_tie_and_rejects_incomplete_review(self):
        experiment = _experiment("/tmp/subject.safetensors")
        summary = evaluation_presets.summarize_votes(experiment, {0: 0, 1: 1})
        self.assertIsNone(summary["winner"])
        self.assertEqual(summary["tied_variant_indices"], [0, 1])
        with self.assertRaisesRegex(evaluation_presets.EvaluationPresetError, "every prompt"):
            evaluation_presets.summarize_votes(
                experiment, {0: 0, 1: 1}, require_complete=True
            )

    def test_vote_summary_rejects_unknown_variant(self):
        with self.assertRaisesRegex(evaluation_presets.EvaluationPresetError, "unknown variant"):
            evaluation_presets.summarize_votes(_experiment("unused"), {0: 9})

    def test_comparison_evidence_is_bounded_to_server_output_urls(self):
        result = _evaluation_result("/tmp/subject.safetensors")
        evidence = evaluation_presets.comparison_evidence(result)
        self.assertEqual(evidence["image_count"], 9)
        self.assertNotIn("ignored", evidence["cells"][0])
        result["cells"][0]["images"] = ["https://example.com/untrusted.png"]
        with self.assertRaisesRegex(evaluation_presets.EvaluationPresetError, "image reference"):
            evaluation_presets.comparison_evidence(result)

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

    def test_requested_older_training_run_builds_its_own_preset(self):
        with tempfile.TemporaryDirectory() as root:
            older_lora = Path(root) / "older.safetensors"
            latest_lora = Path(root) / "latest.safetensors"
            older_lora.write_bytes(b"older")
            latest_lora.write_bytes(b"latest")
            manifest = _manifest_with_run(str(older_lora))
            manifest["training_runs"][0]["job_id"] = "run-older"
            manifest["training_runs"][0]["seed"] = 11
            manifest["training_runs"][0]["recipe"]["seed"] = 11
            manifest["training_runs"].append({
                "job_id": "run-latest",
                "status": "done",
                "profile": "quality",
                "lora_path": str(latest_lora),
                "seed": 99,
            })

            preset = evaluation_presets.build_evaluation_preset(
                manifest, run_id="run-older"
            )

            self.assertEqual(preset["training_run_id"], "run-older")
            self.assertEqual(preset["seed"], 11)
            self.assertEqual(preset["variants"][1]["lora_path"], str(older_lora))

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

    def test_authenticated_verdict_api_persists_server_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Subject", "person", "subject_x", [("subject.png", _image_file())]
            )
            lora_path = Path(root) / "subject.safetensors"
            lora_path.write_bytes(b"weights")
            datasets.record_training_run(root, created["id"], {
                "job_id": "run-verdict", "status": "done", "lora_path": str(lora_path),
            })
            evaluation_id = "evaluation-verdict-api"
            result = _evaluation_result(str(lora_path))
            output_root = Path(root) / "output"
            _write_evaluation_outputs(output_root, result)
            main._evaluation_progress[evaluation_id] = {
                "status": "done", "result": result,
            }
            main.app.dependency_overrides[main.get_current_user] = lambda: "tester"
            client = TestClient(main.app)
            try:
                with (
                    patch.object(main, "_datasets_root", return_value=root),
                    patch.object(main, "OUTPUT_DIR", str(output_root)),
                ):
                    response = client.post(
                        f"/api/datasets/{created['id']}/evaluation-verdict",
                        json={
                            "training_run_id": "run-verdict",
                            "evaluation_id": evaluation_id,
                            "votes": {"0": 1, "1": 1, "2": 2},
                        },
                    )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["summary"]["winner"]["label"], "Subject 0.7")
                with (
                    patch.object(main, "_datasets_root", return_value=root),
                    patch.object(main, "OUTPUT_DIR", str(output_root)),
                ):
                    updated_response = client.post(
                        f"/api/datasets/{created['id']}/evaluation-verdict",
                        json={
                            "training_run_id": "run-verdict",
                            "evaluation_id": evaluation_id,
                            "votes": {"0": 0, "1": 0, "2": 1},
                        },
                    )
                self.assertEqual(updated_response.status_code, 200, updated_response.text)
                self.assertEqual(updated_response.json()["summary"]["verdict"], "base")
                stored = datasets.get_dataset(root, created["id"])["evaluation_runs"]
                self.assertEqual(len(stored), 1)
                self.assertEqual(stored[0]["training_run_id"], "run-verdict")
                self.assertEqual(stored[0]["summary"]["winner"]["label"], "Base")
                self.assertEqual(stored[0]["comparison"]["image_count"], 9)
                evidence_url = stored[0]["comparison"]["cells"][0]["images"][0]
                self.assertEqual(
                    evidence_url,
                    f"/api/datasets/{created['id']}/evaluations/{evaluation_id}/images/"
                    "eval_00000000000000000000000000000000.png",
                )
                for output_file in output_root.iterdir():
                    output_file.unlink()
                with patch.object(main, "_datasets_root", return_value=root):
                    evidence_response = client.get(evidence_url)
                self.assertEqual(evidence_response.status_code, 200)
                self.assertEqual(evidence_response.content, b"png evidence")
            finally:
                main._evaluation_progress.pop(evaluation_id, None)
                main.app.dependency_overrides.clear()

    def test_verdict_rejects_evaluation_from_another_lora(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Subject", "person", "subject_x", [("subject.png", _image_file())]
            )
            trained_lora = Path(root) / "trained.safetensors"
            trained_lora.write_bytes(b"weights")
            datasets.record_training_run(root, created["id"], {
                "job_id": "run-mismatch", "status": "done", "lora_path": str(trained_lora),
            })
            evaluation_id = "evaluation-mismatch"
            mismatch_result = _evaluation_result(str(Path(root) / "other.safetensors"))
            output_root = Path(root) / "output"
            _write_evaluation_outputs(output_root, mismatch_result)
            main._evaluation_progress[evaluation_id] = {
                "status": "done",
                "result": mismatch_result,
            }
            main.app.dependency_overrides[main.get_current_user] = lambda: "tester"
            client = TestClient(main.app)
            try:
                with (
                    patch.object(main, "_datasets_root", return_value=root),
                    patch.object(main, "OUTPUT_DIR", str(output_root)),
                ):
                    response = client.post(
                        f"/api/datasets/{created['id']}/evaluation-verdict",
                        json={
                            "training_run_id": "run-mismatch",
                            "evaluation_id": evaluation_id,
                            "votes": {"0": 1, "1": 1, "2": 2},
                        },
                    )
                self.assertEqual(response.status_code, 409)
                self.assertIn("does not test the LoRA", response.json()["detail"])
            finally:
                main._evaluation_progress.pop(evaluation_id, None)
                main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()

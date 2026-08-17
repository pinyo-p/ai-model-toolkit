import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

import main
from core import datasets, expansion


def _image_file(color, size=(768, 768)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class ExpansionRecipeTests(unittest.TestCase):
    def test_person_recipe_prioritizes_identity_and_varied_views(self):
        variations = expansion.build_variations("person", 8)

        self.assertEqual(len(variations), 8)
        self.assertTrue(all("Preserve the exact identity" in item.instruction for item in variations))
        self.assertEqual(len({item.caption for item in variations}), 8)

    def test_style_recipe_changes_subject_but_preserves_style(self):
        variation = expansion.build_variations("style", 1)[0]

        self.assertIn("only as a visual-style reference", variation.instruction)
        self.assertIn("Do not copy their subject", variation.instruction)

    def test_candidate_count_is_bounded(self):
        with self.assertRaisesRegex(expansion.ExpansionError, "between 1 and 24"):
            expansion.build_variations("object", 25)


class ExpansionCandidateTests(unittest.TestCase):
    def test_accept_and_reject_candidates_are_persisted(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Chair", "object", "chair_x", [("chair.png", _image_file("brown"))]
            )
            run_id = "run-a1b2"
            candidate_dir = datasets.expansion_candidate_dir(root, created["id"], run_id)
            Image.new("RGB", (640, 896), "blue").save(candidate_dir / "candidate-0001.png")
            Image.new("RGB", (896, 640), "green").save(candidate_dir / "candidate-0002.png")
            datasets.record_expansion_run(root, created["id"], {
                "run_id": run_id,
                "status": "done",
                "candidates": [
                    {"id": "candidate-0001", "filename": "candidate-0001.png", "caption": "a blue chair", "status": "pending"},
                    {"id": "candidate-0002", "filename": "candidate-0002.png", "caption": "a green chair", "status": "pending"},
                ],
            })

            accepted = datasets.review_expansion_candidates(
                root, created["id"], run_id, ["candidate-0001"], "accept"
            )
            rejected = datasets.review_expansion_candidates(
                root, created["id"], run_id, ["candidate-0002"], "reject"
            )

            self.assertEqual(accepted["analysis"]["image_count"], 2)
            self.assertEqual(accepted["dataset_revision"], 2)
            self.assertEqual(rejected["expansion_runs"][-1]["candidates"][0]["status"], "accepted")
            self.assertEqual(rejected["expansion_runs"][-1]["candidates"][1]["status"], "rejected")
            generated = rejected["images"][-1]
            self.assertEqual(generated["caption"], "a blue chair")
            self.assertTrue((Path(root) / created["id"] / "images" / "image-00002.txt").is_file())

    def test_background_runner_publishes_reviewable_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Car", "vehicle", "car_x", [("car.png", _image_file("red"))]
            )
            job_id = "expansion-test"
            now = "now"
            with main._expansion_lock:
                main._expansion_jobs[job_id] = {
                    "run_id": job_id,
                    "job_id": job_id,
                    "dataset_id": created["id"],
                    "dataset_name": "Car",
                    "status": "queued",
                    "model": expansion.MODEL_ID,
                    "candidate_count": 2,
                    "completed": 0,
                    "created_at": now,
                    "candidates": [],
                }
                main._expansion_cancel_events[job_id] = threading.Event()

            def fake_generate(refs, instruction, seed, progress_cb, cancel_event):
                progress_cb(50, 50)
                return Image.new("RGB", (768, 768), "cyan" if seed == 10 else "magenta")

            with (
                patch.object(main, "_datasets_root", return_value=root),
                patch.object(main.expansion, "generate_candidate", side_effect=fake_generate),
            ):
                main._run_dataset_expansion(job_id, created["id"], 2, 10)

            job = main._expansion_jobs[job_id]
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["completed"], 2)
            self.assertEqual(len(job["candidates"]), 2)
            persisted = datasets.get_dataset(root, created["id"])["expansion_runs"][-1]
            self.assertEqual(persisted["status"], "done")
            self.assertTrue(
                datasets.expansion_candidate_path(
                    root, created["id"], job_id, "candidate-0001"
                ).is_file()
            )

    def test_authenticated_candidate_preview_and_review_api(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Room", "environment", "room_x", [("room.png", _image_file("white"))]
            )
            run_id = "run-api"
            candidate_dir = datasets.expansion_candidate_dir(root, created["id"], run_id)
            Image.new("RGB", (768, 768), "orange").save(candidate_dir / "candidate-0001.png")
            datasets.record_expansion_run(root, created["id"], {
                "run_id": run_id,
                "status": "done",
                "candidates": [{
                    "id": "candidate-0001",
                    "filename": "candidate-0001.png",
                    "caption": "a warm room",
                    "status": "pending",
                }],
            })
            main.app.dependency_overrides[main.get_current_user] = lambda: "tester"
            client = TestClient(main.app)
            try:
                with patch.object(main, "_datasets_root", return_value=root):
                    preview = client.get(
                        f"/api/datasets/{created['id']}/expansion/{run_id}/candidates/candidate-0001"
                    )
                    reviewed = client.post(
                        f"/api/datasets/{created['id']}/expansion/{run_id}/review",
                        json={"candidate_ids": ["candidate-0001"], "decision": "accept"},
                    )
                self.assertEqual(preview.status_code, 200)
                self.assertTrue(preview.headers["content-type"].startswith("image/png"))
                self.assertEqual(reviewed.status_code, 200)
                self.assertEqual(reviewed.json()["analysis"]["image_count"], 2)
            finally:
                main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()

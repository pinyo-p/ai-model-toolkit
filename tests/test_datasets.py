import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

import main
from core import datasets


def _image_file(color, size=(768, 768), image_format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    buffer.seek(0)
    return buffer


class DatasetWorkspaceTests(unittest.TestCase):
    def test_create_dataset_deduplicates_and_analyzes_quick_sources(self):
        with tempfile.TemporaryDirectory() as root:
            first = _image_file("red")
            duplicate = io.BytesIO(first.getvalue())
            result = datasets.create_dataset(
                root,
                "My Person",
                "person",
                "sks person",
                [("portrait.png", first), ("copy.png", duplicate)],
            )

            self.assertEqual(result["analysis"]["image_count"], 1)
            self.assertEqual(result["analysis"]["duplicate_count"], 1)
            self.assertEqual(result["analysis"]["source_mode"], "quick")
            self.assertEqual(result["analysis"]["status"], "needs_captions")
            self.assertEqual(result["dataset_revision"], 1)
            self.assertEqual(result["training_state"]["status"], "none")
            self.assertTrue((Path(root) / result["id"] / "dataset.json").is_file())

    def test_caption_updates_write_trainer_sidecars(self):
        with tempfile.TemporaryDirectory() as root:
            result = datasets.create_dataset(
                root,
                "Room style",
                "environment",
                "",
                [("room.png", _image_file("blue"))],
            )
            image = result["images"][0]
            updated = datasets.update_captions(
                root, result["id"], {image["id"]: "a blue modern room"}
            )

            sidecar = Path(root) / result["id"] / "images" / "image-00001.txt"
            self.assertEqual(sidecar.read_text().strip(), "a blue modern room")
            self.assertEqual(updated["analysis"]["captioned_count"], 1)
            self.assertEqual(updated["analysis"]["status"], "needs_expansion")
            self.assertEqual(updated["dataset_revision"], 2)

            unchanged = datasets.update_captions(
                root, result["id"], {image["id"]: "a blue modern room"}
            )
            self.assertEqual(unchanged["dataset_revision"], 2)

    def test_add_images_appends_unique_files_and_skips_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            first = _image_file("red")
            created = datasets.create_dataset(
                root, "Growing set", "object", "object_x", [("first.png", first)]
            )
            duplicate = _image_file("red")
            second = _image_file("blue")

            updated = datasets.add_images(
                root,
                created["id"],
                [("duplicate.png", duplicate), ("second.png", second)],
            )

            self.assertEqual(updated["import_result"], {"added": 1, "duplicates": 1})
            self.assertEqual([item["id"] for item in updated["images"]], [
                "image-00001", "image-00002",
            ])
            self.assertEqual(updated["analysis"]["image_count"], 2)
            self.assertEqual(updated["dataset_revision"], 2)
            self.assertTrue(
                (Path(root) / created["id"] / "images" / "image-00002.png").is_file()
            )

            duplicate_only = datasets.add_images(
                root, created["id"], [("duplicate-again.png", _image_file("blue"))]
            )
            self.assertEqual(duplicate_only["dataset_revision"], 2)

    def test_training_state_marks_older_dataset_revision(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Versioned set", "object", "object_x", [("first.png", _image_file("red"))]
            )
            datasets.record_training_run(root, created["id"], {
                "job_id": "run-current",
                "status": "done",
                "lora_path": "/models/current.safetensors",
                "dataset_revision": 1,
                "dataset_image_count": 1,
            })
            current = datasets.get_dataset(root, created["id"])
            self.assertEqual(current["training_state"]["status"], "current")

            changed = datasets.add_images(
                root, created["id"], [("second.png", _image_file("blue"))]
            )
            self.assertEqual(changed["training_state"]["status"], "stale")
            self.assertEqual(changed["training_state"]["trained_image_count"], 1)
            self.assertIn("stale_training", {
                issue["code"] for issue in changed["analysis"]["issues"]
            })

    def test_remove_images_deletes_image_and_sidecar_and_advances_revision(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root,
                "Curated set",
                "style",
                "style_x",
                [("first.png", _image_file("red")), ("second.png", _image_file("blue"))],
            )
            first_id = created["images"][0]["id"]
            datasets.update_captions(root, created["id"], {first_id: "remove me"})
            removed = datasets.remove_images(root, created["id"], [first_id])

            self.assertEqual(removed["remove_result"], {"removed": 1})
            self.assertEqual(removed["dataset_revision"], 3)
            self.assertEqual([item["id"] for item in removed["images"]], ["image-00002"])
            images_dir = Path(root) / created["id"] / "images"
            self.assertFalse((images_dir / "image-00001.png").exists())
            self.assertFalse((images_dir / "image-00001.txt").exists())

    def test_remove_images_keeps_one_and_rolls_back_on_manifest_failure(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root,
                "Safe curation",
                "object",
                "object_x",
                [("first.png", _image_file("red")), ("second.png", _image_file("blue"))],
            )
            first_id = created["images"][0]["id"]
            with patch.object(datasets, "_save_manifest", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    datasets.remove_images(root, created["id"], [first_id])
            stored = datasets.get_dataset(root, created["id"])
            self.assertEqual(len(stored["images"]), 2)
            self.assertTrue(
                (Path(root) / created["id"] / "images" / "image-00001.png").is_file()
            )
            with self.assertRaisesRegex(datasets.DatasetError, "Keep at least one"):
                datasets.remove_images(
                    root, created["id"], [item["id"] for item in created["images"]]
                )

    def test_dataset_settings_only_revision_training_relevant_changes(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Original", "person", "person_x", [("first.png", _image_file("red"))]
            )
            renamed = datasets.update_dataset_settings(
                root, created["id"], "Renamed", "person", "person_x"
            )
            self.assertEqual(renamed["name"], "Renamed")
            self.assertEqual(renamed["dataset_revision"], 1)
            self.assertFalse(renamed["settings_result"]["training_changed"])

            changed = datasets.update_dataset_settings(
                root, created["id"], "Renamed", "style", "paint_x"
            )
            self.assertEqual(changed["type"], "style")
            self.assertEqual(changed["trigger_word"], "paint_x")
            self.assertEqual(changed["dataset_revision"], 2)
            self.assertTrue(changed["settings_result"]["training_changed"])

            with self.assertRaisesRegex(datasets.DatasetError, "Unsupported dataset type"):
                datasets.update_dataset_settings(
                    root, created["id"], "Renamed", "unknown", "paint_x"
                )

    def test_add_images_rolls_back_when_any_upload_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            created = datasets.create_dataset(
                root, "Stable set", "style", "style_x", [("first.png", _image_file("red"))]
            )
            dataset_dir = Path(root) / created["id"]

            with self.assertRaisesRegex(datasets.DatasetError, "Invalid image"):
                datasets.add_images(
                    root,
                    created["id"],
                    [("valid.png", _image_file("blue")), ("broken.png", io.BytesIO(b"bad"))],
                )

            stored = datasets.get_dataset(root, created["id"])
            self.assertEqual(len(stored["images"]), 1)
            self.assertEqual(
                sorted(path.name for path in (dataset_dir / "images").iterdir()),
                ["image-00001.png"],
            )
            self.assertFalse(any(path.name.startswith(".staging-add-") for path in dataset_dir.iterdir()))

    def test_rejects_non_image_uploads_without_leaving_staging_data(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(datasets.DatasetError, "Invalid image"):
                datasets.create_dataset(
                    root, "Bad", "style", "", [("notes.txt", io.BytesIO(b"hello"))]
                )
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_background_caption_job_persists_generated_caption(self):
        with tempfile.TemporaryDirectory() as root:
            result = datasets.create_dataset(
                root,
                "Car",
                "vehicle",
                "mycar",
                [("car.png", _image_file("green"))],
            )
            job_id = "caption-test"
            with main._dataset_caption_lock:
                main._dataset_caption_jobs[job_id] = {
                    "job_id": job_id,
                    "dataset_id": result["id"],
                    "status": "queued",
                }
                main._dataset_caption_cancel_events[job_id] = threading.Event()

            with (
                patch.object(main, "_datasets_root", return_value=root),
                patch.object(
                    main.caption,
                    "auto_caption",
                    return_value=[{"caption": "a green sports car", "source": "blip"}],
                ),
            ):
                main._run_dataset_caption_job(job_id, result["id"])

            manifest = datasets.get_dataset(root, result["id"])
            self.assertEqual(manifest["images"][0]["caption"], "a green sports car")
            self.assertEqual(main._dataset_caption_jobs[job_id]["status"], "done")

    def test_authenticated_dataset_api_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            main.app.dependency_overrides[main.get_current_user] = lambda: "tester"
            client = TestClient(main.app)
            try:
                with patch.object(main, "_datasets_root", return_value=root):
                    created_response = client.post(
                        "/api/datasets",
                        data={
                            "name": "Jacket",
                            "dataset_type": "clothing",
                            "trigger_word": "blue_jacket",
                        },
                        files={"files": ("jacket.png", _image_file("navy"), "image/png")},
                    )
                    self.assertEqual(created_response.status_code, 200)
                    created = created_response.json()

                    settings_response = client.put(
                        f"/api/datasets/{created['id']}",
                        json={
                            "name": "Jacket study",
                            "dataset_type": "clothing",
                            "trigger_word": "blue_jacket",
                        },
                    )
                    self.assertEqual(settings_response.status_code, 200, settings_response.text)
                    self.assertEqual(settings_response.json()["name"], "Jacket study")

                    listed = client.get("/api/datasets").json()["datasets"]
                    self.assertEqual([item["id"] for item in listed], [created["id"]])

                    image_id = created["images"][0]["id"]
                    update = client.put(
                        f"/api/datasets/{created['id']}/captions",
                        json={"captions": {image_id: "a blue jacket on a mannequin"}},
                    )
                    self.assertEqual(update.status_code, 200)
                    self.assertEqual(update.json()["analysis"]["captioned_count"], 1)

                    image_response = client.get(
                        f"/api/datasets/{created['id']}/images/{image_id}"
                    )
                    self.assertEqual(image_response.status_code, 200)
                    self.assertTrue(image_response.headers["content-type"].startswith("image/"))

                    add_response = client.post(
                        f"/api/datasets/{created['id']}/images",
                        files=[
                            ("files", ("duplicate.png", _image_file("navy"), "image/png")),
                            ("files", ("new.png", _image_file("white"), "image/png")),
                        ],
                    )
                    self.assertEqual(add_response.status_code, 200, add_response.text)
                    self.assertEqual(
                        add_response.json()["import_result"],
                        {"added": 1, "duplicates": 1},
                    )
                    self.assertEqual(add_response.json()["analysis"]["image_count"], 2)

                    remove_response = client.request(
                        "DELETE",
                        f"/api/datasets/{created['id']}/images",
                        json={"image_ids": [image_id]},
                    )
                    self.assertEqual(remove_response.status_code, 200, remove_response.text)
                    self.assertEqual(remove_response.json()["remove_result"], {"removed": 1})
                    self.assertEqual(remove_response.json()["analysis"]["image_count"], 1)
            finally:
                main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()

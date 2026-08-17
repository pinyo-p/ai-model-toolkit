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
            finally:
                main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()

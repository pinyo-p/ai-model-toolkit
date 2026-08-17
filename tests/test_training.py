import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

import main
from core import datasets, training


def _image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (768, 768), color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _ready_manifest(image_count=12, dataset_type="person"):
    return {
        "id": "dataset-1",
        "name": "Dataset",
        "type": dataset_type,
        "trigger_word": "sks subject",
        "images": [
            {
                "id": f"image-{index:05d}",
                "filename": f"image-{index:05d}.png",
                "caption": f"caption {index}",
            }
            for index in range(1, image_count + 1)
        ],
        "analysis": {"image_count": image_count, "captioned_count": image_count},
    }


class TrainingRecipeTests(unittest.TestCase):
    def test_balanced_recipe_hides_official_krea2_defaults(self):
        recipe = training.build_krea2_recipe(_ready_manifest(), "balanced", seed=7)

        self.assertEqual(recipe.training_model, "krea/Krea-2-Raw")
        self.assertEqual(recipe.inference_model, "krea/Krea-2-Turbo")
        self.assertEqual((recipe.rank, recipe.lora_alpha), (32, 32))
        self.assertEqual(recipe.learning_rate, 3e-4)
        self.assertEqual(recipe.instance_prompt, "a photo of sks subject person")
        self.assertTrue(recipe.cache_latents)
        self.assertTrue(recipe.gradient_checkpointing)
        self.assertEqual(recipe.seed, 7)

    def test_quick_or_uncaptioned_dataset_is_not_trainable(self):
        quick = _ready_manifest(image_count=5)
        with self.assertRaisesRegex(training.TrainingError, "need expansion"):
            training.build_krea2_recipe(quick)

        missing = _ready_manifest()
        missing["analysis"]["captioned_count"] = 11
        with self.assertRaisesRegex(training.TrainingError, "Every image needs a caption"):
            training.build_krea2_recipe(missing)

    def test_missing_trigger_uses_a_stable_dataset_trigger(self):
        manifest = _ready_manifest()
        manifest["trigger_word"] = ""
        manifest["id"] = "linen-shirt-a1b2c3d4"

        recipe = training.build_krea2_recipe(manifest)

        self.assertEqual(recipe.instance_prompt, "a photo of linen_shirt_a1b2c3d4 person")

    def test_seed_range_is_validated_server_side(self):
        with self.assertRaisesRegex(training.TrainingError, "Seed must be between"):
            training.build_krea2_recipe(_ready_manifest(), seed=-1)

    def test_style_export_keeps_content_caption_and_appends_style_anchor(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_dir = Path(root) / "style-set"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "image-00001.png").write_bytes(_image_bytes("orange").getvalue())
            manifest = _ready_manifest(image_count=1, dataset_type="style")
            manifest["images"] = [{
                "id": "image-00001",
                "filename": "image-00001.png",
                "caption": "an astronaut beside a rover",
            }]
            manifest["analysis"] = {"image_count": 6, "captioned_count": 6}
            recipe = training.build_krea2_recipe(manifest)

            result = training.prepare_imagefolder(dataset_dir, manifest, recipe)
            record = json.loads((result / "metadata.jsonl").read_text().strip())
            self.assertEqual(
                record,
                {
                    "file_name": "image-00001.png",
                    "text": "an astronaut beside a rover, sks subject",
                },
            )

            manifest["trigger_word"] = ""
            manifest["id"] = "retro-style-a1b2c3d4"
            fallback_recipe = training.build_krea2_recipe(manifest)
            training.prepare_imagefolder(dataset_dir, manifest, fallback_recipe)
            fallback_record = json.loads((result / "metadata.jsonl").read_text().strip())
            self.assertEqual(fallback_record["text"], "an astronaut beside a rover, retro_style_a1b2c3d4")

    def test_command_uses_pinned_official_trainer_and_safe_argument_list(self):
        recipe = training.build_krea2_recipe(_ready_manifest(), "quality")
        with patch.object(training, "accelerate_executable", return_value="/venv/bin/accelerate"):
            command = training.build_krea2_command(
                "/vendor/trainer.py", "/dataset/images", "/models/lora/run", recipe
            )

        self.assertEqual(command[:4], ["/venv/bin/accelerate", "launch", "--num_processes=1", "/vendor/trainer.py"])
        self.assertIn("--use_aspect_ratio_buckets", command)
        self.assertIn("--skip_final_inference", command)
        self.assertIn("to_q,to_k,to_v,to_out.0,to_gate", command)

    def test_progress_parser_reads_tqdm_step_and_loss(self):
        parsed = training.parse_training_progress(
            "Steps:  12%|## | 120/1000 [01:20<10:00, loss=0.123, lr=0.0003]"
        )
        self.assertEqual(parsed, {"completed_steps": 120, "total_steps": 1000, "loss": 0.123})


class TrainingJobTests(unittest.TestCase):
    def test_runner_publishes_completed_lora(self):
        colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as models_root:
            created = datasets.create_dataset(
                root,
                "Subject",
                "person",
                "sks subject",
                [(f"{index}.png", _image_bytes(color)) for index, color in enumerate(colors)],
            )
            captions = {item["id"]: f"portrait {index}" for index, item in enumerate(created["images"])}
            datasets.update_captions(root, created["id"], captions)
            job_id = "training-test"
            with main._training_lock:
                main._training_jobs[job_id] = {
                    "job_id": job_id,
                    "dataset_id": created["id"],
                    "dataset_name": "Subject",
                    "profile": "fast",
                    "status": "queued",
                    "created_at": "now",
                    "logs": [],
                }
                main._training_cancel_events[job_id] = threading.Event()

            fake_process = Mock()
            fake_process.stdout = io.StringIO("Steps: 100%|###| 500/500 [loss=0.25]\n")
            fake_process.wait.return_value = 0
            fake_process.poll.return_value = 0

            def fake_lora(output_dir):
                path = Path(output_dir) / "pytorch_lora_weights.safetensors"
                path.write_bytes(b"weights")
                return path

            with (
                patch.object(main, "_datasets_root", return_value=root),
                patch.dict(main.settings, {"models_path": models_root}),
                patch.object(main.training, "ensure_krea2_trainer", return_value=Path("/trainer.py")),
                patch.object(main.training, "build_krea2_command", return_value=["accelerate"]),
                patch.object(main.training, "find_final_lora", side_effect=fake_lora),
                patch.object(main.subprocess, "Popen", return_value=fake_process),
            ):
                main._run_krea2_training(job_id, created["id"], "fast", 42)

            job = main._training_jobs[job_id]
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["completed_steps"], 500)
            self.assertTrue(job["lora_path"].endswith("pytorch_lora_weights.safetensors"))
            manifest = datasets.get_dataset(root, created["id"])
            self.assertEqual(manifest["training_runs"][-1]["status"], "done")


if __name__ == "__main__":
    unittest.main()

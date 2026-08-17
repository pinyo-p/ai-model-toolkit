import threading
import unittest
from unittest.mock import patch

import main


class EvaluationJobTests(unittest.TestCase):
    def setUp(self):
        with main._evaluation_lock:
            main._evaluation_progress.clear()
        with main._evaluation_cancel_lock:
            main._evaluation_cancel_events.clear()

    def test_manifest_assigns_same_seed_set_to_all_variants(self):
        variants = [
            {"label": "Base", "model_path": "model", "steps": 20, "cfg": 7.0},
            {"label": "LoRA", "model_path": "model", "steps": 20, "cfg": 7.0},
        ]

        manifest = main._comparison_manifest(
            ["portrait", "full body"], variants, "", 1024, 1024, 2, 50
        )

        self.assertEqual(manifest["seeds_by_prompt"], [[50, 51], [52, 53]])
        self.assertEqual(len(manifest["variants"]), 2)

    def test_background_evaluation_publishes_result(self):
        evaluation_id = "evaluation-test"
        variants = [
            {"label": "Base", "model_path": "model", "steps": 20, "cfg": 7.0}
        ]
        manifest = main._comparison_manifest(
            ["portrait"], variants, "", 512, 512, 1, 42
        )
        with main._evaluation_cancel_lock:
            main._evaluation_cancel_events[evaluation_id] = threading.Event()

        fake_cells = [{
            "x": 0,
            "y": 0,
            "images": [],
            "seeds": [42],
            "steps": 20,
            "cfg": 7.0,
        }]

        def fake_generate(*args, **kwargs):
            kwargs["progress_cb"](1, 1)
            return fake_cells, ["Base"]

        with patch.object(main.sdxl, "comparison_generate", side_effect=fake_generate):
            main._run_comparison_evaluation(
                evaluation_id, ["portrait"], variants, "", 512, 512, 1, 42, manifest
            )

        with main._evaluation_lock:
            progress = main._evaluation_progress[evaluation_id]
        self.assertEqual(progress["status"], "done")
        self.assertEqual(progress["completed"], 1)
        self.assertEqual(progress["result"]["cells"][0]["seeds"], [42])
        self.assertEqual(progress["result"]["experiment"]["base_seed"], 42)


if __name__ == "__main__":
    unittest.main()

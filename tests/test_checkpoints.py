import gc
import unittest
from pathlib import Path

import torch

from app.evaluation.unified import METHOD_CHECKPOINTS, load_classifier


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_all_required_checkpoints_load_and_emit_four_logits(self) -> None:
        checkpoint_dir = Path(__file__).resolve().parents[1] / "app/models"
        sample = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        for method, filename in METHOD_CHECKPOINTS.items():
            with self.subTest(method=method):
                model = load_classifier(checkpoint_dir / filename, torch.device("cpu"))
                with torch.no_grad():
                    output = model(sample)
                self.assertEqual(tuple(output.shape), (1, 4))
                del model
                gc.collect()


if __name__ == "__main__":
    unittest.main()

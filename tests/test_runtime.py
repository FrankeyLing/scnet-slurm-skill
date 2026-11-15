import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNTIME = (Path(__file__).resolve().parents[1] / "skill" / "scnet-slurm" /
           "scripts" / "runtime.py")


class RuntimeIntegrationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)

    def train(self, name, progress, code, extra=()):
        self.marker = self.work / (name + ".marker.json")
        self.receipt = self.work / (name + ".receipt.json")
        return subprocess.run(
            [sys.executable, str(RUNTIME), "run", "--marker", str(self.marker),
             "--out", str(self.receipt), "--progress", str(progress), *extra,
             "--", sys.executable, "-c",
             "import os, sys\nfrom pathlib import Path\n"
             "sys.path.insert(0, {!r})\n".format(str(RUNTIME.parent)) + code],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )

    def test_zero_comes_with_a_fresh_verified_marker(self):
        missing = self.train("missing", 1, "pass")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("without a completion marker", missing.stderr)
        self.assertFalse(self.receipt.exists())

        code = "from runtime import mark_complete\nmark_complete(1)"
        successful = self.train("valid", 1, code)
        self.assertEqual(successful.returncode, 0, successful.stderr)
        self.assertEqual(json.loads(self.receipt.read_text(encoding="utf-8"))["progress"], 1)

        duplicate = self.train("valid", 1, code)
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("refusing to reuse", duplicate.stderr)

    def test_child_failure_cannot_turn_into_success(self):
        result = self.train("failed", 1, "from runtime import mark_complete\n"
                            "mark_complete(1)\nsys.exit(23)")
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertTrue(self.marker.exists())
        self.assertFalse(self.receipt.exists())

    def test_resume_requires_exact_checkpoint_attested_by_prior_progress(self):
        checkpoint = self.work / "segment.pt"
        first = self.train(
            "segment", 2,
            "from runtime import mark_complete\n"
            "path = Path(os.environ['SCNET_CHECKPOINT_OUT'])\n"
            "path.write_bytes(b'completed step 2')\nmark_complete(2, path)",
            ("--checkpoint-out", str(checkpoint)),
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        prior = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(prior["checkpoint"]["sha256"],
                         hashlib.sha256(b"completed step 2").hexdigest())
        extra = ("--checkpoint-in", str(checkpoint), "--previous-receipt", str(self.receipt),
                 "--previous-progress", "2")
        code = ("from runtime import mark_complete\n"
                "if Path(os.environ['SCNET_CHECKPOINT_IN']).read_bytes() != b'completed step 2':"
                "\n    sys.exit(17)\n"
                "mark_complete(4)")
        result = self.train("resume", 4, code, extra)
        self.assertEqual(result.returncode, 0, result.stderr)

        checkpoint.write_bytes(b"changed checkpoint")
        result = self.train("modified", 4, "print('child launched')", extra)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match its receipt", result.stderr)
        self.assertNotIn("child launched", result.stdout)
        self.assertFalse(self.receipt.exists())

    def test_checkpoint_changed_after_marker_prevents_receipt(self):
        result = self.train(
            "changed", 3,
            "from runtime import mark_complete\n"
            "path = Path(os.environ['SCNET_CHECKPOINT_OUT'])\n"
            "path.write_bytes(b'good')\nmark_complete(3, path)\n"
            "path.write_bytes(b'different')",
            ("--checkpoint-out", str(self.work / "out.pt")),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match its completion marker", result.stderr)
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()

"""Exercise rendering and the submit boundary without contacting a scheduler."""

import hashlib
import json
import os
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill/scnet-slurm/scripts"))
import scnet


class FakeScheduler:
    def __init__(self, replies=(), cluster="fixture", accounting=""):
        self.replies = iter(replies)
        self.cluster = cluster
        self.accounting = accounting
        self.calls = []
        self.timeouts = []

    @property
    def real_submits(self):
        return [call for call in self.calls if "--parsable" in call]

    def __call__(self, argv, timeout=30, check=True):
        self.calls.append(argv)
        self.timeouts.append((argv, timeout, check))
        if argv == ["scontrol", "show", "config"]:
            stdout = "ClusterName = " + self.cluster
        elif argv[:4] == ["scontrol", "-o", "show", "partition"]:
            stdout = "PartitionName=short State=UP MaxTime=02:00:00"
        elif argv[0] == "sinfo":
            stdout = "short|02:00:00|dcu:Hygon:4(S:0-3)|32|126500|mix|2"
        elif argv[0] == "bash" and argv[1] == "-n":
            stdout = ""
        elif argv[0] == "sbatch" and "--test-only" in argv:
            stdout = "Test-only accepted"
        elif argv[0] == "sbatch" and "--parsable" in argv:
            stdout = next(self.replies)
        elif argv[0] == "sacct":
            stdout = self.accounting
        elif argv[0] in ("squeue", "sshare"):
            stdout = ""
        else:
            raise AssertionError("Unexpected scheduler call: " + repr(argv))
        return {"stdout": stdout, "stderr": "", "returncode": 0}


class SubmissionTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        directory = Path(scratch.name)
        route = {"center": "Fixture", "cluster": "fixture", "partition": "short", "cards": 2}
        candidate = dict(route, id="fixture", wall_minutes=120, queue_minutes=2, setup_minutes=0,
                         startup_minutes=7, seconds_per_step=1, checkpoint_margin_minutes=8)
        manifest = {
            "identity": {"revision": "test-revision", "data": "test-data", "initial_weights": "test-weights",
                         "training_recipe": "test-recipe", "runtime": "test-runtime"},
            "total_steps": 10000,
            "jobs": {"fixture": {"launch": {
                "route": route, "workdir": "/fixture/project", "python": "python3",
                "setup_file": None, "entrypoint": "/fixture/project/train.py",
                "args": ["--stop", "{end_step}", "--target", "{total_steps}",
                         "--name", "BUNDLE/$(echo SHOULD_STAY_LITERAL)"],
                "resume_args": ["--resume", "{checkpoint_in}"],
                "checkpoint_args": ["--save", "{checkpoint_out}"],
                "cpus_per_task": 8, "memory_mib": 4096, "threads_per_rank": 2,
                "probe_timeout_seconds": 60, "gres": "dcu", "env": {},
            }, "files": {"/fixture/project/train.py": "a" * 64}}},
        }
        content = json.dumps(manifest).encode("utf-8")
        (directory / "workload.json").write_bytes(content)
        spec = {
            "schema_version": 1, "framework": "mature", "purpose": "formal",
            "workload_manifest": "workload.json", "workload_sha256": hashlib.sha256(content).hexdigest(),
            "total_steps": 10000, "supports_resume": True, "candidates": [candidate], "evidence": [],
        }
        spec_path = directory / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        self.bundle = directory / "bundle"
        self.plan = scnet.render(spec_path, self.bundle)
        # These fixtures use Linux paths even when running offline tests on Windows.
        for scope in (patch.object(scnet, "os", types.SimpleNamespace(name="posix", fsync=os.fsync)),
                      patch.object(scnet, "verify_files")):
            scope.start()
            self.addCleanup(scope.stop)

    def test_two_segments_preserve_horizon_and_quote_user_arguments(self):
        self.assertEqual(self.plan["selected"]["segments"], [6300, 10000])
        first, second = [(self.bundle / name).read_bytes() for name in self.plan["scripts"]]
        for script in (first, second):
            self.assertTrue(script.startswith(b"#!/bin/bash\n"))
            self.assertNotIn(b"\r", script)
            self.assertIn(b"'BUNDLE/$(echo SHOULD_STAY_LITERAL)'", script)
            self.assertIn(b"--target 10000", script)
        self.assertNotIn(b"--resume ", first)
        self.assertIn(b"--resume ", second)
        self.assertIn(b"--previous-progress 6300", second)

    def test_dry_run_then_guarded_chain_and_duplicate_rejection(self):
        scheduler = FakeScheduler(["1001", "1002"])
        with patch.object(scnet, "runner", scheduler):
            scnet.preflight(self.bundle, [1, 2])
            self.assertEqual(scheduler.real_submits, [])
            results = scnet.submit(self.bundle, None, True)
            self.assertEqual([result["job_id"] for result in results], ["1001", "1002"])
            self.assertIn("--dependency=afterok:1001", scheduler.real_submits[1])
            self.assertIn("--kill-on-invalid-dep=yes", scheduler.real_submits[1])
            with self.assertRaisesRegex(ValueError, "fresh bundle"):
                scnet.submit(self.bundle, None, True)
            self.assertEqual(len(scheduler.real_submits), 2)

    def test_wrong_cluster_and_tampered_bundle_stop_before_scheduler_submit(self):
        scheduler = FakeScheduler(cluster="other")
        with patch.object(scnet, "runner", scheduler):
            with self.assertRaisesRegex(ValueError, "ClusterName"):
                scnet.preflight(self.bundle, [1])
            self.assertFalse(any(call[0] == "sbatch" for call in scheduler.calls))
            (self.bundle / self.plan["scripts"][0]).write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                scnet.preflight(self.bundle, [1])

    def test_ambiguous_submit_is_recorded_and_cannot_be_retried(self):
        scheduler = FakeScheduler(["unexpected output"])
        with patch.object(scnet, "runner", scheduler):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                scnet.submit(self.bundle, 1, False)
            receipt = self.bundle / "submissions" / "001.json"
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["state"], "intent")
            with self.assertRaises(FileExistsError):
                scnet.submit(self.bundle, 1, False)
            self.assertEqual(len(scheduler.real_submits), 1)

    def test_manual_later_segment_requires_completed_job_and_matching_checkpoint(self):
        scheduler = FakeScheduler(["1001", "1002"])
        with patch.object(scnet, "runner", scheduler):
            scnet.submit(self.bundle, 1, False)
            with self.assertRaisesRegex(ValueError, "not confirmed completed"):
                scnet.submit(self.bundle, 2, False)
            scheduler.accounting = "1001|COMPLETED|0:0\n"
            checkpoint = self.bundle / "artifacts" / "001.checkpoint.pt"
            payload = b"checkpoint fixture"
            checkpoint.write_bytes(payload)
            prior = self.bundle / "artifacts" / "001.receipt.json"
            receipt = {"schema": 1, "kind": "training", "status": "passed", "progress": 6300,
                       "slurm_job_id": "9900", "checkpoint": {
                           "path": str(checkpoint), "sha256": hashlib.sha256(payload).hexdigest(),
                           "bytes": len(payload),
                       }}
            prior.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "job ID does not match accounting"):
                scnet.submit(self.bundle, 2, False)
            receipt["slurm_job_id"] = "1001"
            prior.write_text(json.dumps(receipt), encoding="utf-8")
            checkpoint.write_bytes(b"unexpected bytes")
            with self.assertRaisesRegex(ValueError, "does not match its receipt"):
                scnet.submit(self.bundle, 2, False)
            self.assertEqual(len(scheduler.real_submits), 1)
            checkpoint.write_bytes(payload)
            second = scnet.submit(self.bundle, 2, False, scheduler_timeout_seconds=120)
            self.assertEqual(second[0]["job_id"], "1002")
            self.assertNotIn("--dependency=afterok:1001", scheduler.real_submits[1])
            self.assertEqual([wait for argv, wait, _ in scheduler.timeouts if argv[0] == "sacct"][-1], 120)

    def test_explicit_timeout_reaches_inspect_preflight_submit_and_status(self):
        scheduler = FakeScheduler(["1201"])
        with patch.object(scnet, "runner", scheduler):
            self.assertEqual(scnet.inspect_cluster("short", 120)["scheduler_timeout_seconds"], 120)
            reports = scnet.preflight(self.bundle, [1, 2], 120)[1]
            self.assertTrue(all(row["scheduler_timeout_seconds"] == 120 for row in reports))
            self.assertEqual(scheduler.real_submits, [])
            result = scnet.submit(self.bundle, 1, False, 120)
            self.assertEqual(result[0]["scheduler_timeout_seconds"], 120)
            self.assertEqual(scnet.status("1201", 120)["scheduler_timeout_seconds"], 120)
        for argv, wait, _ in scheduler.timeouts:
            self.assertEqual(wait, 30 if argv[0] == "bash" else 120, argv)
        self.assertEqual(len(scheduler.real_submits), 1)
        recorded = json.loads((self.bundle / "submissions/001.json").read_text(encoding="utf8"))
        self.assertEqual(recorded["scheduler_timeout_seconds"], 120)

    def test_invalid_timeout_stops_before_files_or_scheduler(self):
        with patch.object(scnet, "runner") as run, patch.object(scnet, "staged") as staged:
            for value in (0, -1, 601, 1.5, True, None, "120"):
                for operation in (lambda: scnet.live_cluster(value),
                                  lambda: scnet.inspect_cluster("short", value),
                                  lambda: scnet.preflight(self.bundle, [1], value),
                                  lambda: scnet.submit(self.bundle, 1, False, value),
                                  lambda: scnet.status("1201", value)):
                    with self.subTest(timeout=value), self.assertRaisesRegex(ValueError, "1 and 600"):
                        operation()
            run.assert_not_called()
            staged.assert_not_called()

    def test_cli_timeout_validation_and_forwarding(self):
        commands = ((["inspect", "--partition", "short"], "inspect_cluster"),
                    (["preflight", str(self.bundle)], "preflight"),
                    (["submit", str(self.bundle), "--segment", "1", "--execute"], "submit"),
                    (["status", "1201"], "status"))
        for arguments, target in commands:
            with self.subTest(command=arguments[0]), patch.object(sys, "argv", ["scnet.py"]+arguments+[
                    "--scheduler-timeout-seconds", "0"]), patch.object(scnet, "runner") as run, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    scnet.main()
                self.assertEqual(stopped.exception.code, 2)
                run.assert_not_called()
            # Main must pass the CLI value, rather than silently using the API default.
            with self.subTest(command=arguments[0]), patch.object(sys, "argv", ["scnet.py"]+arguments+[
                    "--scheduler-timeout-seconds", "120"]), patch.object(scnet, target,
                    return_value=(None, None) if target == "preflight" else None) as entry:
                scnet.main()
                self.assertEqual(entry.call_args[0][-1], 120)

    def test_preflight_timeout_has_no_intent_or_real_submission_and_no_retry(self):
        scheduler = FakeScheduler(["should-not-submit"])
        def stalled(argv, timeout=30, check=True):
            if argv[0] == "sbatch" and "--test-only" in argv:
                scheduler.calls.append(argv)
                raise subprocess.TimeoutExpired(argv, timeout)
            return scheduler(argv, timeout, check)
        with patch.object(scnet, "runner", side_effect=stalled):
            with self.assertRaises(subprocess.TimeoutExpired) as failure:
                scnet.submit(self.bundle, 1, False, 120)
        self.assertEqual(failure.exception.timeout, 120)
        self.assertEqual(sum("--test-only" in argv for argv in scheduler.calls), 1)
        self.assertEqual(scheduler.real_submits, [])
        self.assertEqual(list((self.bundle / "submissions").iterdir()), [])

    def test_real_submit_timeout_retains_intent_and_blocks_second_mutation(self):
        scheduler = FakeScheduler()
        def stalled(argv, timeout=30, check=True):
            if argv[0] == "sbatch" and "--parsable" in argv:
                scheduler.calls.append(argv)
                raise subprocess.TimeoutExpired(argv, timeout)
            return scheduler(argv, timeout, check)
        with patch.object(scnet, "runner", side_effect=stalled):
            with self.assertRaises(subprocess.TimeoutExpired):
                scnet.submit(self.bundle, 1, False, 120)
            path = self.bundle / "submissions/001.json"
            intent = json.loads(path.read_text(encoding="utf8"))
            self.assertEqual(intent["state"], "intent")
            self.assertEqual(intent["scheduler_timeout_seconds"], 120)
            with self.assertRaises(FileExistsError):
                scnet.submit(self.bundle, 1, False, 120)
            self.assertEqual(json.loads(path.read_text(encoding="utf8")), intent)
        self.assertEqual(len(scheduler.real_submits), 1)

    def test_timeout_bounds_and_default_keep_checks_enabled(self):
        scheduler = FakeScheduler()
        with patch.object(scnet, "runner", scheduler):
            scnet.inspect_cluster("short", 1)
            scnet.inspect_cluster("short", 600)
            scnet.status("1201")
        self.assertEqual([wait for _, wait, _ in scheduler.timeouts], [1]*5+[600]*5+[30]*3)
        scheduler = FakeScheduler(cluster="wrong")
        with patch.object(scnet, "runner", scheduler), self.assertRaisesRegex(ValueError, "ClusterName"):
            scnet.preflight(self.bundle, [1], 120)
        self.assertEqual(scheduler.real_submits, [])


class RunnerTimeoutTest(unittest.TestCase):
    def test_selected_timeout_is_passed_to_subprocess_without_retry(self):
        reply = types.SimpleNamespace(returncode=0, stdout="ClusterName = fixture", stderr="")
        with patch.object(scnet.subprocess, "run", return_value=reply) as run:
            self.assertEqual(scnet.live_cluster(120), "fixture")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args[1]["timeout"], 120)


if __name__ == "__main__":
    unittest.main()

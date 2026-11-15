import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill" / "scnet-slurm" / "scripts"))
from policy import DDP_CHECKS, RESUME_CHECKS, plan  # noqa: E402


WORKLOAD_HASH = "a" * 64


def candidate(**changes):
    return {
        "id": "ks-short", "center": "昆山", "cluster": "ks-example", "partition": "short",
        "cards": 1, "wall_minutes": 120, "queue_minutes": 5, "setup_minutes": 10,
        "startup_minutes": 4, "seconds_per_step": 6, "checkpoint_margin_minutes": 6,
        **changes,
    }


def spec(**changes):
    return {
        "schema_version": 1,
        "framework": "custom", "purpose": "formal", "workload_sha256": WORKLOAD_HASH,
        "supports_resume": True, "total_steps": 1000, "evidence": [], "candidates": [candidate()],
        **changes,
    }


def evidence(kind, route, **changes):
    return {
        "type": kind, "workload_sha256": WORKLOAD_HASH, "passed": True,
        "artifact_sha256": "b" * 64, "artifact": "audit-{}.json".format(kind),
        "checks": sorted(DDP_CHECKS if kind == "ddp" else RESUME_CHECKS),
        **{field: route[field] for field in ("center", "cluster", "partition", "cards")},
        **changes,
    }


class PolicyTest(unittest.TestCase):
    def test_two_hour_path_with_accepted_resume_beats_long_queue(self):
        fast = candidate()
        slow = candidate(id="hb-long", center="哈尔滨", cluster="hb-example", partition="long",
                         wall_minutes=720, queue_minutes=240, seconds_per_step=4)
        job = spec(total_steps=2500, candidates=[slow, fast], evidence=[evidence("resume", fast)])
        chosen = plan(job)["selected"]
        self.assertEqual(chosen["id"], fast["id"])
        self.assertEqual(chosen["segments"], [1100, 2200, 2500])
        self.assertEqual(chosen["requested_minutes"], [120, 120, 40])
        self.assertEqual(chosen["estimated_total_queue_minutes"], 15)
        self.assertEqual(chosen["eta_minutes"], 305)
        no_evidence = plan({**job, "evidence": []})
        self.assertEqual(no_evidence["selected"]["id"], slow["id"])
        self.assertIn("resume acceptance", no_evidence["alternatives"][0]["reason"])

    def test_custom_ddp_and_resume_are_independent(self):
        route = candidate(cards=4)
        base = spec(total_steps=1500, candidates=[route])
        ddp, resume = evidence("ddp", route), evidence("resume", route)
        for records, message in (([ddp], "resume acceptance"), ([resume], "ddp acceptance")):
            result = plan({**base, "evidence": records})
            self.assertIsNone(result["selected"])
            self.assertIn(message, result["alternatives"][0]["reason"])
        self.assertIsNotNone(plan({**base, "evidence": [ddp, resume]})["selected"])

    def test_evidence_does_not_carry_between_workloads_centers_queues_or_card_counts(self):
        route = candidate(cards=4)
        base = spec(candidates=[route])
        for changes in ({"workload_sha256": "c" * 64}, {"center": "哈尔滨"},
                        {"cluster": "other"}, {"partition": "long"}, {"cards": 2}):
            with self.subTest(changes=changes):
                result = plan({**base, "evidence": [evidence("ddp", route, **changes)]})
                self.assertIsNone(result["selected"])
        failed = evidence("ddp", route, passed=False, checks=["gradients"])
        result = plan({**base, "evidence": [evidence("ddp", route), failed]})
        self.assertIn("failing record", result["alternatives"][0]["reason"])

    def test_synthetic_or_single_step_evidence_cannot_claim_a_pass(self):
        route = candidate(cards=2)
        partial = evidence("ddp", route, checks=["gradients", "actual_topology"])
        with self.assertRaisesRegex(ValueError, "multi_step"):
            plan(spec(candidates=[route], evidence=[partial]))

    def test_custom_multi_card_smoke_is_one_short_nonresumable_job(self):
        route = candidate(cards=4)
        selected = plan(spec(purpose="smoke", total_steps=20, candidates=[route]))["selected"]
        self.assertEqual(selected["requested_minutes"], [12])
        over = plan(spec(purpose="smoke", total_steps=100, candidates=[route]))
        self.assertIsNone(over["selected"])
        self.assertIn("one job", over["alternatives"][0]["reason"])
        with self.assertRaisesRegex(ValueError, "smoke mode"):
            plan(spec(purpose="smoke", resume_existing=True, start_step=1))

    def test_mature_ddp_is_preferred_only_when_finish_is_comparable(self):
        single = candidate()
        multi = candidate(id="ks-four", cards=4, queue_minutes=10)
        job = spec(framework="mature", candidates=[single, multi])
        self.assertEqual(plan(job)["selected"]["id"], multi["id"])
        self.assertEqual(plan({**job, "max_card_hours": 2})["selected"]["id"], single["id"])
        multi["queue_minutes"] = 90
        self.assertEqual(plan(job)["selected"]["id"], single["id"])

    def test_resume_existing_keeps_total_target_and_requires_acceptance(self):
        route = candidate()
        resumed = spec(candidates=[route], resume_existing=True, start_step=1050, total_steps=2100)
        self.assertIsNone(plan(resumed)["selected"])
        resumed["evidence"] = [evidence("resume", route)]
        chosen = plan(resumed)["selected"]
        self.assertEqual(chosen["segments"], [2100])
        self.assertEqual(chosen["requested_minutes"], [115])
        resumed["supports_resume"] = False
        self.assertIsNone(plan(resumed)["selected"])

    def test_queue_and_completion_budgets_include_all_segments(self):
        route = candidate()
        job = spec(total_steps=2500, candidates=[route], evidence=[evidence("resume", route)])
        self.assertIsNone(plan({**job, "max_queue_minutes": 14})["selected"])
        self.assertIsNone(plan({**job, "max_eta_minutes": 304})["selected"])
        self.assertIsNone(plan({**job, "max_segments": 2})["selected"])

    def test_estimate_never_rounds_a_request_down(self):
        at_limit = candidate(wall_minutes=1, startup_minutes=0.1, checkpoint_margin_minutes=0.1,
                             seconds_per_step=0.2)
        feasible = spec(total_steps=240, candidates=[at_limit])
        self.assertEqual(plan(feasible)["selected"]["requested_minutes"], [1])
        self.assertIsNone(plan({**feasible, "total_steps": 241})["selected"])

    def test_refuses_unknown_fields_nonfinite_values_and_bool_as_integer(self):
        base = spec()
        for values in ({"max_eta_minute": 5}, {"total_steps": True}, {"max_eta_minutes": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                plan({**base, **values})
        copy_spec = copy.deepcopy(base)
        copy_spec["candidates"][0]["cards"] = True
        with self.assertRaises(ValueError):
            plan(copy_spec)


if __name__ == "__main__":
    unittest.main()

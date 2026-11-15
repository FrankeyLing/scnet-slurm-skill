#!/usr/bin/env python3
"""Generate a two-step smoke plan; run on the chosen login node from this repository."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("center", "cluster", "partition", "python", "runtime-id"):
        p.add_argument("--" + name, required=True)
    for name in ("cards", "wall-minutes", "cpus-per-task", "memory-mib", "probe-timeout-seconds"):
        p.add_argument("--" + name, type=int, required=True)
    p.add_argument("--setup-file", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.cards not in (1, 2, 4) or args.wall_minutes < 6 or not 0 < args.probe_timeout_seconds < 180:
        p.error("the demo requires 1, 2 or 4 cards, at least 6 wall minutes and 1–179 probe seconds")
    if args.cpus_per_task < args.cards or args.memory_mib <= 0:
        p.error("provide at least one CPU core per card and positive memory")
    adapter = Path(__file__).resolve().with_name("train_demo.py")
    setup = args.setup_file.resolve() if args.setup_file else None
    files = {str(item): sha(item) for item in (adapter, setup) if item}
    route = {key: getattr(args, key) for key in ("center", "cluster", "partition", "cards")}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "workload.json", {
        "identity": {"revision": "demo:" + files[str(adapter)], "data": "synthetic global batches derived from step seed",
                     "initial_weights": "explicit seed 41", "training_recipe": "demo linear regression; 2 explicit optimizer steps",
                     "runtime": args.runtime_id},
        "total_steps": 2,
        "jobs": {"demo": {"files": files, "launch": {
            "route": route, "workdir": str(adapter.parent), "python": args.python,
            "setup_file": str(setup) if setup else None, "entrypoint": str(adapter),
            "args": ["--seed", "41", "--global-batch", "8", "--width", "16", "--lr", "0.01", "--start-step",
                     "{start_step}", "--end-step", "{end_step}", "--total-steps", "{total_steps}"],
            "resume_args": [], "checkpoint_args": [], "cpus_per_task": args.cpus_per_task, "memory_mib": args.memory_mib,
            "threads_per_rank": 1, "probe_timeout_seconds": args.probe_timeout_seconds, "gres": "dcu", "env": {},
        }}},
    })
    # Fixed demonstration estimates, not queue measurements; swap for same-run observations on real tasks.
    write(args.output / "spec.json", {
        "schema_version": 1, "framework": "custom", "purpose": "smoke", "supports_resume": False,
        "total_steps": 2, "evidence": [], "workload_manifest": "workload.json", "workload_sha256": sha(args.output / "workload.json"),
        "candidates": [dict(route, id="demo", wall_minutes=args.wall_minutes, queue_minutes=0, setup_minutes=0,
                            startup_minutes=4, seconds_per_step=1, checkpoint_margin_minutes=1)],
    })
    print(args.output / "spec.json")


if __name__ == "__main__":
    main()

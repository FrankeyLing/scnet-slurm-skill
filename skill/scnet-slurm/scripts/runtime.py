"""Compute-node checks and training receipts for a single Slurm allocation."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time


class RuntimeFailure(Exception):
    """A runtime contract was not satisfied."""


def require(condition, message):
    if not condition:
        raise RuntimeFailure(message)


def file_path(value):
    path = Path(value).expanduser()
    require(path.name not in ("", ".", ".."), "expected a file path: {}".format(value))
    return path.parent.resolve() / path.name


def require_absent(path):
    require(not os.path.lexists(str(path)), "refusing to reuse {}".format(path))


def regular_file(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size > 0,
            "expected a nonempty regular file: {}".format(path))
    return info


def new_json(path, data):
    """Publish an atomic, write-once file in its final directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".scnet-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A replace would let concurrent attempts overwrite one another.
        os.link(temporary, str(path))
    finally:
        os.unlink(temporary)


def read_json(path):
    require(regular_file(path).st_size <= 1024 * 1024,
            "JSON file exceeds 1 MiB: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    require(isinstance(result, dict), "expected a JSON object: {}".format(path))
    return result


def checkpoint_identity(path):
    def identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    before = regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before),
                "checkpoint changed before reading: {}".format(path))
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        require(identity(os.fstat(stream.fileno())) == identity(before),
                "checkpoint changed while reading: {}".format(path))
    require(identity(regular_file(path)) == identity(before),
            "checkpoint changed after reading: {}".format(path))
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": before.st_size}


def receipt(kind, **results):
    result = {
        "schema": 1,
        "kind": kind,
        "status": "passed",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    result.update(results)
    return result


def torch_devices(expected):
    # Keep the controller and marker usable with the standard library. The template
    # runs the dedicated hardware check before launching any training.
    import torch

    require(bool(getattr(torch.version, "hip", None)),
            "PyTorch has no HIP build identifier (CPU or NVIDIA build, or unsupported vendor build)")
    require(torch.cuda.is_available(), "HIP build cannot initialize a device")
    actual = torch.cuda.device_count()
    require(actual == expected,
            "expected {} visible devices, found {}".format(expected, actual))
    return torch


def device_probe(args):
    torch = torch_devices(args.cards)
    devices = []
    for index in range(args.cards):
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", None)
        if args.arch:
            require(isinstance(arch, str) and arch.split(":")[0] == args.arch,
                    "device {}: expected arch {}, found {}".format(index, args.arch, arch))
        left = torch.ones((16, 16), dtype=torch.float32, device=device, requires_grad=True)
        right = torch.full((16, 16), 2.0, dtype=torch.float32, device=device)
        value = left @ right
        value.sum().backward()
        torch.cuda.synchronize(index)
        require(value.device == device and left.grad.device == device and
                bool((value == 32).all().item()) and bool((left.grad == 32).all().item()),
                "unexpected accelerator matmul/backward output on {}".format(device))
        devices.append({
            "index": index, "name": props.name, "arch": arch,
            "total_memory_bytes": int(props.total_memory),
        })
        print("scnet: device {} passed forward/backward ({})".format(index, props.name),
              flush=True)
    new_json(file_path(args.out), {
        "torch": str(torch.__version__), "hip": str(torch.version.hip), "devices": devices,
    })
    return 0


def collective_probe(args):
    torch = torch_devices(args.cards)
    import torch.distributed as dist

    require(dist.is_available() and dist.is_nccl_available(),
            "PyTorch distributed NCCL backend unavailable (uses RCCL on HIP)")
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    require(world_size == args.cards and 0 <= local == rank < world_size and
            args.local_rank in (None, local), "unexpected single-node launcher topology")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=timedelta(seconds=args.timeout_seconds))
    value = torch.tensor([rank + 1], dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local)
    expect = world_size * (world_size + 1) // 2
    require(value.item() == expect,
            "rank {} received {}, expected {}".format(rank, value.item(), expect))
    dist.destroy_process_group()
    print("scnet: rank {} all-reduce passed".format(rank), flush=True)
    return 0


def bounded_probe(command, timeout, stage):
    # Keep the timeout outside of PyTorch to cover a stuck import, device initialization,
    # rendezvous, or kernel. A TERM allows the elastic agent to stop its workers.
    require(timeout > 0, "hardware probe deadline expired before {}".format(stage))
    process = subprocess.Popen(command, start_new_session=(os.name == "posix"))
    try:
        code = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        raise RuntimeFailure("{} interrupted or timed out".format(stage))
    require(code == 0, "{} exited with code {}".format(stage, code))


def probe(args):
    out = file_path(args.out)
    require_absent(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    deadline = time.monotonic() + args.timeout_seconds
    with tempfile.TemporaryDirectory(prefix=".scnet-probe-", dir=str(out.parent)) as directory:
        device_record = Path(directory) / "devices.json"
        command = [sys.executable, "-u", script, "_devices", "--cards", str(args.cards),
                   "--out", str(device_record)]
        if args.arch:
            command += ["--arch", args.arch]
        bounded_probe(command, deadline - time.monotonic(), "device probe")
        result = read_json(device_record)
        if args.cards > 1:
            remaining = deadline - time.monotonic()
            command = [
                sys.executable, "-u", "-m", "torch.distributed.run",
                "--nnodes=1", "--nproc_per_node={}".format(args.cards), "--max_restarts=0",
                "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
                "--rdzv_id={}".format(secrets.token_hex(12)),
                script, "_collective", "--cards", str(args.cards),
                "--timeout-seconds", str(max(1, math.ceil(remaining))),
            ]
            bounded_probe(command, remaining, "distributed collective probe")
        new_json(out, receipt("probe", cards=args.cards, **result,
                              collective="nccl" if args.cards > 1 else None))
    return 0


def mark_complete(progress, checkpoint=None):
    """Call from global rank zero after validating actual progress and saved state."""
    require(type(progress) is int and progress >= 1, "actual progress must be a positive integer")
    require(os.getenv("RANK", "0") == "0", "only global rank zero may mark completion")
    expected_progress = int(os.environ["SCNET_EXPECTED_PROGRESS"])
    require(progress == expected_progress,
            "expected completed progress {}, found {}".format(expected_progress, progress))
    expected_path = os.getenv("SCNET_CHECKPOINT_OUT")
    path = file_path(checkpoint) if checkpoint is not None else None
    require(path == (file_path(expected_path) if expected_path is not None else None),
            "marker checkpoint does not match the requested output path")
    new_json(file_path(os.environ["SCNET_COMPLETION_FILE"]), {
        "token": os.environ["SCNET_COMPLETION_TOKEN"],
        "progress": progress,
        "checkpoint": checkpoint_identity(path) if path else None,
    })


def verify_previous(checkpoint, previous_receipt, progress):
    checkpoint = file_path(checkpoint)
    prior = read_json(file_path(previous_receipt))
    require(type(prior.get("schema")) is int and prior["schema"] == 1 and
            prior.get("status") == "passed" and prior.get("kind") == "training" and
            type(prior.get("progress")) is int and prior["progress"] == progress,
            "predecessor receipt did not verify progress {}".format(progress))
    require(prior.get("checkpoint") == checkpoint_identity(checkpoint),
            "predecessor checkpoint does not match its receipt: {}".format(checkpoint))


def run_training(args):
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    require(bool(command), "missing command after --")

    marker, out = file_path(args.marker), file_path(args.out)
    checkpoint_out = file_path(args.checkpoint_out) if args.checkpoint_out else None
    checkpoint_in = file_path(args.checkpoint_in) if args.checkpoint_in else None
    previous = file_path(args.previous_receipt) if args.previous_receipt else None
    paths = [path for path in (marker, out, checkpoint_out, checkpoint_in, previous) if path]
    require(len(set(paths)) == len(paths), "marker, receipts and checkpoints must be distinct paths")
    if checkpoint_in or previous or args.previous_progress is not None:
        require(checkpoint_in and previous and args.previous_progress is not None,
                "--checkpoint-in, --previous-receipt and --previous-progress must be used together")
        require(args.previous_progress < args.progress, "progress must advance after resume")
        verify_previous(checkpoint_in, previous, args.previous_progress)
    for path in (marker, out, checkpoint_out):
        if path:
            require_absent(path)
            path.parent.mkdir(parents=True, exist_ok=True)

    token = secrets.token_hex(24)
    env = dict(os.environ)
    env.update(
        SCNET_COMPLETION_TOKEN=token,
        SCNET_COMPLETION_FILE=str(marker),
        SCNET_EXPECTED_PROGRESS=str(args.progress),
    )
    for key, value in {
        "SCNET_CHECKPOINT_IN": checkpoint_in, "SCNET_CHECKPOINT_OUT": checkpoint_out,
    }.items():
        env.pop(key, None)
        if value:
            env[key] = str(value)
    result = subprocess.run(command, env=env)
    if result.returncode:
        print("scnet: training child exited with {}".format(result.returncode), file=sys.stderr)
        return result.returncode if result.returncode > 0 else 128 - result.returncode

    require(marker.exists(), "training exited zero without a completion marker: {}".format(marker))
    data = read_json(marker)
    require(set(data) == {"token", "progress", "checkpoint"} and
            data["token"] == token and type(data["progress"]) is int and
            data["progress"] == args.progress, "invalid token or progress in completion marker")
    checkpoint = checkpoint_identity(checkpoint_out) if checkpoint_out else None
    require(data.get("checkpoint") == checkpoint,
            "final checkpoint does not match its completion marker")
    new_json(out, receipt("training", progress=args.progress, checkpoint=checkpoint))
    print("scnet: verified completion at progress {}".format(args.progress), flush=True)
    return 0


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    for name, fn in (("probe", probe), ("_devices", device_probe),
                     ("_collective", collective_probe)):
        sub = commands.add_parser(name)
        sub.add_argument("--cards", type=positive, required=True)
        if name in ("probe", "_devices"):
            sub.add_argument("--arch")
            sub.add_argument("--out", required=True)
        if name in ("probe", "_collective"):
            sub.add_argument("--timeout-seconds", required=True, type=positive)
        if name == "_collective":
            sub.add_argument("--local-rank", "--local_rank", type=int)
        sub.set_defaults(fn=fn)
    launch = commands.add_parser("run")
    launch.add_argument("--marker", required=True)
    launch.add_argument("--out", required=True)
    launch.add_argument("--progress", type=positive, required=True)
    launch.add_argument("--checkpoint-out")
    launch.add_argument("--checkpoint-in")
    launch.add_argument("--previous-receipt")
    launch.add_argument("--previous-progress", type=positive)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    launch.set_defaults(fn=run_training)
    mark = commands.add_parser("mark")
    mark.add_argument("--progress", type=positive, required=True)
    mark.add_argument("--checkpoint")
    mark.set_defaults(fn=lambda args: mark_complete(args.progress, args.checkpoint) or 0)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (RuntimeFailure, OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print("scnet: {}: {}".format(args.action, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

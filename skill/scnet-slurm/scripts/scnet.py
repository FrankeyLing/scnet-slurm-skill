#!/usr/bin/env python3
"""Plan, stage and submit one-node SCNet Slurm jobs without third-party packages."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

from policy import plan


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")
JOBID = re.compile(r"[1-9][0-9]*\Z")
ROUTE = ("center", "cluster", "partition", "cards")
MAX_LOG = 8192


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schema(obj, required, optional=()):
    require(isinstance(obj, dict), "expected an object")
    missing = set(required) - obj.keys()
    extra = obj.keys() - set(required) - set(optional)
    require(not (missing or extra), "missing keys: {}; unknown keys: {}".format(sorted(missing), sorted(extra)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_file(path, value):
    require(isinstance(value, str) and SHA.fullmatch(value), "expected SHA-256 for {}".format(path))
    require(Path(path).is_file() and digest(path) == value, "file missing or hash mismatch: {}".format(path))


def text(value):
    return isinstance(value, str) and bool(value) and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def posix_path(value):
    require(text(value) and value.startswith('/') and not value.startswith('//') and '..' not in PurePosixPath(value).parts,
            "expected an absolute Linux path without '..': {!r}".format(value))


def pairs(rows):
    obj = {}
    for key, val in rows:
        require(key not in obj, "duplicate JSON key: {}".format(key))
        obj[key] = val
    return obj


def bad_constant(value):
    raise ValueError("non-finite number in JSON: " + value)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)


def dump(path, data, mode="w"):
    with Path(path).open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def replace_json(path, data):
    tmp = path.with_suffix(".tmp")
    dump(tmp, data, "x")
    tmp.replace(path)


def job_schema(job, route):
    schema(job, ("route", "workdir", "python", "setup_file", "entrypoint", "args", "resume_args", "checkpoint_args",
                 "cpus_per_task", "memory_mib", "threads_per_rank", "probe_timeout_seconds", "gres", "env"),
           ("account", "qos", "arch"))
    require(job["route"] == {key: route[key] for key in ROUTE}, "manifest job route does not match candidate")
    for key in ("workdir", "entrypoint"):
        posix_path(job[key])
    if job["setup_file"] is not None:
        posix_path(job["setup_file"])
    require(text(job["python"]), "specify the site Python executable")
    if job["python"].startswith("/"):
        posix_path(job["python"])
    else:
        require(NAME.fullmatch(job["python"]), "invalid Python executable")
    for key in ("cpus_per_task", "memory_mib", "threads_per_rank", "probe_timeout_seconds"):
        require(type(job[key]) is int and job[key] > 0, "{} must be a positive integer".format(key))
    require(job["threads_per_rank"] * route["cards"] <= job["cpus_per_task"], "threads per rank oversubscribe requested CPUs")
    require(isinstance(job["gres"], str) and re.fullmatch(r"dcu(?::[A-Za-z0-9_.-]+)?", job["gres"]),
            "gres must be dcu or dcu:<type>, without count")
    for key in ("account", "qos", "arch"):
        if key in job:
            require(isinstance(job[key], str) and NAME.fullmatch(job[key]), "invalid " + key)
    placeholders = {"{start_step}", "{end_step}", "{total_steps}", "{checkpoint_in}", "{checkpoint_out}"}
    for key, allowed in (("args", placeholders - {"{checkpoint_in}", "{checkpoint_out}"}),
                         ("resume_args", {"{checkpoint_in}"}), ("checkpoint_args", {"{checkpoint_out}"})):
        require(isinstance(job[key], list), key + " must be an array of argument tokens")
        for arg in job[key]:
            require(text(arg), key + " must contain nonempty single-line strings")
            require(("{" not in arg and "}" not in arg) or arg in allowed, "invalid argument placeholder: " + arg)
    require(job["args"].count("{end_step}") == 1 and job["args"].count("{total_steps}") == 1,
            "args require {end_step} and {total_steps} as separate tokens")
    require(job["args"].count("{start_step}") <= 1, "duplicate {start_step}")
    for key, placeholder in (("resume_args", "{checkpoint_in}"), ("checkpoint_args", "{checkpoint_out}")):
        require(not job[key] or job[key].count(placeholder) == 1, "invalid " + key)
    require(isinstance(job["env"], dict), "env must be an object")
    for key, value in job["env"].items():
        require(re.fullmatch(r"[A-Z][A-Z0-9_]*", key) and text(value), "invalid environment assignment: " + key)
        require(not key.startswith(("SLURM_", "SBATCH_", "SCNET_", "PMI_", "MASTER_", "TORCHELASTIC_")) and
                key not in DIST_ENV and key not in ("PATH", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "BASH_ENV", "ENV",
                                                    "PYTHONPATH", "PYTHONHOME", "SHELLOPTS"), "reserved environment variable: " + key)
        require(not key.endswith("_VISIBLE_DEVICES"), "preserve scheduler device masks: " + key)


DIST_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT")


def inspect_spec(path):
    spec = load(path)
    require(isinstance(spec, dict) and type(spec.get("schema_version")) is int and spec["schema_version"] == 1, "set schema_version to 1")
    require(text(spec.get("workload_manifest")), "specify workload_manifest")
    manifest_path = path.parent / spec["workload_manifest"]
    checked_file(manifest_path, spec.get("workload_sha256"))
    workload = load(manifest_path)
    schema(workload, ("identity", "total_steps", "jobs"))
    schema(workload["identity"], ("revision", "data", "initial_weights", "training_recipe", "runtime"))
    require(all(text(x) for x in workload["identity"].values()), "provide explicit workload identities")
    require(type(workload["total_steps"]) is int and workload["total_steps"] == spec.get("total_steps"), "workload horizon differs from spec")
    result = plan(spec)
    require(isinstance(workload["jobs"], dict) and set(workload["jobs"]) == {c["id"] for c in spec["candidates"]},
            "workload jobs must match the candidate ids")
    for candidate in spec["candidates"]:
        job = workload["jobs"][candidate["id"]]
        require(isinstance(candidate["cluster"], str) and NAME.fullmatch(candidate["cluster"]) and
                isinstance(candidate["partition"], str) and NAME.fullmatch(candidate["partition"]), "invalid cluster or partition name")
        schema(job, ("launch", "files"))
        job_schema(job["launch"], candidate)
        require(isinstance(job["files"], dict) and job["launch"]["entrypoint"] in job["files"] and
                (job["launch"]["setup_file"] is None or job["launch"]["setup_file"] in job["files"]),
                "files must pin entrypoint and setup_file content by SHA-256")
        for filename, value in job["files"].items():
            posix_path(filename)
            require(isinstance(value, str) and SHA.fullmatch(value), "invalid file hash for " + filename)
    for evidence in spec["evidence"]:
        checked_file(path.parent / evidence["artifact"], evidence["artifact_sha256"])
    initial = spec.get("initial_checkpoint")
    if spec.get("resume_existing", False):
        schema(initial, ("path", "sha256"))
        posix_path(initial["path"])
        require(isinstance(initial["sha256"], str) and SHA.fullmatch(initial["sha256"]), "invalid initial checkpoint SHA-256")
    else:
        require(initial is None, "initial_checkpoint requires resume_existing")
    return spec, manifest_path, workload, result


class BundlePath(str):
    """An internal bundle-relative path, never taken from user input."""


def shell_arg(value):
    return '"$bundle/' + value + '"' if isinstance(value, BundlePath) else shlex.quote(value)


def shell_path(relative):
    return BundlePath(relative)


def render_segment(route, definition, spec, index, start_step, end_step, requested_minutes):
    job = definition["launch"]
    cards = route["cards"]
    header = {"job-name": "scnet-{}-{:03d}".format(route["id"], index), "partition": route["partition"],
              "nodes": 1, "ntasks": 1, "cpus-per-task": job["cpus_per_task"], "mem": str(job["memory_mib"]) + "M",
              "gres": job["gres"] + ":" + str(cards), "time": requested_minutes, "output": "logs/%x-%j.out",
              "error": "logs/%x-%j.err", "open-mode": "append", "export": "ALL"}
    for key in ("account", "qos"):
        if key in job:
            header[key] = job[key]
    lines = ["#!/bin/bash"] + ["#SBATCH --{}={}".format(key, val) for key, val in header.items()]
    lines += ["#SBATCH --no-requeue", "", "set -euo pipefail", 'bundle="$PWD"', 'test -r "$bundle/bundle.json"',
              ': "${SLURM_JOB_ID:?must be started by Slurm}"', 'unset ' + ' '.join(DIST_ENV),
              'printf "SCNET start job=%s node=%s\\n" "$SLURM_JOB_ID" "${HOSTNAME:-unknown}"']
    if job["setup_file"]:
        # Site module scripts sometimes reference optional unset variables; failing commands still fail the job.
        setup = shlex.quote(job["setup_file"])
        lines += ["setup_digest=$(sha256sum < " + setup + ")",
                  '[[ "${setup_digest%% *}" == ' + definition["files"][job["setup_file"]] + ' ]] || { echo "SCNET: setup file hash mismatch" >&2; exit 2; }',
                  "set +u", "source " + setup, "set -euo pipefail", 'unset ' + ' '.join(DIST_ENV)]
    for key, value in sorted(job["env"].items()):
        lines.append("export {}={}".format(key, shlex.quote(value)))
    lines += ["export PYTHONUNBUFFERED=1", "export OMP_NUM_THREADS=" + str(job["threads_per_rank"]),
              'export PYTHONPATH="$bundle${PYTHONPATH:+:$PYTHONPATH}"']
    py = job["python"]
    marker = shell_path("artifacts/{:03d}.complete.json".format(index))
    receipt = shell_path("artifacts/{:03d}.receipt.json".format(index))
    checkpoint = shell_path("artifacts/{:03d}.checkpoint.pt".format(index)) if job["checkpoint_args"] else None
    previous = None
    run = [py, shell_path("runtime.py"), "run", "--marker", marker, "--progress", str(end_step), "--out", receipt]
    if checkpoint:
        run += ["--checkpoint-out", checkpoint]
    if index > 1:
        previous = shell_path("artifacts/{:03d}.checkpoint.pt".format(index - 1))
        run += ["--checkpoint-in", previous, "--previous-receipt", shell_path("artifacts/{:03d}.receipt.json".format(index - 1)),
                "--previous-progress", str(start_step)]
    elif spec.get("resume_existing", False):
        previous = spec["initial_checkpoint"]["path"]
    require(not previous or job["resume_args"], "resuming requires resume_args")
    probe = [py, shell_path("runtime.py"), "probe", "--cards", str(cards), "--timeout-seconds", str(job["probe_timeout_seconds"]),
             "--out", shell_path("artifacts/{:03d}.probe.json".format(index))]
    if "arch" in job:
        probe += ["--arch", job["arch"]]
    launch = shell_arg(py)
    if cards > 1:
        launch += " -m torch.distributed.run --nnodes=1 --nproc_per_node=" + str(cards)
        launch += ' --max_restarts=0 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 "--rdzv_id=scnet-$SLURM_JOB_ID"'
    vals = {"{total_steps}": str(spec["total_steps"]), "{start_step}": str(start_step), "{end_step}": str(end_step),
            "{checkpoint_in}": previous, "{checkpoint_out}": checkpoint}
    args = job["args"] + (job["resume_args"] if previous else []) + job["checkpoint_args"]
    adap = [job["entrypoint"]] + [vals.get(x, x) for x in args]
    verify = [py, shell_path("scnet.py"), "_verify-files", shell_path("bundle.json"), str(index)]
    lines += [" ".join(map(shell_arg, verify)), "cd -- " + shlex.quote(job["workdir"]), " ".join(map(shell_arg, probe)),
              "exec " + " ".join(map(shell_arg, run)) + " -- " + launch + " " + " ".join(map(shell_arg, adap))]
    return "\n".join(lines) + "\n"


def render(spec_path, bundle):
    spec, manifest, workload, decision = inspect_spec(spec_path)
    route = decision["selected"]
    require(route is not None, "no viable route; see decide output")
    require(isinstance(route["id"], str) and NAME.fullmatch(route["id"]) and len(route["id"]) <= 40, "unsafe job name")
    job = workload["jobs"][route["id"]]["launch"]
    require(len(route["segments"]) == 1 or (job["resume_args"] and job["checkpoint_args"]),
            "segmented jobs require both resume_args and checkpoint_args")
    start = spec.get("start_step", 0)
    scripts = []
    for idx, (end, minutes) in enumerate(zip(route["segments"], route["requested_minutes"]), 1):
        require(job["probe_timeout_seconds"] < minutes * 60, "probe timeout must be below requested wall time")
        scripts.append(render_segment(route, workload["jobs"][route["id"]], spec, idx, start, end, minutes))
        start = end
    bundle.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "artifacts", "submissions"):
        (bundle / name).mkdir()
    shutil.copyfile(str(manifest), str(bundle / "workload.json"))
    for name in ("scnet.py", "policy.py", "runtime.py"):
        shutil.copyfile(str(Path(__file__).with_name(name)), str(bundle / name))
    names = []
    for idx, script in enumerate(scripts, 1):
        name = "segment-{:03d}.slurm".format(idx)
        (bundle / name).write_bytes(script.encode("utf-8"))
        names.append(name)
    metadata = {"selected": route, "purpose": spec["purpose"], "framework": spec["framework"],
                "workload_sha256": spec["workload_sha256"], "evidence_sha256": [e["artifact_sha256"] for e in spec["evidence"]],
                "initial_checkpoint": spec.get("initial_checkpoint"), "scripts": names,
                "hashes": {name: digest(bundle / name) for name in names + ["scnet.py", "policy.py", "runtime.py", "workload.json"]}}
    dump(bundle / "bundle.json", metadata, "x")
    return metadata


def runner(argv, timeout=30, check=True):
    removed = ("SBATCH_", "SQUEUE_", "SINFO_", "SACCT_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(removed) and k != "SLURM_CLUSTERS"}
    obj = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout,
                         encoding="utf-8", errors="replace")
    if check and obj.returncode:
        raise ValueError("{} exited {}: {}{}".format(argv[0], obj.returncode, obj.stdout[-MAX_LOG:], obj.stderr[-MAX_LOG:]))
    return {"stdout": obj.stdout, "stderr": obj.stderr, "returncode": obj.returncode}


def live_cluster():
    match = re.search(r"^ClusterName\s*=\s*(\S+)", runner(["scontrol", "show", "config"])["stdout"], re.MULTILINE)
    require(match, "Slurm ClusterName missing")
    return match.group(1)


def inspect_cluster(partition):
    require(NAME.fullmatch(partition), "invalid partition")
    return {"cluster": live_cluster(), "partition": runner(["scontrol", "-o", "show", "partition", partition])["stdout"],
            "nodes": runner(["sinfo", "-e", "-h", "-p", partition, "-o", "%R|%l|%G|%c|%m|%t|%D"])["stdout"],
            "queue": runner(["squeue", "-h", "-p", partition, "-o", "%i|%T|%D|%l|%R"], check=False),
            "fairshare": runner(["sshare", "-U"], check=False)}


def wall_seconds(value):
    if value == "UNLIMITED":
        return float("inf")
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    require(match, "unrecognized Slurm wall time: " + value)
    day, h, m, s = (int(x or 0) for x in match.groups())
    require(m < 60 and s < 60, "invalid Slurm wall time")
    return day * 86400 + h * 3600 + m * 60 + s


def compatible_node(rows, route, job):
    requested = job["gres"].split(":")
    for row in rows.splitlines():
        fields = row.split("|")
        if len(fields) != 7 or fields[0].strip() != route["partition"]:
            continue
        _, _, gres, cpus, mem, state, count = [x.strip() for x in fields]
        if not (cpus.isdigit() and mem.isdigit() and count.isdigit() and int(count) > 0):
            continue
        if int(cpus) < job["cpus_per_task"] or int(mem) < job["memory_mib"]:
            continue
        # The count is per node; never sum devices across node rows or mix typed requests.
        capacity = 0
        for name, model, n in re.findall(r"(?:^|,)([\w.-]+)(?::([\w.-]+))?:(\d+)(?:\([^)]*\))?(?=,|$)", gres):
            if name == requested[0] and (len(requested) == 1 or model == requested[1]):
                capacity += int(n)
        if capacity >= route["cards"]:
            return True
    return False


def staged(bundle):
    manifest = load(bundle / "bundle.json")
    for name, value in manifest["hashes"].items():
        require(NAME.fullmatch(name), "unsafe bundle filename")
        checked_file(bundle / name, value)
    checked_file(bundle / "workload.json", manifest["workload_sha256"])
    workload = load(bundle / "workload.json")
    route = manifest["selected"]
    return manifest, route, workload["jobs"][route["id"]]


def verify_files(bundle, index):
    manifest, _, definition = staged(bundle)
    require(Path(definition["launch"]["workdir"]).is_dir(), "training workdir does not exist")
    for name, value in definition["files"].items():
        checked_file(name, value)
    if index == 1 and manifest["initial_checkpoint"]:
        checked_file(manifest["initial_checkpoint"]["path"], manifest["initial_checkpoint"]["sha256"])


def preflight(bundle, indices):
    manifest, route, job = staged(bundle)
    require(os.name == "posix", "run on the Linux login node for the chosen route")
    require(live_cluster() == route["cluster"], "connected Slurm ClusterName does not match the selected route")
    partition = route["partition"]
    raw = runner(["scontrol", "-o", "show", "partition", partition])["stdout"]
    info = dict(re.findall(r"(\w+)=(\S+)", raw))
    require(info.get("PartitionName") == partition and info.get("State") == "UP", "partition is missing or not UP")
    limit = wall_seconds(info.get("MaxTime", ""))
    rows = runner(["sinfo", "-e", "-h", "-p", partition, "-o", "%R|%l|%G|%c|%m|%t|%D"])["stdout"]
    require(compatible_node(rows, route, job["launch"]), "no matching one-node GRES/CPU/memory shape in this partition")
    verify_files(bundle, 1)
    reports = []
    for idx in indices:
        require(1 <= idx <= len(manifest["scripts"]), "invalid segment index")
        require(route["requested_minutes"][idx - 1] * 60 <= limit, "request exceeds live partition MaxTime")
        script = bundle / manifest["scripts"][idx - 1]
        runner(["bash", "-n", str(script)])
        report = runner(["sbatch", "--test-only", "--chdir=" + str(bundle), str(script)])
        reports.append({"segment": idx, "scheduler": report})
    return manifest, reports


def submit(bundle, index, chain):
    manifest, route, _ = staged(bundle)
    indices = list(range(1, len(manifest["scripts"]) + 1)) if chain else [index]
    preflight(bundle, indices)
    if chain:
        require(not any((bundle / "submissions").iterdir()), "chain requires a fresh bundle; reconcile prior submission records")
    previous = None
    if not chain and index > 1:
        prev = load(bundle / "submissions" / "{:03d}.json".format(index - 1))
        require(prev.get("state") == "submitted" and JOBID.fullmatch(prev["job_id"]), "previous segment has no confirmed job ID")
        raw = runner(["sacct", "-n", "-X", "-P", "-j", prev["job_id"], "-o", "JobIDRaw,State,ExitCode"])["stdout"]
        require(any(row.rstrip("|").split("|") == [prev["job_id"], "COMPLETED", "0:0"] for row in raw.splitlines()),
                "previous segment not confirmed completed; inspect accounting and runtime receipt")
        from runtime import RuntimeFailure, verify_previous
        try:
            previous_receipt = bundle / "artifacts" / "{:03d}.receipt.json".format(index - 1)
            require(load(previous_receipt).get("slurm_job_id") == prev["job_id"], "previous receipt job ID does not match accounting")
            verify_previous(bundle / "artifacts" / "{:03d}.checkpoint.pt".format(index - 1),
                            previous_receipt, route["segments"][index - 2])
        except RuntimeFailure as exc:
            raise ValueError(str(exc)) from exc
    records = []
    for idx in indices:
        receipt = bundle / "submissions" / "{:03d}.json".format(idx)
        argv = ["sbatch", "--parsable", "--chdir=" + str(bundle)]
        if previous:
            argv.extend(["--dependency=afterok:" + previous, "--kill-on-invalid-dep=yes"])
        argv.append(str(bundle / manifest["scripts"][idx - 1]))
        # Persist intent before the network mutation. If no ID returns, never automatically retry.
        dump(receipt, {"state": "intent", "segment": idx, "argv": argv}, "x")
        response = runner(argv)
        parts = response["stdout"].strip().split(";")
        require(len(parts) in (1, 2) and JOBID.fullmatch(parts[0]) and (len(parts) == 1 or parts[1] == route["cluster"]),
                "ambiguous sbatch response; reconcile in squeue/sacct before retrying")
        previous = parts[0]
        result = {"state": "submitted", "segment": idx, "job_id": previous, "cluster": route["cluster"], "argv": argv}
        replace_json(receipt, result)
        records.append(result)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("decide"); p.add_argument("spec", type=Path)
    p = sub.add_parser("render"); p.add_argument("spec", type=Path); p.add_argument("bundle", type=Path)
    p = sub.add_parser("inspect"); p.add_argument("--partition", required=True)
    p = sub.add_parser("preflight"); p.add_argument("bundle", type=Path)
    p = sub.add_parser("submit"); p.add_argument("bundle", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--segment", type=int); mode.add_argument("--chain", action="store_true")
    p.add_argument("--execute", action="store_true", required=True)
    p = sub.add_parser("status"); p.add_argument("job_id")
    p = sub.add_parser("_verify-files"); p.add_argument("manifest", type=Path); p.add_argument("segment", type=int)
    args = parser.parse_args()
    result = None
    try:
        if args.command == "decide":
            result = inspect_spec(args.spec)[-1]
        elif args.command == "render":
            result = render(args.spec, args.bundle.resolve())
        elif args.command == "inspect":
            result = inspect_cluster(args.partition)
        elif args.command == "preflight":
            bundle = args.bundle.resolve()
            result = preflight(bundle, range(1, len(staged(bundle)[0]["scripts"]) + 1))[1]
        elif args.command == "submit":
            result = submit(args.bundle.resolve(), args.segment, args.chain)
        elif args.command == "status":
            require(JOBID.fullmatch(args.job_id), "expected a numeric job id")
            current = live_cluster()
            result = {"cluster": current, "queue": runner(["squeue", "-j", args.job_id], check=False),
                      "accounting": runner(["sacct", "-j", args.job_id, "-P", "-o",
                                            "JobID,State,ExitCode,Elapsed,AllocTRES,Partition"], check=False)}
        elif args.command == "_verify-files":
            verify_files(args.manifest.parent, args.segment)
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, "scnet: {}\n".format(exc))
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

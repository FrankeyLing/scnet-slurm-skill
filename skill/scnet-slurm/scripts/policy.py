"""Offline capacity planning and workload-scoped evidence checks."""

import math
import re
from fractions import Fraction


DDP_CHECKS = frozenset({
    "actual_topology", "same_initial_state", "same_effective_batches", "real_loss",
    "loss_terms", "gradients", "optimizer_updates", "multi_step", "relevant_edge_cases",
})
RESUME_CHECKS = frozenset({
    "actual_topology", "uninterrupted_baseline", "checkpoint_roundtrip", "optimizer_state",
    "scheduler_state", "progress_and_data_order", "rng_state", "extra_training_state",
    "resumed_trajectory",
})
EVIDENCE_CHECKS = {"ddp": DDP_CHECKS, "resume": RESUME_CHECKS}
ROUTE_FIELDS = ("center", "cluster", "partition", "cards")
CANDIDATE_FIELDS = {
    "id", *ROUTE_FIELDS, "wall_minutes", "queue_minutes", "setup_minutes",
    "startup_minutes", "seconds_per_step", "checkpoint_margin_minutes",
}
SPEC_REQUIRED = {
    "framework", "purpose", "workload_sha256", "total_steps", "supports_resume",
    "candidates", "evidence",
}
SPEC_OPTIONAL = {
    "schema_version", "start_step", "resume_existing", "max_segments", "max_eta_minutes",
    "max_queue_minutes", "max_card_hours", "smoke_max_minutes", "ddp_preference_percent", "workload_manifest",
    "initial_checkpoint",
}
EVIDENCE_FIELDS = {
    "type", "workload_sha256", *ROUTE_FIELDS, "passed", "artifact_sha256", "artifact", "checks",
}
SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _fields(value, required, optional, label):
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError("{} must be an object with string keys".format(label))
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing:
        raise ValueError("{} is missing: {}".format(label, ", ".join(sorted(missing))))
    if extra:
        raise ValueError("{} has unknown fields: {}".format(label, ", ".join(sorted(extra))))


def _label(value, label):
    if type(value) is not str or not value or not value.isprintable() or value != value.strip():
        raise ValueError("{} must be nonempty printable text without leading/trailing space".format(label))
    return value


def _token(value, label):
    if type(value) is not str or not SAFE_TOKEN.fullmatch(value):
        raise ValueError("{} must match {}".format(label, SAFE_TOKEN.pattern))
    return value


def _digest(value, label):
    if type(value) is not str or not SHA256.fullmatch(value):
        raise ValueError("{} must be a lowercase SHA-256 hex digest".format(label))
    return value


def _bool(value, label):
    if type(value) is not bool:
        raise ValueError("{} must be a boolean".format(label))
    return value


def _int(value, minimum, label):
    if type(value) is not int or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(label, minimum))
    return value


def _amount(value, label, positive=False):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("{} must be a finite number".format(label))
    if value < 0 or (positive and value == 0):
        raise ValueError("{} must be {}".format(label, "positive" if positive else "nonnegative"))
    # Decimal text is exact here. A requested wall time never rounds down at a boundary.
    return Fraction(str(value))


def _out_number(value):
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("planned duration or consumption exceeds numeric range") from None
    if not math.isfinite(result):
        raise ValueError("planned duration or consumption exceeds numeric range")
    return result


def _route(item, label):
    return (
        _label(item["center"], label + ".center"),
        _token(item["cluster"], label + ".cluster"),
        _token(item["partition"], label + ".partition"),
        _int(item["cards"], 1, label + ".cards"),
    )


def _load_evidence(items):
    if type(items) is not list:
        raise ValueError("evidence must be a list")
    statuses = {}
    for index, item in enumerate(items):
        label = "evidence[{}]".format(index)
        _fields(item, EVIDENCE_FIELDS, set(), label)
        kind = item["type"]
        if kind not in ("ddp", "resume"):
            raise ValueError(label + ".type must be ddp or resume")
        digest = _digest(item["workload_sha256"], label + ".workload_sha256")
        route = _route(item, label)
        passed = _bool(item["passed"], label + ".passed")
        _digest(item["artifact_sha256"], label + ".artifact_sha256")
        _label(item["artifact"], label + ".artifact")
        checks = item["checks"]
        if type(checks) is not list or any(type(check) is not str for check in checks):
            raise ValueError(label + ".checks must be a list of check names")
        if len(set(checks)) != len(checks) or set(checks) - EVIDENCE_CHECKS[kind]:
            raise ValueError(label + ".checks has duplicated or unknown check names")
        if passed and set(checks) != EVIDENCE_CHECKS[kind]:
            missing = EVIDENCE_CHECKS[kind] - set(checks)
            raise ValueError(label + ".checks is missing: " + ", ".join(sorted(missing)))
        statuses.setdefault((kind, digest, route), set()).add(passed)
    return statuses


def _check_gate(kind, evidence, digest, route):
    status = evidence.get((kind, digest, route))
    if status == {True}:
        return None
    if status:
        return "{} acceptance has a failing record for this workload and route".format(kind)
    return "{} acceptance is missing for this workload and route".format(kind)


def _evaluate(spec, route, digest, evidence, start, total, max_segments, smoke_max, ceilings):
    wall = _int(route["wall_minutes"], 1, route["id"] + ".wall_minutes")
    queue, setup, startup, cost, margin = (
        _amount(route[key], route["id"] + "." + key, positive=(key == "seconds_per_step"))
        for key in ("queue_minutes", "setup_minutes", "startup_minutes", "seconds_per_step",
                    "checkpoint_margin_minutes")
    )
    target = _route(route, route["id"])
    smoke = spec["purpose"] == "smoke"
    remaining = total - start
    wall_budget = min(wall, smoke_max) if smoke else wall
    if smoke and (startup + margin + remaining * cost / 60) > wall_budget:
        return "smoke must fit in one job within queue and smoke time limits"
    capacity = (wall_budget - startup - margin) * 60 // cost
    if capacity < 1:
        return "wall limit cannot fit startup, checkpoint reserve and one step"
    count = (remaining + capacity - 1) // capacity
    if count > max_segments:
        return "requires {} segments, above max_segments={}".format(count, max_segments)
    continuation = count > 1 or spec.get("resume_existing", False)
    if continuation and not spec["supports_resume"]:
        return "continuation requires explicit full-state resume support"
    if continuation and margin == 0:
        return "continuation requires a positive checkpoint time reserve"
    if spec["framework"] == "custom":
        if route["cards"] > 1 and not smoke:
            reason = _check_gate("ddp", evidence, digest, target)
            if reason:
                return reason
        if continuation:
            reason = _check_gate("resume", evidence, digest, target)
            if reason:
                return reason

    segments = [min(total, start + capacity * index) for index in range(1, count + 1)]
    requests = [
        math.ceil(startup + margin + (end - begin) * cost / 60)
        for begin, end in zip([start] + segments, segments)
    ]
    allocated = sum(requests)
    queue_total = queue * count
    eta = setup + queue_total + allocated
    card_hours = Fraction(allocated * route["cards"], 60)
    for key, value, unit in (("max_queue_minutes", queue_total, "minutes of queue time"),
                             ("max_eta_minutes", eta, "minutes to finish"),
                             ("max_card_hours", card_hours, "requested card-hours")):
        if key in ceilings and value > ceilings[key]:
            return "estimated {:.2f} {} exceeds {}".format(_out_number(value), unit, key)
    return {
        **route,
        "segments": segments,
        "requested_minutes": requests,
        "estimated_total_queue_minutes": _out_number(queue_total),
        "eta_minutes": _out_number(eta),
        "card_hours": _out_number(card_hours),
    }


def _rank_key(route):
    return route["eta_minutes"], route["card_hours"], len(route["segments"]), route["id"]


def plan(spec):
    """Rank feasible routes. Input errors raise ValueError and the result contains all rejections."""
    _fields(spec, SPEC_REQUIRED, SPEC_OPTIONAL, "plan")
    if "schema_version" in spec and (type(spec["schema_version"]) is not int or spec["schema_version"] != 1):
        raise ValueError("schema_version must be 1")
    if spec["framework"] not in ("mature", "custom"):
        raise ValueError("framework must be mature or custom")
    if spec["purpose"] not in ("smoke", "formal"):
        raise ValueError("purpose must be smoke or formal")
    digest = _digest(spec["workload_sha256"], "workload_sha256")
    total = _int(spec["total_steps"], 1, "total_steps")
    start = _int(spec.get("start_step", 0), 0, "start_step")
    if start >= total:
        raise ValueError("start_step must be less than total_steps")
    resume_existing = _bool(spec.get("resume_existing", False), "resume_existing")
    _bool(spec["supports_resume"], "supports_resume")
    if start and not resume_existing:
        raise ValueError("nonzero start_step requires resume_existing=true")
    if spec["purpose"] == "smoke" and resume_existing:
        raise ValueError("smoke mode starts from the supplied initial state, without continuation")

    max_segments = _int(spec.get("max_segments", 256), 1, "max_segments")
    ceilings = {key: _amount(spec[key], key) for key in ("max_eta_minutes", "max_queue_minutes", "max_card_hours")
                if key in spec}
    smoke_max = _int(spec.get("smoke_max_minutes", 15), 1, "smoke_max_minutes")
    ddp_window = _amount(spec.get("ddp_preference_percent", 10), "ddp_preference_percent")
    if ddp_window > 100:
        raise ValueError("ddp_preference_percent must be <= 100")
    evidence = _load_evidence(spec["evidence"])
    candidates = spec["candidates"]
    if type(candidates) is not list or not candidates:
        raise ValueError("candidates must be a nonempty list")

    eligible, rejected, ids = [], [], set()
    for index, route in enumerate(candidates):
        label = "candidates[{}]".format(index)
        _fields(route, CANDIDATE_FIELDS, set(), label)
        route_id = _token(route["id"], label + ".id")
        if route_id in ids:
            raise ValueError("duplicate candidate id: " + route_id)
        ids.add(route_id)
        result = _evaluate(spec, route, digest, evidence, start, total, max_segments, smoke_max, ceilings)
        if isinstance(result, str):
            rejected.append({**route, "status": "rejected", "reason": result})
        else:
            eligible.append(result)

    eligible.sort(key=_rank_key)
    selected = eligible[0] if eligible else None
    reason = "lowest estimated finish time among eligible routes" if eligible else "no eligible candidate"
    if selected and spec["framework"] == "mature":
        boundary = Fraction(str(selected["eta_minutes"])) * (1 + ddp_window / 100)
        for route in eligible:
            if route["cards"] > 1 and Fraction(str(route["eta_minutes"])) <= boundary:
                if selected is not route:
                    selected = route
                    reason = "mature distributed route is within the configured finish-time preference"
                break
    alternatives = [
        {**route, "status": "eligible", "reason": "not selected by finish-time ranking and framework preference"}
        for route in eligible if route is not selected
    ] + rejected
    return {"selected": selected, "selection_reason": reason, "alternatives": alternatives}

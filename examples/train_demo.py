"""Small PyTorch adapter with explicit rank slicing and native checkpoint state."""

import argparse
from datetime import timedelta
import math
import os
import sys
import tempfile

from runtime import RuntimeFailure, file_path, mark_complete, positive, require, torch_devices


def save_checkpoint(torch, path, state):
    descriptor, temporary = tempfile.mkstemp(prefix=".demo-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train(args):
    require(0 <= args.start_step < args.end_step <= args.total_steps,
            "require 0 <= start-step < end-step <= total-steps")
    require(0 <= args.seed < (1 << 63), "seed must be in [0, 2**63)")
    require(math.isfinite(args.lr) and args.lr > 0, "lr must be finite and positive")
    require(bool(args.checkpoint_in) == (args.start_step != 0),
            "checkpoint-in is required exactly when resuming at a nonzero step")
    require(os.environ["SCNET_EXPECTED_PROGRESS"] == str(args.end_step),
            "end-step differs from the runtime target")
    out = file_path(args.checkpoint_out) if args.checkpoint_out else None
    expected_out = os.getenv("SCNET_CHECKPOINT_OUT")
    require(out == (file_path(expected_out) if expected_out else None),
            "checkpoint-out differs from the runtime output")

    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    require(0 <= rank == local_rank < world and args.local_rank in (None, local_rank),
            "expected single-node rank mapping")
    require(args.global_batch % world == 0,
            "global batch must divide evenly across ranks")
    local_batch = args.global_batch // world
    torch = torch_devices(world)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    model = torch.nn.Linear(args.width, 1, bias=True).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)
    recipe = {
        "seed": args.seed, "global_batch": args.global_batch, "width": args.width,
        "lr": args.lr, "total_steps": args.total_steps, "world_size": world,
    }
    if args.checkpoint_in:
        state = torch.load(args.checkpoint_in, map_location="cpu")
        require(state["format"] == 1 and state["recipe"] == recipe and
                state["progress"] == args.start_step,
                "checkpoint recipe or progress differs from this run")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        require(scheduler.last_epoch == args.start_step,
                "scheduler progress differs from the checkpoint step")

    runner = model
    if world > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(seconds=120),
        )
        runner = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )
    runner.train()
    criterion = torch.nn.MSELoss(reduction="mean")
    completed = args.start_step
    for step in range(args.start_step, args.end_step):
        # A step determines its complete batch. Every rank takes a disjoint slice,
        # and a resumed step needs no hidden sampler position or random state.
        generator = torch.Generator(device="cpu").manual_seed((args.seed + step) % (1 << 63))
        full_batch = torch.randn(
            args.global_batch, args.width, device="cpu", dtype=torch.float32,
            generator=generator,
        )
        batch = full_batch[rank * local_batch:(rank + 1) * local_batch].to(device)
        target = batch.sum(dim=1, keepdim=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(runner(batch), target)
        scalar = loss.item()
        if world > 1:
            # Complete the same collective on all ranks before branching on bad loss.
            finite = torch.tensor([int(math.isfinite(scalar))], dtype=torch.int64, device=device)
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            require(finite.item() == 1, "nonfinite loss on at least one rank")
        else:
            require(math.isfinite(scalar), "nonfinite loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        completed += 1
        if rank == 0:
            print("step={} rank0_loss={:.6g} next_lr={:.6g}".format(
                completed, scalar, scheduler.get_last_lr()[0]), flush=True)

    if rank == 0:
        if out:
            save_checkpoint(torch, out, {
                "format": 1, "recipe": recipe, "progress": completed,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            })
        mark_complete(completed, out)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--global-batch", type=positive, required=True)
    parser.add_argument("--width", type=positive, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=positive, required=True)
    parser.add_argument("--total-steps", type=positive, required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--checkpoint-out")
    try:
        train(parser.parse_args())
    except (RuntimeFailure, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        print("train_demo: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

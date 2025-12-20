#!/usr/bin/env python3
"""Replay a saved forward reference and compare outputs in training builds."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch

from flashmoe import _C, flashmoe_forward


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA is required for FlashMoE forward replay.")


def _get_local_rank() -> int:
    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",
            os.environ.get(
                "PMI_RANK",
                os.environ.get("SLURM_PROCID", "0"),
            ),
        )
    )


def _format_snapshot_path(template: str, rank: int) -> str:
    if "{rank}" in template:
        return template.format(rank=rank)
    if template.endswith(".pt"):
        return template
    return os.path.join(template, f"forward_reference.rank{rank}.pt")


def _load_reference(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    required = {"input", "gate_weights", "expert_weights", "output"}
    missing = required.difference(checkpoint.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Snapshot {path} is missing tensors: {missing_str}")
    return checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and validate FlashMoE forward snapshots.")
    parser.add_argument(
        "--snapshot",
        default=os.path.join("artifacts", "forward_reference.rank{rank}.pt"),
        help="Path or template for the reference snapshot. Use {rank} as a placeholder.",
    )
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for comparison.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for comparison.")
    parser.add_argument(
        "--dump-diffs",
        action="store_true",
        help="Print max/mean absolute differences before asserting closeness.",
    )
    return parser.parse_args()


def main() -> None:
    _require_cuda()
    args = _parse_args()

    local_rank = _get_local_rank()
    device_count = torch.cuda.device_count()
    if device_count == 0:
        sys.exit("No CUDA devices found.")
    device_id = local_rank % device_count
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    snapshot_path = _format_snapshot_path(args.snapshot, local_rank)
    if not os.path.exists(snapshot_path):
        sys.exit(f"Snapshot not found for rank {local_rank}: {snapshot_path}")

    checkpoint = _load_reference(snapshot_path)
    initialized = False
    try:
        _C.initialize()
        initialized = True

        def _move(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, non_blocking=True)

        input_tensor = _move(checkpoint["input"])
        gate_weights = _move(checkpoint["gate_weights"])
        expert_weights = _move(checkpoint["expert_weights"])
        reference_output = checkpoint["output"].to(device=device, non_blocking=True)

        # Check gate routing determinism - compute gate logits manually
        if args.dump_diffs:
            # input_tensor: [1, S, H], gate_weights: [H, E] -> gate_logits: [1, S, E]
            gate_logits = torch.matmul(input_tensor, gate_weights)
            topk_vals, topk_idx = gate_logits.topk(2, dim=-1)  # top-2 experts per token
            # Print routing at specific positions (including where max diff occurred)
            print(f"Rank {local_rank}: gate_logits shape={gate_logits.shape}", flush=True)
            print(f"Rank {local_rank}: gate topk[0,100,:] idx={topk_idx[0,100,:].tolist()} vals={topk_vals[0,100,:].tolist()}", flush=True)
            print(f"Rank {local_rank}: gate topk[0,166,:] idx={topk_idx[0,166,:].tolist()} vals={topk_vals[0,166,:].tolist()}", flush=True)
            print(f"Rank {local_rank}: gate topk[0,7788,:] idx={topk_idx[0,7788,:].tolist()} vals={topk_vals[0,7788,:].tolist()}", flush=True)
            # Check overall routing distribution
            expert_counts = torch.bincount(topk_idx.flatten(), minlength=gate_logits.shape[-1])
            print(f"Rank {local_rank}: expert usage min={expert_counts.min().item()} max={expert_counts.max().item()} mean={expert_counts.float().mean().item():.1f}", flush=True)

        output = flashmoe_forward(input_tensor, gate_weights, expert_weights)
        torch.cuda.synchronize()

        if args.dump_diffs:
            diff = (output - reference_output).abs()
            print(
                f"Rank {local_rank}: max abs diff={diff.max().item():.6e}, mean abs diff={diff.mean().item():.6e}",
                flush=True,
            )
            # Additional diagnostics
            print(f"Rank {local_rank}: output has NaN={output.isnan().any().item()}, Inf={output.isinf().any().item()}", flush=True)
            print(f"Rank {local_rank}: output range [{output.min().item():.2f}, {output.max().item():.2f}]", flush=True)
            print(f"Rank {local_rank}: ref range [{reference_output.min().item():.2f}, {reference_output.max().item():.2f}]", flush=True)
            # Check for zeros (potential unwritten memory)
            zeros_out = (output == 0).sum().item()
            zeros_ref = (reference_output == 0).sum().item()
            total_elems = output.numel()
            print(f"Rank {local_rank}: zeros in output={zeros_out}/{total_elems}, ref={zeros_ref}/{total_elems}", flush=True)
            # Find where max diff occurs
            flat_diff = diff.flatten()
            max_idx = flat_diff.argmax().item()
            print(f"Rank {local_rank}: max diff at flat idx {max_idx}, output={output.flatten()[max_idx].item():.4f}, ref={reference_output.flatten()[max_idx].item():.4f}", flush=True)

        torch.testing.assert_close(
            output,
            reference_output,
            rtol=args.rtol,
            atol=args.atol,
            msg=f"Rank {local_rank}: forward replay mismatch",
        )
        print(f"Rank {local_rank}: forward replay matched reference within tolerances.", flush=True)
    finally:
        if initialized:
            _C.finalize()


if __name__ == "__main__":
    main()

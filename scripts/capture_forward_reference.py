#!/usr/bin/env python3
"""Capture deterministic forward tensors from an inference build."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch

from flashmoe import _C, flashmoe_forward, get_compiled_config


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA is required for FlashMoE snapshot capture.")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture FlashMoE forward tensors for replay.")
    parser.add_argument(
        "--snapshot",
        default=os.path.join("artifacts", "forward_reference.rank{rank}.pt"),
        help="Path or template for the snapshot. Use {rank} for per-rank filenames.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing snapshot for this rank if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed to make tensor generation deterministic.",
    )
    return parser.parse_args()


def _ensure_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _create_tensors(device: torch.device) -> Dict[str, torch.Tensor]:
    cfg = get_compiled_config()
    n_local_experts = int(_C.get_num_local_experts())

    input_tensor = torch.randn(1, cfg["S"], cfg["H"], device=device, dtype=torch.float32)
    gate_weights = torch.randn(cfg["H"], cfg["E"], device=device, dtype=torch.float32)
    expert_weights = torch.randn(n_local_experts, 2, cfg["P"], cfg["H"], device=device, dtype=torch.float32)

    return {
        "input": input_tensor,
        "gate_weights": gate_weights,
        "expert_weights": expert_weights,
    }


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

    torch.manual_seed(args.seed)
    snapshot_path = _format_snapshot_path(args.snapshot, local_rank)
    if os.path.exists(snapshot_path) and not args.force:
        sys.exit(
            f"Snapshot already exists for rank {local_rank}: {snapshot_path}. "
            "Pass --force to overwrite."
        )

    _ensure_directory(snapshot_path)

    initialized = False
    try:
        _C.initialize()
        initialized = True

        tensors = _create_tensors(device)
        output = flashmoe_forward(tensors["input"], tensors["gate_weights"], tensors["expert_weights"])
        torch.cuda.synchronize()

        checkpoint = {**{k: v.cpu() for k, v in tensors.items()}, "output": output.cpu()}
        torch.save(checkpoint, snapshot_path, _use_new_zipfile_serialization=False)
        print(f"Rank {local_rank}: saved snapshot to {snapshot_path}", flush=True)
    finally:
        if initialized:
            _C.finalize()


if __name__ == "__main__":
    main()

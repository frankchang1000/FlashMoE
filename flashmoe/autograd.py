"""
Autograd helpers for FlashMoE.

The module exposes a custom :class:`FlashMoEFunction` that saves the tensors
sourced by the forward pass, calls ``_C.moe_backward`` in ``backward``, and
returns the computed gradients together with the recorded kernel duration.
"""

from __future__ import annotations

from typing import Tuple

import torch

from . import _C

Tensor = torch.Tensor


class FlashMoEFunction(torch.autograd.Function):
    """Custom autograd function that drives FlashMoE forward/backward passes."""

    last_backward_duration: float = float("nan")

    @staticmethod
    def forward(
        ctx: torch.autograd.FunctionCtx,
        input: Tensor,
        gate_weights: Tensor,
        expert_weights: Tensor
    ) -> Tensor:
        ctx.save_for_backward(input, gate_weights, expert_weights)
        return _C.moe_forward(input, gate_weights, expert_weights)

    @staticmethod
    def backward(ctx: torch.autograd.FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        input, gate_weights, expert_weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        # print(f"DEBUG backward: grad_output shape={grad_output.shape}, input shape={input.shape}")

        result = _C.moe_backward(grad_output, input, gate_weights, expert_weights)

        # print(f"DEBUG backward: moe_backward returned {len(result)} items")
        # for i, r in enumerate(result):
        #     if isinstance(r, Tensor):
        #         print(f"  result[{i}]: Tensor shape={r.shape}, dtype={r.dtype}, has_nan={r.isnan().any().item()}")
        #     else:
        #         print(f"  result[{i}]: {type(r).__name__} = {r}")

        (
            grad_input,
            grad_gate_weights,
            grad_expert_up,
            grad_expert_down,
            _grad_bias_up,
            _grad_bias_down,
            duration
        ) = result

        FlashMoEFunction.last_backward_duration = float(duration)
        grad_expert_weights = torch.stack((grad_expert_up, grad_expert_down), dim=1)

        # print(f"DEBUG backward: returning grad_input={grad_input is not None}, grad_gate={grad_gate_weights is not None}, grad_expert={grad_expert_weights is not None}")

        return grad_input, grad_gate_weights, grad_expert_weights


def flashmoe_forward(input: Tensor, gate_weights: Tensor, expert_weights: Tensor) -> Tensor:
    """Torch-friendly entry point that uses ``FlashMoEFunction``."""
    return FlashMoEFunction.apply(input, gate_weights, expert_weights)


def get_last_backward_duration() -> float:
    """Return the duration (ms) that the last backward kernel spent on the GPU."""
    return FlashMoEFunction.last_backward_duration


__all__ = ["FlashMoEFunction", "flashmoe_forward", "get_last_backward_duration"]

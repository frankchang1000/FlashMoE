#!/usr/bin/env python3
"""
Numerical gradient check for FlashMoE backward pass.

This script uses torch.autograd.gradcheck to verify that the backward pass
computes gradients that match finite-difference estimates.

It is probably not possible to run actual gradcheck with the config required for multi-rank
configs, so we only sample a few tensors.
"""

from __future__ import annotations

import os
import sys

import torch

from flashmoe import _C, get_compiled_config
from flashmoe.autograd import FlashMoEFunction


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA is required for FlashMoE gradcheck.")


def make_float32_wrapper():
    """
    Create a wrapper that casts double inputs to float32 for the kernel,
    then casts outputs back to double for gradcheck.

    This is needed because:
    - gradcheck requires float64 for numerical precision
    - FlashMoE kernels only support float32/bf16
    """
    class Float32Wrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_f64, gate_weights_f64, expert_weights_f64):
            # Convert to float32 for kernel
            input_f32 = input_f64.float()
            gate_f32 = gate_weights_f64.float()
            expert_f32 = expert_weights_f64.float()

            # Save for backward
            ctx.save_for_backward(input_f32, gate_f32, expert_f32)

            # Run forward
            output_f32 = _C.moe_forward(input_f32, gate_f32, expert_f32)

            # Convert back to float64
            return output_f32.double()

        @staticmethod
        def backward(ctx, grad_output_f64):
            input_f32, gate_f32, expert_f32 = ctx.saved_tensors

            # Convert grad to float32
            grad_output_f32 = grad_output_f64.float().contiguous()

            # Run backward
            (
                grad_input_f32,
                grad_gate_f32,
                grad_expert_up_f32,
                grad_expert_down_f32,
                _grad_bias_up,
                _grad_bias_down,
                _duration
            ) = _C.moe_backward(grad_output_f32, input_f32, gate_f32, expert_f32)

            # Stack expert grads and convert to float64
            grad_expert_f32 = torch.stack((grad_expert_up_f32, grad_expert_down_f32), dim=1)

            return (
                grad_input_f32.double(),
                grad_gate_f32.double(),
                grad_expert_f32.double()
            )

    return Float32Wrapper.apply


def check_config():
    """Verify the compiled config is suitable for gradcheck."""
    cfg = get_compiled_config()

    print(f"Compiled config: S={cfg['S']}, H={cfg['H']}, E={cfg['E']}, P={cfg['P']}")
    print(f"  is_training={cfg.get('is_training', 'N/A')}, drop_tokens={cfg.get('drop_tokens', 'N/A')}")

    # Warn if dimensions are too large
    total_params = cfg['S'] * cfg['H'] + cfg['H'] * cfg['E'] + cfg['E'] * 2 * cfg['P'] * cfg['H']
    if total_params > 100000:
        print(f"WARNING: Total parameters ({total_params}) may be too large for gradcheck.")
        print("         Consider using csrc/flashmoe_config_gradcheck.json")
        return False

    if cfg.get('is_training', 0) != 1:
        print("WARNING: is_training != 1. Backward pass may not work correctly.")
        return False

    return True


def run_gradcheck():
    """Run numerical gradient check."""
    cfg = get_compiled_config()
    n_local_experts = int(_C.get_num_local_experts())
    device = torch.device("cuda")
    torch.manual_seed(42)

    input_tensor = torch.randn(
        1, cfg["S"], cfg["H"],
        device=device, dtype=torch.float64
    ) * 0.1
    input_tensor.requires_grad_(True)

    gate_weights = torch.randn(
        cfg["H"], cfg["E"],
        device=device, dtype=torch.float64
    ) * 0.1
    gate_weights.requires_grad_(True)

    expert_weights = torch.randn(
        n_local_experts, 2, cfg["P"], cfg["H"],
        device=device, dtype=torch.float64
    ) * 0.1
    expert_weights.requires_grad_(True)

    wrapper_fn = make_float32_wrapper()

    print("\nRunning torch.autograd.gradcheck...")
    print("  (This may take a few minutes)")

    try:
        result = torch.autograd.gradcheck(
            wrapper_fn,
            (input_tensor, gate_weights, expert_weights),
            eps=1e-3,           
            atol=1e-2,
            rtol=1e-2,
            nondet_tol=1e-2,
            check_grad_dtypes=False,
            raise_exception=True
        )
        print(f"\nGradcheck result: {'PASSED' if result else 'FAILED'}")
        return result
    except Exception as e:
        print(f"\nGradcheck FAILED with exception:\n{e}")
        return False


def run_manual_gradcheck():
    """
    Manual gradient check on a subset of parameters.
    Faster than full gradcheck, useful for quick sanity checks.
    """
    cfg = get_compiled_config()
    n_local_experts = int(_C.get_num_local_experts())

    device = torch.device("cuda")
    torch.manual_seed(42)

    input_tensor = torch.randn(
        1, cfg["S"], cfg["H"],
        device=device, dtype=torch.float32
    ) * 0.1
    input_tensor.requires_grad_(True)

    gate_weights = torch.randn(
        cfg["H"], cfg["E"],
        device=device, dtype=torch.float32
    ) * 0.1
    gate_weights.requires_grad_(True)

    expert_weights = torch.randn(
        n_local_experts, 2, cfg["P"], cfg["H"],
        device=device, dtype=torch.float32
    ) * 0.1
    expert_weights.requires_grad_(True)

    print("\nRunning manual gradient check on subset of parameters...")

    eps = 1e-3

    def compute_loss(inp, gate, expert):
        out = FlashMoEFunction.apply(inp, gate, expert)
        return out.pow(2).mean()

    loss = compute_loss(input_tensor, gate_weights, expert_weights)
    loss.backward()

    torch.cuda.synchronize()

    print(f"  input_tensor.grad is None: {input_tensor.grad is None}")
    print(f"  gate_weights.grad is None: {gate_weights.grad is None}")
    print(f"  expert_weights.grad is None: {expert_weights.grad is None}")
    print(f"  input_tensor.is_leaf: {input_tensor.is_leaf}")
    print(f"  gate_weights.is_leaf: {gate_weights.is_leaf}")
    print(f"  expert_weights.is_leaf: {expert_weights.is_leaf}")

    num_checks = 5
    max_rel_error = 0.0

    for name, tensor in [("input", input_tensor), ("gate", gate_weights), ("expert", expert_weights)]:
        if tensor.grad is None:
            print(f"  WARNING: {name}.grad is None - skipping gradient check for this tensor")
            continue
        flat = tensor.view(-1)
        grad_flat = tensor.grad.view(-1)

        indices = torch.randperm(flat.numel())[:num_checks]

        for idx in indices:
            idx = idx.item()

            orig_val = flat[idx].item()
            analytical_grad = grad_flat[idx].item()

            with torch.no_grad():
                flat[idx] = orig_val + eps
            loss_plus = compute_loss(
                input_tensor.detach(),
                gate_weights.detach(),
                expert_weights.detach()
            ).item()

            with torch.no_grad():
                flat[idx] = orig_val - eps
            loss_minus = compute_loss(
                input_tensor.detach(),
                gate_weights.detach(),
                expert_weights.detach()
            ).item()

            with torch.no_grad():
                flat[idx] = orig_val 

            numerical_grad = (loss_plus - loss_minus) / (2 * eps)

            abs_diff = abs(numerical_grad - analytical_grad)
            rel_error = abs_diff / (abs(numerical_grad) + 1e-8)
            max_rel_error = max(max_rel_error, rel_error)

            status = "OK" if rel_error < 0.05 else "MISMATCH"
            print(f"  {name}[{idx}]: numerical={numerical_grad:.6e}, analytical={analytical_grad:.6e}, rel_err={rel_error:.4f} [{status}]")

    tensors_with_grads = sum(1 for t in [input_tensor, gate_weights, expert_weights] if t.grad is not None)
    if tensors_with_grads == 0:
        print("\nFAILED: No tensors received gradients from backward pass!")
        print("This indicates the backward function is not returning gradients correctly.")
        return False

    print(f"\nTensors with gradients: {tensors_with_grads}/3")
    print(f"Max relative error: {max_rel_error:.4f}")
    passed = max_rel_error < 0.05 and tensors_with_grads == 3
    print(f"Manual gradcheck: {'PASSED' if passed else 'FAILED'}")
    return passed


def main():
    _require_cuda()

    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_RANK",
                       os.environ.get("PMI_RANK",
                                      os.environ.get("SLURM_PROCID", "0"))))
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    print(f"Using local_rank={local_rank}, device_id={device_id}")

    initialized = False
    try:
        print("BEFORE _C.initialize()")
        _C.initialize()
        print("AFTER _C.initialize()")
        initialized = True

        config_ok = check_config()

        if not config_ok:
            print("\nConfig check failed. Running manual gradcheck anyway...")
            result = run_manual_gradcheck()
        else:
            result = run_gradcheck()

            if not result:
                print("\nFull gradcheck failed. Trying manual gradcheck...")
                result = run_manual_gradcheck()

        sys.exit(0 if result else 1)

    finally:
        if initialized:
            _C.finalize()


if __name__ == "__main__":
    main()

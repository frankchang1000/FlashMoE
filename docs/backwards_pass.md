# Backward Pass Architecture

## Overview

The backward pass computes gradients for input, gate weights, and expert weights given `grad_output`. It reuses the forward pass infrastructure but with reversed data flow semantics.

## Two-Path Asymmetry

Unlike forward pass, gradient tasks can originate from two sources:

```
Path 1: P2P Experts (token GPU can access expert GPU via NVLink)
───────────────────────────────────────────────────────────────
dispatch ──► initial decoder ──► gradCombine (TNx per row) ──► notifyGradient ──► gradPostGEMM ──► gradPreGEMM
                 │                         │                                              │
                 │                   splits grad_output                          completion signal
                 │                   to expert buffer                                     │
                 │                                                                        ▼
                 └──► gradGateCombine (1 per row)                                   P2P final decoder
                            │                                                             │
                            ▼                                                             ▼
                    notifyGateGradient                                    gradCombine (TNx) + gradInputCombine (TNx)
                            │                                                    isPeerRemote = false
                            ▼                                                             │
                      gradGateGEMM                                       reads grad_input via P2P from xM
                                                                         atomicAdd to gradInputBasePtr

Path 2: Remote Experts (token GPU accesses expert GPU via NVSHMEM only)
───────────────────────────────────────────────────────────────────────
dispatch ──► initial decoder ──► gradCombine (TNx per row) ──► notifyGradient ──► gradPostGEMM ──► gradPreGEMM
                 │                         │                                              │
                 │                   splits grad_output                          grad_input → heap
                 │                   to expert buffer                            nvshmem signal
                 │                                                                        │
                 └──► gradGateCombine (1 per row)                                         ▼
                            │                                                   remote final decoder
                            ▼                                                             │
                    notifyGateGradient                                                    ▼
                            │                                    gradCombine (TNx) + gradInputCombine (TNx)
                            ▼                                              if isRemoteExpert
                      gradGateGEMM                                         isPeerRemote = true
                                                                                 │
                                                                    reads grad_input from heap
                                                                    atomicAdd to gradInputBasePtr
```

**Path 1 (P2P Experts)**: When a token's expert is on a P2P-accessible GPU (same node), the initial decoder on the expert GPU emits TNx `gradCombine` tasks plus 1 `gradGateCombine` per row tile. These trigger the gradPostGEMM → gradPreGEMM chain. Upon completion, the **P2P final decoder** emits per tileIdx: 1 `gradCombine` and 1 `gradInputCombine` **unconditionally**. The `isPeerRemote=false` flag means grad_input is read from xM via P2P. Note: gradGateCombine is only emitted by the initial decoder (not the final decoder) to avoid double-counting in gateBuffer.

**Path 2 (Remote Experts)**: When a token's expert is on a remote GPU (NVSHMEM only), the initial decoder works identically. However, `gradPreGEMM` writes grad_input to the heap and sends an NVSHMEM signal. The token GPU's **remote final decoder** emits: TNx `gradCombine` and TNx `gradInputCombine` **only if `isRemoteExpert = (peerIdx != epRank)`**. The `isPeerRemote=true` flag means grad_input is read from the heap. Note: gradGateCombine is only emitted by the initial decoder.

## Task Types

| Task | Created By | Operation |
|------|------------|-----------|
| `gradCombine` | Initial/Final decoder | Split `grad_output[tokenIdx]` to expert gradient buffer: computes `(grad_output / prob) * scale
| `gradPostGEMM` | `notifyGradient` | `grad_intermediate = (grad * act'(z2)) @ W2^T` + dW2 computation |
| `gradPreGEMM` | `notifyGradPreGEMM` | `grad_input = (grad_intermediate * act'(z1)) @ W1^T` + dW1 computation |
| `gradGateCombine` | Initial/Final decoder | Compute dot product `grad_output · y_e` for routing gradients |
| `gradGateGEMM` | `notifyGateGradient` | Gate weight grads + input grads from routing |
| `gradInputCombine` | Final decoder (gradient) | Accumulate `grad_input` from experts to token positions |

## Gradient Flow

```
grad_output[S,H]
       │
       ├──────────────────────────────┐
       ▼                              ▼
  gradCombine                   gradGateCombine
  (split to experts)            (routing grads)
       │                              │
       ▼                              ▼
  gradPostGEMM                  gradGateGEMM
  ((grad * act'(z2)) @ W2^T)   (softmax jacobian, dW_gate)
  (compute dW2)                       │
       │                              │
       ▼                              │
  gradPreGEMM                         │
  ((grad_intermediate * act'(z1)) @ W1^T)
  (compute dW1)                       │
       │                              │
       ▼                              │
  gradInputCombine                    │
  (atomicAdd to gradInputBasePtr)     │
       │                              │
       ▼                              ▼
  grad_input[S,H] ◄───────────────────┘
       (accumulated from both paths)
```

The backward pass splits `grad_output` into two parallel computation streams. The **expert gradient path** runs `gradCombine` → `gradPostGEMM` → `gradPreGEMM` → `gradInputCombine`: `gradCombine` distributes `grad_output` (scaled by routing probability) to per-expert buffers, `gradPostGEMM` computes `(grad * act'(z2)) @ W2^T` to get intermediate gradients plus `dW2`, `gradPreGEMM` computes `(grad_intermediate * act'(z1)) @ W1^T` plus `dW1`, and finally `gradInputCombine` atomicAdds the result to `gradInputBasePtr`. The **gate gradient path** runs `gradGateCombine` → `gradGateGEMM`: `gradGateCombine` computes the dot product `grad_output · y_e` (the true softmax derivative), and `gradGateGEMM` applies the softmax jacobian to produce `dW_gate`. Both paths contribute to the final `grad_input`—expert gradients via `gradInputCombine` accumulation, and gate gradients via `gradGateGEMM`'s contribution to input gradients from the routing layer.

## Grad Combine Math
Forward pass (combine):
output[token] = Σ_e (expert_output_e / prob_e) * scale_e

backward pass (splitGradients):
∂L/∂(expert_output_e) = ∂L/∂output * ∂output/∂(expert_output_e)
                        = grad_output * (scale_e / prob_e)
                        = (grad_output / prob_e) * scale_e

## Gate Gradient Math

The precise gradient for the gate logits requires the expert outputs:
```
∂L/∂gate_logit[t,e] = softmax_gate_derivative · dL/dprob[t,e]
                  ≃ Σ_h grad_output[t,h] * y_e[t,h]
```
where `y_e[t,h]` is the expert response that was mixed by `prob_e`. `gradGateCombine` reads each expert's `y_e` from `savedZ2` (derived via the xM row index in `cData[1]`), computes `Σ_h grad_output[t,h] * y_e[t,h]` tile-by-tile, and writes that exact dot product into `gateBuffer`. This avoids racing with `gradCombine` which writes to the packet buffer, and works correctly when the output activation is identity (`z2 == y_e`).

forward training saves the expert activations (`z2()`/`xM()`) and routing scores. `gradGateGEMM` already reads `gateRoutingScores()` when reconstructing the softmax (`processor.cuh:1691-1760`), so the necessary per-token/expert outputs are available for the corrected dot-product accumulation (`grad_output · y_e`). The current implementation just reuses those activations inside `gradGateCombine` instead of recomputing them, which keeps the math accurate without new buffers.

## Key Buffers

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `z1()` | `[world*nLx*pEC, P]` | Saved pre-activation from preGEMM |
| `z2()` | `[world*nLx*pEC, H]` | Saved pre-activation from postGEMM |
| `xM()` | `[world*nLx*pEC, P]` | Intermediate activations (preGEMM output) |
| `gW()` | Per-expert: `[2*P*H + P + H]` | Weight gradients (dW1, dW2, db1, db2) |
| `gGateW()` | `[H, E]` | Gate weight gradients |
| `gGateCombine()` | `[S, E]` | Routing gradient accumulation |
| `tSA()` | `[2*gtQCl]` | Extended sync counters (gradPostGEMM + combine tasks) |

## Sync Counter Layout

The sync counters use two separate regions within `tQS` (which points to `tSA()`):
```cpp
tQS[0, gtQCl)           // gradPostGEMM tile sync
tQS[gtQCl, 2*gtQCl)     // gradCombine + gradGateCombine sync
```

Note: `sQ()` (queue management) is a separate buffer that starts at `tSA() + 2*gtQCl`, not within tSA.

Combine task synchronization uses offset `combineSyncIdx = syncIdx + gtQCl`:
```cpp
// processor.cuh:1531-1543
const auto combineSyncIdx = rCurrentTask.syncIdx + bookkeeping.gtQCl;
// P2P: tNx gradCombine + 1 gradGateCombine = tNx+1 tasks per syncIdx
// Remote: tNx gradCombine + 1 gradGateCombine = tNx+1 tasks per syncIdx
const auto threshold = tNx + 1;
enqueue = atomicAdd(pA.tQS + combineSyncIdx, 1U) + 1 == threshold;
```

## Final Decoder Task Emission

**P2P final decoder** (packet.cuh) emits gradCombine and gradInputCombine **unconditionally**:
```cpp
{
    Task inputTask{
        TaskType::gradInputCombine,
        tokenIndices,                    // aData: TPS array
        ...
    };
    inputTask.cData[0] = const_cast<cuda::std::byte*>(packet);  // Source: grad_input [M, H]
    inputTask.isPeerRemote = false;      // P2P accessible
    emitTask(inputTask);  // Always emitted
}
```
Note: gradGateCombine is NOT emitted by the final decoder. The initial decoder already emits it,
and emitting again would cause double-counting in gateBuffer via atomicAdd.

**Remote final decoder** (packet.cuh) emits gradCombine and gradInputCombine **conditionally**:
```cpp
// packet.cuh
const bool isRemoteExpert = (peerIdx != dA.epRank);
const auto totalTasks = isRemoteExpert ? (tNx + tNx) : tNx;  // gradCombine + gradInputCombine

// packet.cuh
if (isRemoteExpert) {
    for (uint i = 0; i < tNx; ++i) {
        const auto inputIdx = DQ::next(qIdx, tNx + i);  // After tNx gradCombine tasks
        Task inputTask{TaskType::gradInputCombine, ...};
        inputTask.cData[0] = const_cast<cuda::std::byte*>(packet);  // grad_input [M, H]
        inputTask.cData[1] = const_cast<cuda::std::byte*>(packet);  // heap location (used)
        inputTask.isPeerRemote = true;
        dA.tQ[inputIdx] = inputTask;
    }
}
```
The `isRemoteExpert` check distinguishes:
- `peerIdx == epRank`: Signal from local GPU (shouldn't happen for remote decoder)
- `peerIdx != epRank`: Signal from remote expert GPU (normal case, emit gradInputCombine)

## Final Decoder Task Fields

| Field | Set By | Used By | Purpose |
|-------|--------|---------|---------|
| `flags` | Final subscriber | gradPreGEMM | Signal completion to token GPU |
| `rcData` | Final subscriber | gradPreGEMM | Remote destination for grad_input |
| `bData[0]` | Final subscriber | gradPreGEMM | W1 weights for grad @ W1^T |
| `bData[1]` | Final subscriber | gradPostGEMM | W2 weights for grad @ W2^T |

Saved activations (z1, z2) are derived from `cData[1]` (xM row pointer) combined with `bookkeeping.z1()`/`bookkeeping.z2()` global buffers

## Scheduler Stride Configuration

The backward pass requires more task slots per syncIdx than the forward pass:

| Pass | Slot Size | Tasks |
|------|-----------|-------|
| Forward | `TN + TNx` | postGEMM (TNx) |
| Backward | `TN + 2*TNx` | gradPostGEMM (TN) + gradGateGEMM (TNx) + gradPreGEMM (TNx) |

The scheduler's `blockQStride` is parameterized via `os::start<..., IsBackward>`:
- `os::start<processors, d>` (forward): uses default stride `TN + TNx`
- `os::start<processors, d, true>` (backward): passes stride `TN + 2*TNx`

This ensures the scheduler reads ptQ at the same stride that `notifyGradientImpl` writes:
```cpp
// os.cuh - compute stride based on IsBackward
constexpr auto blockQStride = IsBackward
    ? (ACC::TN::value + 2 * ACC::TNx::value)
    : (ACC::TN::value + ACC::TNx::value);
scheduler::start<processors, blockQStride>(...);
```

### Secondary tQH Domain (Backward only)

gradPreGEMM tasks use a separate tQH domain to avoid contention with gradPostGEMM/gradGateGEMM:

| Domain | tQH Index | Task Types | ptQ Offset |
|--------|-----------|------------|------------|
| Primary | `syncIdx` | gradPostGEMM, gradGateGEMM | 0, TN |
| Secondary | `syncIdx + gtQCl` | gradPreGEMM | TN + TNx |

- tQH extended to `2*gtQCl` entries (types.cuh)
- `notifyGradPreGEMM` writes to `tQH[syncIdx + gtQCl]`
- Scheduler uses `secondaryDomainOffset = TN+TNx` to compute ptQ offset for secondary domain

## Pointer Offset Semantics

### Column-Only tileIdx in Backward Pass

Unlike the forward pass where `tileIdx = offset + taskIdx` encodes both row and column information, the backward pass uses **column-only tileIdx** with row offsets baked into pointers:

```cpp
// notifyGradientImpl (processor.cuh:1167-1169)
// tileIdx is column tile only - cData pointers are already row-offset
const auto tileIdx = taskIdx;  // NOT offset + taskIdx
```

The initial decoder pre-computes row-offset pointers and stores them in `cData`:
- `cData[0] = packet + rowIdx * BLOCK_M * H * sizeof(Element)` (gradient buffer)
- `cData[1] = xMBase + rowIdx * BLOCK_M * P * sizeof(Element)` (intermediate buffer)

Downstream tasks (`gradPostGEMM`, `gradPreGEMM`) use `tileIdx` as a column index only (range `[0, tNx)`), avoiding double row-offset calculation.

### Function Pointer Expectations

| Function | Pointer Expectation | tileIdx Handling |
|----------|--------------------|-----------------------------|
| `fGETGrad` | Row-offset pointer (from `cData`) | Column tile index only (`idx2crd` sees row=0) |
| `fGET` | Row-offset pointer (from `cData`) | Column tile index only |
| `computeWeightGradients` | Pre-offset pointer | No internal offset handling |

`fGETGrad` is the backward-only GEMM that applies the activation derivative *before* the matrix multiply (mathematically correct ordering), then uses tensor cores for the GEMM. It computes `(grad * act'(z)) @ W` where the elementwise activation derivative happens before the matmul. Both `gradPostGEMM` and `gradPreGEMM` use `fGETGrad` instead of `fGET`.

When `notifyGradPreGEMM` creates tasks, `aData` is set to the row-offset pointer from `cData[gradIndex]`. All tasks for different column tiles share the same row-offset base pointer.

For `gradPreGEMM`:
- `fGETGrad` reads/writes at column tile offset (uses `tileIdx` as column index)
- `originalInput` from heap is already offset: `heap::advance(..., batchIdx * BLOCK_M)`
- `aData` is passed directly to `computeWeightGradients` without offset


## Why gradInputCombine Exists
- Forward pass `combine` writes expert outputs directly to `moeOutput[tokenIdx]` using the TPS mapping.
- Backward pass reverses this: `gradPreGEMM` computes `grad_input` on the **expert GPU**, but it must be accumulated to `gradInputBasePtr[tokenIdx]` on the **token GPU**.
- For **all experts** (P2P accessible), the final decoder unconditionally emits `gradInputCombine` to read from xM via P2P and accumulate to `gradInputBasePtr`.
- For **remote experts**, `gradPreGEMM` additionally sends grad_input via NVSHMEM to the heap, then signals the token GPU. The final subscriber creates `gradInputCombine` tasks that read from the heap location (`cData[1]`) and atomicAdd to `gradInputBasePtr`.
- **gradInputCombine performs NO probability scaling** - `gradCombine` already applied `prob` when splitting gradients to experts. gradInputCombine is pure aggregation.


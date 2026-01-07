# Memory Layout for Backwards Pass
layouts are defined in csrc/include/flashmoe/types.cuh.

## gradGateCombine

**Reads:**
- `gradOutputBasePtr` [S, H] — global grad_output buffer, indexed by tokenIdx
- `z2Base` (bookkeeping.z2()) [world×nLx×pEC, H] — saved expert outputs (y_e)
  - Row offset derived from `cData[1]` (xMPtr): `baseRowIdx = (xMPtr - xMBase) / P`
  - Accesses `z2RowBase = z2Base + baseRowIdx * H`
- `aData` → TPS array [tileSize] — token indices

**Writes:**
- `gGateCombine()` [S, E] — routing gradient accumulation buffer
  - Write index: `tokenIdx * E + globalExpertIdx`
  - Value: `Σ_h gradOutput[tokenIdx, h] * z2[row, h]` (dot product)
  - Uses atomicAdd for accumulation across tiles

## gradGateGEMM

**Reads:**
- `bookkeeping.gGateCombine()` [S, E] — accumulated routing gradients
- `bookkeeping.gateRoutingScores()` [S, E] — saved routing scores (logits) from forward pass
- `cData[0]` → `gGateCombine()` [S, E] — accumulated routing gradients
- `bData[0]` → `hiddenStatesPtr` [S, H] — original input hidden states
- `bData[1]` → `gateWeightsPtr` [H, E] — gate weights
- `aData` → TPS array — token indices

**Writes:**
- `bookkeeping.gGateW()` [H, E] — gate weight gradients (atomicAdd accumulation)
- `dData[0]` → `gradInputBasePtr` [S, H] — input gradients from routing

## gradCombine

**Reads:**
- `gradOutputBasePtr` [S, H] — global grad_output buffer, indexed by tokenIdx from TPS
- `aData` → TPS array [tileSize] — token indices and routing probabilities
- `routingScores` [S, E] — saved routing scores from forward pass (via sW/scale parameter)

**Writes:**
- `cData[0]` (rowPacket) [BLOCK_M, H] — per-row destination for split gradients
  - Row index: token position within tile (0 to tileSize-1)
  - Col index: tileIdx selects column tile (0 to tNx-1)
  - Write pattern: `gExpertGrad[row, col] = gradOutput[tokenIdx, col] / probability`

**Passes downstream:**
- `cData[1]` (rowXM) — xM pointer for notifyGradient chain

## gradPostGEMM

Computes `grad_intermediate = (grad_output * act'(z2)) @ W2^T` and `dW2 = grad_z2^T @ a1`.

**Reads:**
- `aData` [M, H] — split grad_output from gradCombine (row-offset pointer via cData[gradIndex])
- `bData[1]` (w2Index) [H, P] — W2 expert weights for second layer
- `cData[1]` (w2Index) → rowXM [M, P] — intermediate activations from forward pass
  - Used to derive `z2Activation = bookkeeping.z2() + rowIdx * H` [M, H]
  - `rowIdx = (xMPtr - xMBase) / P`
  - Also used as `a1Activation = xMPtr` [M, P] for weight gradients
- `bookkeeping.z2()` [world×nLx×pEC, H] — saved pre-activations (before output activation)

**Writes:**
- `cData[1]` (w2Index) [M, P] — grad_intermediate output
  - Shape: [M, P] where M = tileSize, P = intermediate dimension
  - Computed by `fGETGrad`: `(grad_output * act'(z2)) @ W2^T`
  - Column tile indexed by `tileIdx` (range [0, tN))
- `gW() + expertIdx * (2*P*H + P + H) + P*H` → dW2Buffer [H, P]
  - Weight gradients for W2
  - Computed by `computeWeightGradients<H, P>`: `grad_z2^T @ a1`
  - Uses atomicAdd for accumulation across tiles

**Passes downstream (via notifyGradPreGEMM):**
- `aData` = cData[1] = grad_intermediate [M, P] (new input for gradPreGEMM)
- `cData`, `bData`, `dData` propagated for z1 derivation and W1 access

**Sync:**
- Uses `tQS[syncIdx]` (NOT combined with gtQCl offset)
- Threshold: `tN` tiles (column tiles for [M, P] output)
- When threshold reached: emits `gradPreGEMM` tasks via `notifyGradPreGEMM`

## gradPreGEMM

Computes `grad_input = (grad_intermediate * act'(z1)) @ W1^T` and `dW1 = grad_intermediate^T @ original_input`.

**Reads:**
- `aData` [M, P] — grad_intermediate from gradPostGEMM (row-offset pointer via cData[gradIndex=1] from notifyGradPreGEMM)
- `bData[0]` (w1Index) [P, H] — W1 expert weights for first layer
- `cData[1]` (xMIndex) — xM row pointer used to derive z1 activation offset
  - `xMOffset = xMPtr - xMBase`
  - `z1Activation = bookkeeping.z1() + xMOffset` [M, P]
  - Note: cData[0] is packet (output dest), cData[1] is xM (for offset calculation)
- `bookkeeping.z1()` [world×nLx×pEC, P] — saved pre-activations from forward preGEMM
- `heap::advance<0, 1>(sHeap, peer, localExpertIdx, tokenOffset)` [M, H] — original input from forward pass
  - Stage 0, Cell 1 of symmetric heap
  - Token offset: `batchIdx * BLOCK_M`

**Writes:**
- `cData[0]` (w1Index) [M, H] — grad_input output
  - Shape: [M, H] where M = tileSize, H = hidden dimension
  - Computed by `fGETGrad`: `(grad_intermediate * act'(z1)) @ W1^T`
  - Column tile indexed by `tileIdx` (range [0, tNx))
- `gW() + expertIdx * (2*P*H + P + H)` → dW1Buffer [P, H]
  - Weight gradients for W1 (first P*H elements of expert gradient buffer)
  - Computed by `computeWeightGradients<P, H>`: `grad_intermediate^T @ original_input`
  - Uses atomicAdd for accumulation across tiles

**Signaling (last GEMM in backward chain):**
- For remote peers (`isPeerRemote == true`):
  - Sync threshold: `tN + tNx` tiles (gradPostGEMM tiles + gradPreGEMM tiles)
  - When threshold reached: `nvshmem_putmem_signal_nbi(rcData, cData[w1Index], tileSize * H * sizeof(Element), ...)`
  - Sends grad_input to token GPU's heap for gradInputCombine
- For P2P peers (`isPeerRemote == false`):
  - Uses `atomicExch_system` to set flags for completion notification
  - grad_input remains in local xM buffer for P2P access

## gradInputCombine

**Reads:**
- `aData` → TPS array [tileSize] — token-probability-score entries
  - Only `tokenIdx` field is used (probability is not needed here)
  - Loaded to shared memory (`sTPS`) for coalesced access
- Source selection based on `isPeerRemote`:
  - **P2P** (`isPeerRemote == false`): `cData[0]` [M, H] — grad_input from xM buffer (P2P accessible)
  - **Remote** (`isPeerRemote == true`): `cData[1]` [M, H] — grad_input from heap (NVSHMEM transferred)

**Writes:**
- `flashmoe::moe::gradInputBasePtr` [S, H] — global grad_input accumulation buffer
  - Write pattern: `gradInputBasePtr[tokenIdx, h] += gradInputSrc[t, h]`
  - Uses atomicAdd for accumulation across multiple experts routing to same token
  - Each token may receive gradients from up to E experts (top-k routing)

**P2P Task Creation (packet.cuh:938-954):**
```cpp
// Unconditionally emit for all P2P experts
{
    Task inputTask{TaskType::gradInputCombine, tokenIndices, ...};
    inputTask.cData[0] = packet;  // grad_input [M, H] from xM
    inputTask.isPeerRemote = false;
    emitTask(inputTask);  // Always emitted
}
```

**Remote Task Creation (packet.cuh:1119-1140):**
```cpp
// Only for remote experts (isRemoteExpert = peerIdx != epRank)
const bool isRemoteExpert = (peerIdx != dA.epRank);
if (isRemoteExpert) {
    for (uint i = 0; i < tNx; ++i) {
        Task inputTask{TaskType::gradInputCombine, tokenIndices, ...};
        inputTask.cData[0] = packet;  // grad_input [M, H]
        inputTask.cData[1] = packet;  // heap location (used for remote)
        inputTask.isPeerRemote = true;
        dA.tQ[inputIdx] = inputTask;
    }
}
```

## Task Queues (tQ and ptQ)

**Memory Layout:**
```
bookTask[0..sT-1]              → tQ  (stride task queue)
bookTask[sT..sT+prT-1]         → ptQ (pending task queue)
```

**tQ (Stride Task Queue):**
- Written by: Decoders (initial/final) in `packet.cuh`
- Size: `sT` tasks (computed per-subscriber capacity)
- Addressing: `tQ[slot]` where slot is obtained via `atomicAdd(tQHead, 1)`
- Scheduled by: Scheduler reads `tQHeads` (shared memory, per-subscriber)

**ptQ (Pending Task Queue):**
- Written by: Processors via `notifyNext` (forward) or `notifyGradientImpl` (backward)
- Size: `prT = world * nLx * TCM * ptQSlotSize` tasks
- Slot size (parameterized by pass type):
  - **Forward**: `ptQSlotSize = TN + TNx` (postGEMM tasks only)
  - **Backward**: `ptQSlotSize = TN + 2*TNx` (gradPostGEMM + gradGateGEMM + gradPreGEMM)
- Addressing: `ptQ[syncIdx * ptQSlotSize + slotOffset]`
  - Forward:
    - `notifyNext` (postGEMM): writes TNx tasks at offset 0
  - Backward:
    - `notifyGradient` (gradPostGEMM): writes TN tasks at offset 0
    - `notifyGateGradient` (gradGateGEMM): writes TNx tasks at offset TN
    - `notifyGradPreGEMM` (gradPreGEMM): writes TNx tasks at offset TN + TNx (uses secondary tQH domain)
- Scheduled by: Scheduler reads `tQH` (global, `bookkeeping.tQH()`, per-syncIdx)
  - Scheduler uses matching `blockQStride` template parameter:
    - Forward: `TN + TNx` (default in `scheduler::start`)
    - Backward: `TN + 2*TNx` (passed from `os::start<..., IsBackward=true>`)
- **Secondary tQH Domain (Backward only):**
  - tQH extended to `2*gtQCl` entries (was `gtQCl`)
  - Primary domain: `tQH[syncIdx]` → gradPostGEMM, gradGateGEMM (offset 0 and TN)
  - Secondary domain: `tQH[syncIdx + gtQCl]` → gradPreGEMM (offset TN + TNx)
  - Scheduler uses `secondaryDomainOffset = TN+TNx` to compute ptQ offset for secondary domain tasks

**Scheduling Flow:**
1. Producer increments head counter (`tQHeads` for tQ, `tQH[syncIdx]` for ptQ)
2. Scheduler polls head counters to discover new tasks
3. Scheduler signals processor with encoded task location
4. Processor decodes signal and reads task from `tQ + decodedSignal`
   - For ptQ tasks, `decodedSignal >= sT` so it reads from ptQ region

**Why Backward ptQ Slot Size is TN + 2*TNx (not larger):**

Backward pass has 6 task types, but only 3 flow through ptQ. The split:

| Queue | Task Types | Created By |
|-------|------------|------------|
| **tQ** | `gradCombine`, `gradGateCombine`, `gradInputCombine` | Decoders (packet.cuh) |
| **ptQ** | `gradPostGEMM`, `gradGateGEMM`, `gradPreGEMM` | Processors (notifyGradientImpl) |

Only processor-to-processor task chains use ptQ. Decoder-created tasks always use tQ.
Per syncIdx, the ptQ tasks sum to: TN (gradPostGEMM) + TNx (gradGateGEMM) + TNx (gradPreGEMM) = **TN + 2*TNx**.

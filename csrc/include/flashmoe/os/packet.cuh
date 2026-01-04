/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH
#include <nvshmem.h>

#include "../types.cuh"
#include "../atomics.cuh"

namespace flashmoe::packet {
    template<
        unsigned int blocks,
        DropTokens d = DropTokens::yes,
        unsigned int superBlockSize = ACC::SBZ::value,
        unsigned int H = ACC::H::value,
        unsigned int E = ACC::E::value,
        unsigned int EC = ACC::EC::value,
        unsigned int pEC = ACC::pEC::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        unsigned int batch = cute::min(cute::ceil_div(ACC::EC::value, superBlockSize), 16U),
        typename Activations
    >
    requires (isTensor<Activations>::value && cutlass::is_pow2<batch>::value)
    __forceinline__ __device__
    void dispatch(const Activations& activations,
        cuda::std::byte* __restrict__ const& workspace, const uint16_t& rSeqBit) {
        static_assert(sizeof(SignalPayload<>) == sizeof(ull_t) && alignof(SignalPayload<>) == alignof(ull_t));
        static_assert(sizeof(flagsType) == sizeof(ull_t) && alignof(flagsType) == alignof(ull_t));
        using Element = typename Activations::value_type;
        using NativeElement = typename ToCDx<Element>::T;
        // Below is always true, but we assert to ensure
        static_assert(sizeof(NativeElement) == sizeof(Element) && alignof(NativeElement) == alignof(Element));
        static_assert(blocks % superBlockSize == 0);
        // Map a static set of blocks to an expert and stride as thus
        constexpr auto numSuperBlocks = blocks / superBlockSize;
        const auto superBlockIdx = blockIdx.x / superBlockSize;
        const auto lBid = blockIdx.x % superBlockSize;
        const bool isLeader = !lBid && !threadIdx.x;

        // cache
        const auto* __restrict__ tP = bookkeeping.tP();
        const auto epRank = bookkeeping.rank;
        auto* __restrict__ pSA = bookkeeping.pSA();
        auto* __restrict__ sHeap = bookkeeping.sHeap;
        auto* __restrict__ flags = bookkeeping.flags;
        constexpr uint16_t gradStart = static_cast<uint16_t>(SignalConstants::gradSequenceStart);
        constexpr uint16_t gradEnd = gradStart + 2;
        const bool isGradientSeq = rSeqBit >= gradStart && rSeqBit <= gradEnd;

        const auto tokenIds = make_tensor(cute::make_gmem_ptr(tP),
            cute::Layout<cute::Shape<cute::Int<E>, cute::Int<pEC>>,
                cute::Stride<cute::Int<pEC>, cute::_1>>{});
        static_assert(cuda::std::is_same_v<TPS, typename decltype(tokenIds)::value_type>);
        /// Populate Data Structures
        const auto* __restrict__ enL = CAST_TO(PEL, workspace);
        const auto* __restrict__ eC = bookkeeping.eC();
        const auto* __restrict__ eL = bookkeeping.pEL();
        constexpr auto oT = E * sizeof(PEL);
        auto* __restrict__ seC = CAST_TO(uint, workspace + oT);

        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            // TODO eliminate bank conflicts
            CAST_TO(PEL, workspace)[i] = eL[i];
            seC[i] = eC[i];
        }
        constexpr auto oT2 = oT + E * sizeof(uint);
        auto* __restrict__ sPTT = CAST_TO(uint, workspace + oT2);
        const auto world = bookkeeping.world;
        #pragma unroll
        for (uint i = threadIdx.x; i < world; i += threads) {
            sPTT[i] = 0U; // clear before accumulation
        }
        __syncthreads();
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            const auto peer = enL[i].peer;
            const auto nTokens = d == DropTokens::yes ? cute::min(seC[i], EC) : seC[i];
            atomicAdd_block(sPTT + peer, Bookkeeping::tiles<BLOCK_M>(nTokens));
        }
        __syncthreads();

        // Update encoding lookup table
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            auto* __restrict__ peL = CAST_TO(PEL, workspace) + i;
            const auto peer = peL->peer;
            peL->eC = seC[i];
            peL->pTTt = static_cast<uint16_t>(sPTT[peer]);
        }
        __syncthreads();
        constexpr auto exL = cute::ceil_div(E, numSuperBlocks);
        static_assert(alignof(PEL) % alignof(uint) == 0);
        constexpr auto pZ = rTCL<PEL>(E * sizeof(PEL));
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(uint, workspace + pZ)),
            cute::Layout<cute::Shape<cute::Int<threads>, cute::Int<batch>>>{});
        cutlass::AlignedArray<uint, batch> rTID{};
        #pragma unroll
        for (uint i = 0; i < exL; ++i) {
            // We do this swizzling to mitigate potential congestion during communication
            const auto swizzleIdx = (i + superBlockIdx) % exL;
            if (const auto expertIdx = superBlockIdx + swizzleIdx * numSuperBlocks; expertIdx < E) {
                const auto lI = enL[expertIdx];
                const auto flagOffset = epRank * lI.nLocalExperts + lI.expertLocalIdx;
                const auto routedTokens = d == DropTokens::yes ?
                    cute::min(lI.eC, EC) : lI.eC;
                auto* __restrict__ peerHeap = lI.isRemote ?
                    heap::advance<0, 0>(sHeap, lI.peer, lI.expertLocalIdx) :
                heap::advance<0, 1>(lI.remoteSHeap, epRank, lI.expertLocalIdx);
#if FLASHMOE_DEBUG
                if (!threadIdx.x && !blockIdx.x && expertIdx == 0 && !lBid) {
                    printf("DEBUG FWD_HEAP: rank=%d expert=%u isRemote=%u peerHeap=%p remoteSHeap=%p sHeap=%p epRank=%u peer=%u\n",
                           nvshmem_my_pe(), expertIdx, lI.isRemote ? 1U : 0U, peerHeap, lI.remoteSHeap, sHeap, epRank, lI.peer);
                }
#endif
                if (routedTokens) {
                    const auto partition = routedTokens / superBlockSize +
                        (lBid < routedTokens % superBlockSize);
                    const auto trips = partition / batch;
                    const auto residueCount = partition - trips * batch;
#if FLASHMOE_DEBUG
                    // Debug: Show dispatch TPS->heap mapping for first expert, first block
                    if (!threadIdx.x && !blockIdx.x && expertIdx == 0 && !lBid) {
                        // Sample first 4 TPS entries for this expert
                        const auto tps0 = tokenIds(expertIdx, 0);
                        const auto tps1 = tokenIds(expertIdx, 1);
                        const auto tps2 = tokenIds(expertIdx, 2);
                        const auto tps3 = tokenIds(expertIdx, 3);
                        printf("DEBUG dispatch rank=%d expert=%u routedTokens=%u "
                               "TPS[0..3]={tok=%u,prob=%.4f},{tok=%u,prob=%.4f},{tok=%u,prob=%.4f},{tok=%u,prob=%.4f}\n",
                               nvshmem_my_pe(), expertIdx, routedTokens,
                               tps0.tokenIdx, static_cast<float>(tps0.probability),
                               tps1.tokenIdx, static_cast<float>(tps1.probability),
                               tps2.tokenIdx, static_cast<float>(tps2.probability),
                               tps3.tokenIdx, static_cast<float>(tps3.probability));
                    }
#endif
// #if FLASHMOE_DEBUG
//                     if (isLeader && !lBid) {
//                         constexpr unsigned int debugMaxExperts = 8;
//                         if (expertIdx < debugMaxExperts) {
//                             const auto totalTiles = Bookkeeping::tiles<BLOCK_M>(routedTokens);
//                             printf("DEBUG dispatch rank=%d sb=%u expert=%u peer=%u routed=%u tiles=%u partition=%u residue=%u remote=%u gradSeq=%u\n",
//                                    nvshmem_my_pe(),
//                                    rSeqBit,
//                                    expertIdx,
//                                    lI.peer,
//                                    routedTokens,
//                                    totalTiles,
//                                    partition,
//                                    residueCount,
//                                    static_cast<unsigned>(lI.isRemote),
//                                    static_cast<unsigned>(isGradientSeq));
//                         }
//                     }
// #endif
                    if (trips) {
                        // prefetch
                        // global -> shared
                        #pragma unroll
                        for (uint k = 0; k < batch; ++k) {
                            const auto [tokenIdx, _] = tokenIds(expertIdx, lBid + k * superBlockSize);
                            sC(threadIdx.x, k) = tokenIdx;
                        }
                    }
                    for (uint j = 0; j < trips; ++j) {
                        // shared -> registers
                        #pragma unroll
                        for (uint k = 0; k < batch; ++k) {
                            rTID[k] = sC(threadIdx.x, k);
                        }
                        if (j + 1 < trips) {
                            // if needed, start loads for the next batch after draining shared memory
                            // global -> shared
                            #pragma unroll
                            for (uint k = 0; k < batch; ++k) {
                                const auto [tokenIdx, _] = tokenIds(expertIdx,
                                    lBid + (k + (j + 1) * batch) * superBlockSize);
                                sC(threadIdx.x, k) = tokenIdx;
                            }
                        }
                        // Communicate these tokens
                        if (!isGradientSeq) {
                            #pragma unroll
                            for (uint k = 0; k < batch; ++k) {
                                const auto tokenIdx = rTID[k];
                                const auto intraIdx = lBid + (k + j * batch) * superBlockSize;
                                auto* __restrict__ localPH = peerHeap + intraIdx * H * sizeof(Element);
                                const auto* __restrict__ aP = CONST_CAST_TO(NativeElement, &activations(tokenIdx, 0));
                                // coalesced copy
                                constexpr auto tL = H / threads;
                                auto* __restrict__ nPH = CAST_TO(NativeElement, localPH);
                                #pragma unroll
                                for (uint l = 0; l < tL; ++l) {
                                    const auto idx = threadIdx.x + l * threads;
                                    nPH[idx] = __ldg(aP + idx);
                                }
                                if constexpr (H % threads != 0) {
                                    if (threadIdx.x < H % threads) {
                                        const auto idx = threadIdx.x + tL * threads;
                                        nPH[idx] = __ldg(aP + idx);
                                    }
                                }
                            }
                        }
                    }
                    // residue
                    if (const auto residue = residueCount; residue) {
                        // global -> shared
                        for (uint k = 0; k < residue; ++k) {
                            const auto [tokenIdx, _] = tokenIds(expertIdx, lBid + (k + trips * batch) * superBlockSize);
                            sC(threadIdx.x, k) = tokenIdx;
                        }
                        // shared -> registers
                        #pragma unroll
                        for (uint k = 0; k < batch; ++k) {
                            if (k < residue) {
                                rTID[k] = sC(threadIdx.x, k);
                            }
                        }
                        if (!isGradientSeq) {
                            #pragma unroll
                            for (uint k = 0; k < batch; ++k) {
                                if (k < residue) {
                                    const auto tokenIdx = rTID[k];
                                    const auto intraIdx = lBid + (k + trips * batch) * superBlockSize;
                                    auto* __restrict__ localPH = peerHeap + intraIdx * H * sizeof(Element);
                                    const auto* __restrict__ aP = CONST_CAST_TO(NativeElement, &activations(tokenIdx, 0));
                                    auto* __restrict__ nPH = CAST_TO(NativeElement, localPH);
                                    constexpr auto tL = H / threads;
                                    #pragma unroll
                                    for (uint l = 0; l < tL; ++l) {
                                        const auto idx = threadIdx.x + l * threads;
                                        nPH[idx] = __ldg(aP + idx);
                                    }
                                    if constexpr (H % threads != 0) {
                                        if (threadIdx.x < H % threads) {
                                            const auto idx = threadIdx.x + tL * threads;
                                            nPH[idx] = __ldg(aP + idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                    // all threads must fence to ensure their heap writes are visible to peer.
                    if (lI.isRemote) {
                        __threadfence();
                    }
                    else {
                        __threadfence_system();
                    }
                    __syncthreads();  // wait for all threads to complete their fences
                    if (!threadIdx.x) {
                        if (atomicIncrement(pSA + expertIdx) + 1 == superBlockSize) {
                            // I am in the last block, let's finalize this transfer.
                            const auto sigPayload = SignalPayload<PacketStage::initial>{
                                routedTokens,
                                lI.pTTt,
                                rSeqBit
                            };
#if FLASHMOE_DEBUG
                            // if (!isGradientSeq) {
                            //     printf("DEBUG fwd dispatch signal rank=%d block=%u sb=%u expert=%u peer=%u routed=%u totalTiles=%u remote=%u flag=%u flagPtr=%p\n",
                            //            nvshmem_my_pe(),
                            //            blockIdx.x,
                            //            rSeqBit,
                            //            expertIdx,
                            //            lI.peer,
                            //            routedTokens,
                            //            lI.pTTt,
                            //            static_cast<unsigned>(lI.isRemote),
                            //            flagOffset,
                            //            flags + flagOffset);
                            // }
                            // if (isGradientSeq) {
                            //     printf("DEBUG grad packet initial signal sb=%u expert=%u peer=%u routed=%u totalTiles=%u partition=%u residue=%u remote=%u flag=%u\n",
                            //            rSeqBit,
                            //            expertIdx,
                            //            lI.peer,
                            //            routedTokens,
                            //            lI.pTTt,
                            //            partition,
                            //            residueCount,
                            //            static_cast<unsigned>(lI.isRemote),
                            //            flagOffset);
                            // }
#endif
                            if (lI.isRemote) {
                                // do RDMA transfer + signal
                                nvshmem_putmem_signal_nbi(
                                    heap::advance<0, 1>(sHeap, epRank, lI.expertLocalIdx),
                                    peerHeap,
                                    sizeof(Element) * routedTokens * H,
                                    flags + flagOffset,
                                    *CONST_CAST_TO(flagsType, &sigPayload),
                                    NVSHMEM_SIGNAL_SET,
                                    lI.pe);
                            }
                            else {
                                // we've done the DMA transfer already, so we set the signal instead
                                atomicExch_system(CAST_TO(ull_t, lI.remoteSFlags + flagOffset),
                                    *CONST_CAST_TO(ull_t, &sigPayload));
                            }
                        }
                    }
                }
                else if (isLeader){
                    // single thread sends a noop packet to notify the remote peer
                    // Pack payload into a single signal word
                    const auto sigPayload = SignalPayload<PacketStage::initial>{
                        0U,
                        lI.pTTt,
                        rSeqBit
                    };
// #if FLASHMOE_DEBUG
//                     if (!isGradientSeq) {
//                         printf("DEBUG fwd dispatch noop rank=%d block=%u sb=%u expert=%u peer=%u totalTiles=%u remote=%u flag=%u\n",
//                                nvshmem_my_pe(),
//                                blockIdx.x,
//                                rSeqBit,
//                                expertIdx,
//                                lI.peer,
//                                lI.pTTt,
//                                static_cast<unsigned>(lI.isRemote),
//                                flagOffset);
//                     }
//                     if (isGradientSeq) {
//                         printf("DEBUG grad packet initial noop sb=%u expert=%u peer=%u totalTiles=%u remote=%u flag=%u\n",
//                                rSeqBit,
//                                expertIdx,
//                                lI.peer,
//                                lI.pTTt,
//                                static_cast<unsigned>(lI.isRemote),
//                                flagOffset);
//                     }
// #endif
                    if (lI.isRemote) {
                        // transmit signal
                        nvshmemx_signal_op(flags + flagOffset,
                            *CONST_CAST_TO(flagsType, &sigPayload), NVSHMEM_SIGNAL_SET, lI.pe);
                    }
                    else {
                        // Better to use below than the volatile
                        // write operation used in the public-facing API
                        atomicExch_system(CAST_TO(ull_t, lI.remoteSFlags + flagOffset),
                            *CONST_CAST_TO(ull_t, &sigPayload));
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x / WARP_SIZE == 1) {
            uint clearEC = 0U;
            const auto laneID = threadIdx.x % WARP_SIZE;
            if (!laneID) {
                constexpr auto expected = ACC::DBZ::value + 1;
                __threadfence();
                clearEC = atomicIncrement(bookkeeping.eCSync()) + 1 == expected;
            }
            __syncwarp();
            clearEC = __shfl_sync(0xffffffff, clearEC, 0);
            if (clearEC) {
                auto* __restrict__ bEC = bookkeeping.eC();
                constexpr auto tL = ACC::E::value / WARP_SIZE;
                for (uint i = 0; i < tL; ++i) {
                    bEC[laneID + i * WARP_SIZE] = 0U;
                }
                if constexpr (constexpr auto residue = ACC::E::value % WARP_SIZE; residue != 0) {
                    if (laneID < residue) {
                        bEC[laneID + tL * WARP_SIZE] = 0U;
                    }
                }
            }
        }
    }

    // Resident in registers
    struct DecoderArg {
        cuda::std::byte* sHeap;
        Task* tQ;
        flagsType* sFlags;
        const unsigned int nLx;
        const unsigned int epRank;
        __device__
        DecoderArg(
            cuda::std::byte* const& _sHeap,
            Task* const& _tQ,
            flagsType* const& _flags) :
        sHeap(_sHeap), tQ(_tQ), sFlags(_flags),
        nLx(bookkeeping.nLx), epRank(bookkeeping.rank) {}
    };
    /// Decodes packets emitted by subscribers.
    template<
        PacketStage s,
        PeerConnectivity p,
        typename Element = void,
        JobMode m = JobMode::forward
    >
    struct Decoder;

    template<
        PeerConnectivity p,
        typename Element,
        JobMode m
    >
    struct Decoder<PacketStage::initial, p, Element, m> {
        static_assert(flashmoe::TensorValueType<Element>);
        __device__ __forceinline__
        void operator()(const DecoderArg& dA,
            cuda::std::byte* const& sHeap,
            flagsType* const& flags,
            const cuda::std::byte* const& packet,
            uint const& routedTokens,
            unsigned int const& localExpertIdx,
            cuda::std::byte* __restrict__ const& pGB, //postGEMM buffer
            const cuda::std::array<const cuda::std::byte*, GEMMs>& weights,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& bias,
            unsigned int const& peer, // relative to the EP group
            unsigned int const& gPeer, // relative to the global group, needed for network operations
            const uint& laneId,
            unsigned int& lTQHead,
            unsigned int* __restrict__ const& tQHead) const {
            constexpr auto jobTaskType = m == JobMode::forward ?
                TaskType::preGEMM : TaskType::gradPostGEMM;
            constexpr auto tN = ACC::TN::value;
            const auto qIdx = DQ::sNext(lTQHead);
            const auto fTilesM = routedTokens / BLOCK_M;
            // pad here to meet tile requirements
            const auto padM = Bookkeeping::pad<BLOCK_M>(routedTokens);
            // expert, peer offset
            const auto sO = ACC::TCM::value * (peer * dA.nLx + localExpertIdx);
            cuda::std::array<cuda::std::byte*, GEMMs> taskResults{};
            // Staging buffer for results of preGEMM
            taskResults[0] = pGB + ((peer * dA.nLx + localExpertIdx) * ACC::pEC::value * ACC::P::value * sizeof(Element));
            // Egress packet buffer
            auto* rcData = heap::advance<1, 1>(sHeap, dA.epRank, localExpertIdx);
            taskResults[1] = p == PeerConnectivity::remote ?
                heap::advance<1, 0>(sHeap, peer, localExpertIdx) : rcData;
            const auto wT = fTilesM * tN;
            const auto fS = wT / WARP_SIZE + (laneId < wT % WARP_SIZE);
            constexpr auto rT = tN % WARP_SIZE;
            const auto lS = tN / WARP_SIZE + (rT > 0 ? laneId < rT : 0);
            const auto tSlice = fS + (routedTokens % BLOCK_M == 0 ? 0 : lS);

#if FLASHMOE_DEBUG
            if (laneId == 0 && localExpertIdx == 0 && routedTokens > 0) {
                printf("DEBUG DECODER_RANKS: rank=%d peer(EP)=%u gPeer(global)=%u epRank=%u routedTokens=%u\n",
                       nvshmem_my_pe(), peer, gPeer, dA.epRank, routedTokens);
            }
#endif
            for (uint i = 0; i < fS; ++i) {
                const auto tileIdx = laneId + i * WARP_SIZE;
                const auto rowIdx = tileIdx / tN;
                dA.tQ[DQ::next(qIdx, i)] = Task{
                    jobTaskType,
                    packet,
                    weights,
                    taskResults,
                    bias,
                    rcData,
                    flags,
                    sO + rowIdx,
                    tileIdx,
                    padM,
                    static_cast<uint16_t>(BLOCK_M),
                    gPeer,
                    rowIdx,
                    p == PeerConnectivity::remote,
                    localExpertIdx
                };
            }

            // residue tile
            if (const auto residue = routedTokens - fTilesM * BLOCK_M; residue) {
                for (uint j = 0; j < lS; j++) {
                    const auto tileIdx = fTilesM * tN + laneId + j * WARP_SIZE;
                    dA.tQ[DQ::next(qIdx, fS + j)] = Task{
                        jobTaskType,
                        packet,
                        weights,
                        taskResults,
                        bias,
                        rcData,
                        flags,
                        sO + fTilesM,
                        tileIdx,
                        padM,
                        static_cast<uint16_t>(residue),
                        gPeer,
                        fTilesM,
                        p == PeerConnectivity::remote,
                        localExpertIdx
                    };
                }
            }

            if (tSlice) {
                lTQHead += tSlice;
                __threadfence();
                // notifies scheduler of work
                atomicAdd_block(tQHead, tSlice);
            }
        }
    };

    // Specialized initial gradient decoder that emits gradCombine/gradGateCombine
    // instead of gradPostGEMM, aligning with the expected flow:
    // gradCombine + gradGateCombine -> notifyGradient -> gradPostGEMM
    template<PeerConnectivity p, typename Element>
    struct Decoder<PacketStage::initial, p, Element, JobMode::gradient> {
        static_assert(flashmoe::TensorValueType<Element>);
        __device__ __forceinline__
        void operator()(const DecoderArg& dA,
            cuda::std::byte* const& sHeap,
            flagsType* const& flags,
            const cuda::std::byte* const& packet,
            uint const& routedTokens,
            unsigned int const& localExpertIdx,
            cuda::std::byte* __restrict__ const& pGB, //postGEMM buffer (xM)
            const cuda::std::array<const cuda::std::byte*, GEMMs>& weights,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& savedActivations,
            unsigned int const& peer, // relative to the EP group
            unsigned int const& gPeer, // relative to the global group
            const uint& laneId,
            unsigned int& lTQHead,
            unsigned int* __restrict__ const& tQHead,
            // Additional parameters for gradient combine path
            const cuda::std::byte* const& tokenIndices, // TPS for this expert
            const unsigned int& expertIdx) const { // global expert index

            constexpr auto tNx = ACC::TNx::value;
            constexpr auto TCM = ACC::TCM::value;
            constexpr auto P = ACC::P::value;
            constexpr auto pEC = ACC::pEC::value;
            const auto qIdx = DQ::sNext(lTQHead);
            const auto fTilesM = routedTokens / BLOCK_M;
            const auto padM = Bookkeeping::pad<BLOCK_M>(routedTokens);

            // xM location for this peer/expert
            const auto xMOffset = (peer * dA.nLx + localExpertIdx) * pEC * P * sizeof(Element);
            auto* xMLocation = pGB + xMOffset;

            // rcData for signaling
            auto* rcData = heap::advance<1, 1>(sHeap, dA.epRank, localExpertIdx);

            // Task distribution across warp
            const auto wT = fTilesM * tNx;
            const auto fS = wT / WARP_SIZE + (laneId < wT % WARP_SIZE);
            constexpr auto rT = tNx % WARP_SIZE;
            const auto lS = tNx / WARP_SIZE + (rT > 0 ? laneId < rT : 0);
            const auto tSlice = fS + (routedTokens % BLOCK_M == 0 ? 0 : lS);

            // Compute residue once for use in both gradCombine and gradGateCombine
            const auto residue = routedTokens - fTilesM * BLOCK_M;

            // Track which batches this lane emits gate tasks for (max ~TCM batches typically small)
            unsigned int myGateCount = 0;
            // Store batch indices for gate tasks (small array, TCM is typically 2-4)
            unsigned int gateBatches[8];  // Should be enough for typical TCM values

#if FLASHMOE_DEBUG
            if (laneId == 0 && localExpertIdx == 0 && routedTokens > 0) {
                printf("DEBUG INITIAL_GRAD_DECODER: rank=%d peer=%u expert=%u(global) routedTokens=%u fTilesM=%u\n",
                       nvshmem_my_pe(), peer, expertIdx, routedTokens, fTilesM);
            }
#endif
            // Full tiles - emit gradCombine and track gate tasks
            for (uint i = 0; i < fS; ++i) {
                const auto tileIdx = laneId + i * WARP_SIZE;
                const auto batchIdx = tileIdx / tNx;
                const auto tileWithinBatch = tileIdx % tNx;

                // syncIdx consistent with final decoder
                const auto sO = TCM * (peer * dA.nLx + localExpertIdx) + batchIdx;
                const auto syncIdx = sO;

                // Emit gradCombine
                Task gradTask{
                    TaskType::gradCombine,
                    tokenIndices,
                    cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                    BLOCK_M,  // tileSize for full tiles
                    tileWithinBatch,
                    expertIdx
                };
                gradTask.cData[0] = xMLocation;
                gradTask.cData[1] = const_cast<cuda::std::byte*>(packet);
                gradTask.syncIdx = syncIdx;
                gradTask.M = padM;
                gradTask.peerIdx = peer;
                gradTask.batchIdx = batchIdx;
                gradTask.isPeerRemote = (p == PeerConnectivity::remote);
                gradTask.bData = weights;
                gradTask.dData = savedActivations;
                gradTask.rcData = rcData;
                gradTask.flags = flags;
                dA.tQ[DQ::next(qIdx, i)] = gradTask;

                // Record gate task for this batch (at first tile of batch)
                if (tileWithinBatch == 0 && myGateCount < 8) {
                    gateBatches[myGateCount++] = batchIdx;
                }
            }

            // Residue tile
            if (residue) {
                for (uint j = 0; j < lS; j++) {
                    const auto tileIdx = fTilesM * tNx + laneId + j * WARP_SIZE;
                    const auto batchIdx = fTilesM; // residue batch
                    const auto tileWithinBatch = tileIdx - fTilesM * tNx;

                    const auto sO = TCM * (peer * dA.nLx + localExpertIdx) + batchIdx;
                    const auto syncIdx = sO;

                    Task gradTask{
                        TaskType::gradCombine,
                        tokenIndices,
                        cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                        residue,  // tileSize for residue tile
                        tileWithinBatch,
                        expertIdx
                    };
                    gradTask.cData[0] = xMLocation;
                    gradTask.cData[1] = const_cast<cuda::std::byte*>(packet);
                    gradTask.syncIdx = syncIdx;
                    gradTask.M = padM;
                    gradTask.peerIdx = peer;
                    gradTask.batchIdx = batchIdx;
                    gradTask.isPeerRemote = (p == PeerConnectivity::remote);
                    gradTask.bData = weights;
                    gradTask.dData = savedActivations;
                    gradTask.rcData = rcData;
                    gradTask.flags = flags;
                    dA.tQ[DQ::next(qIdx, fS + j)] = gradTask;

                    // Record gate task for residue batch (at first tile)
                    if (tileWithinBatch == 0 && myGateCount < 8) {
                        gateBatches[myGateCount++] = batchIdx;
                    }
                }
            }

            // Per-subscriber queue accounting: each subscriber advances its own queue
            // by the tasks it actually emitted, placing gate tasks after local combine tasks
            const auto totalTasks = tSlice + myGateCount;
            if (totalTasks) {
                // Emit gate tasks after this subscriber's combine tasks
                for (uint g = 0; g < myGateCount; ++g) {
                    const auto batch = gateBatches[g];
                    const auto sO = TCM * (peer * dA.nLx + localExpertIdx) + batch;
                    // Gate tileSize: BLOCK_M for full batches, residue for residue batch
                    const auto gateTileSize = (batch < fTilesM) ? BLOCK_M : residue;

                    Task gateTask{
                        TaskType::gradGateCombine,
                        tokenIndices,
                        cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                        gateTileSize,
                        0U, // tileIdx for gate task
                        expertIdx
                    };
                    gateTask.cData[0] = const_cast<cuda::std::byte*>(packet);
                    gateTask.syncIdx = sO;
                    gateTask.M = padM;
                    gateTask.peerIdx = peer;
                    gateTask.batchIdx = batch;
                    gateTask.isPeerRemote = (p == PeerConnectivity::remote);

                    const auto gateSlot = tSlice + g; // local offset after combine tasks
                    dA.tQ[DQ::next(qIdx, gateSlot)] = gateTask;
                }

                lTQHead += totalTasks;
                __threadfence();
                atomicAdd_block(tQHead, totalTasks);
            }
        }
    };

    template<typename Element, JobMode m>
    struct Decoder<PacketStage::last, PeerConnectivity::p2p, Element, m> {
        __device__ __forceinline__
        void operator()(Task* __restrict__ const& tQ,
            unsigned int& lTQHead,
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            const unsigned int& nTokens,
            const unsigned int& tileIdx,
            unsigned int* __restrict__ const& tQHead,
            const unsigned int& expertIdx) const {
            constexpr auto jobTaskType = m == JobMode::forward ?
                TaskType::combine : TaskType::gradCombine;
            constexpr auto isGradient = m == JobMode::gradient;
            const auto emitTask = [&](const Task& task) {
                // Fix: Multiple blocks may share tQHead[tIdx]. Use device-scoped atomic
                // for unique slot reservation across all blocks. Visibility ordering:
                // atomicAdd → write → fence ensures task is visible when count is read.
                const auto slot = atomicAdd(tQHead, 1U);
                tQ[DQ::sNext(slot)] = task;
                __threadfence_system();
            };

            Task gradTask{
                jobTaskType,
                tokenIndices,
                cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                nTokens,
                tileIdx,
                expertIdx
            };
            // cData[0] = output destination for split gradients (writes to expert packet)
            gradTask.cData[0] = const_cast<cuda::std::byte*>(packet);
            emitTask(gradTask);
            if constexpr (isGradient) {
                Task gateTask{
                    TaskType::gradGateCombine,
                    tokenIndices,
                    cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                    nTokens,
                    tileIdx,
                    expertIdx
                };
                gateTask.cData[0] = const_cast<cuda::std::byte*>(packet);
                emitTask(gateTask);
            }
        }
    };

    // Specialized P2P gradient decoder for gradPreGEMM completion signals.
    // Only emits gradInputCombine - gradCombine/gradGateCombine are now
    // handled by the initial gradient decoder to align with the expected flow:
    // initial decoder → gradCombine + gradGateCombine → notifyGradient → gradPostGEMM → gradPreGEMM
    // final decoder (this) ← gradPreGEMM completion → gradInputCombine
    template<typename Element>
    struct Decoder<PacketStage::last, PeerConnectivity::p2p, Element, JobMode::gradient> {
        __device__ __forceinline__
        void operator()(
            const DecoderArg& dA,
            cuda::std::byte* __restrict__ const& pGB, // post GEMM buffer (xM)
            Task* __restrict__ const& tQ,
            unsigned int& lTQHead,
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            const unsigned int& nTokens,
            const unsigned int& tileIdx,
            unsigned int* __restrict__ const& tQHead,
            const unsigned int& expertIdx,
            const unsigned int& peerIdx,          // peer that sent the packet
            const unsigned int& localExpertIdx,   // local expert index on that peer
            const unsigned int& batchIdx,         // batch index from signal
            const cuda::std::array<const cuda::std::byte*, GEMMs>& weights,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& savedActivations,
            flagsType* const& flags) const {
            constexpr auto P = ACC::P::value;
            constexpr auto pEC = ACC::pEC::value;
            constexpr auto TCM = ACC::TCM::value;
            constexpr auto TN = ACC::TN::value;

            const auto emitTask = [&](const Task& task) {
                // Fix: Multiple blocks may share tQHead[tIdx]. Use device-scoped atomic
                // for unique slot reservation across all blocks. Visibility ordering:
                // atomicAdd → write → fence ensures task is visible when count is read.
                const auto slot = atomicAdd(tQHead, 1U);
                tQ[DQ::sNext(slot)] = task;
                __threadfence_system();
            };

            const auto xMOffset = (peerIdx * dA.nLx + localExpertIdx) * pEC * P * sizeof(Element);
            auto* xMLocation = pGB + xMOffset;

            const auto sO = TCM * (peerIdx * dA.nLx + localExpertIdx) + batchIdx;
            const auto syncIdx = sO + (tileIdx / TN);

            const auto padM = Bookkeeping::pad<BLOCK_M>(nTokens);

            // Emit only gradInputCombine for gradPreGEMM completion signals
            // gradCombine and gradGateCombine are now handled by the initial decoder
            {
                Task inputTask{
                    TaskType::gradInputCombine,
                    tokenIndices,                    // aData: TPS array
                    cuda::std::array<const cuda::std::byte*, GEMMs>{},
                    nTokens,
                    tileIdx,
                    expertIdx
                };
                inputTask.cData[0] = xMLocation;     // Source: grad_input from xM (P2P accessible)
                inputTask.syncIdx = syncIdx;
                inputTask.M = padM;
                inputTask.peerIdx = peerIdx;
                inputTask.batchIdx = batchIdx;
                inputTask.isPeerRemote = false;      // P2P accessible
                emitTask(inputTask);
            }
        }
    };

    template<typename Element, JobMode m>
    struct Decoder<PacketStage::last, PeerConnectivity::remote, Element, m> {
        __device__ __forceinline__
        void operator()(const DecoderArg& dA,
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            const unsigned int& nTokens,
            unsigned int& lTQHead,
            unsigned int* __restrict__ const& tQHead,
            const unsigned int& expertIdx) const {
            constexpr auto jobTaskType = m == JobMode::forward ?
                TaskType::combine : TaskType::gradCombine;
            constexpr auto isGradient = m == JobMode::gradient;
            const auto qIdx = DQ::sNext(lTQHead);
            constexpr auto tNx = ACC::TNx::value;
            for (uint i = 0; i < tNx; ++i) {
                Task gradTask{
                    jobTaskType,
                    tokenIndices,
                    cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                    nTokens,
                    i,
                    expertIdx
                };
                // cData[0] = output destination for split gradients (writes to expert packet)
                gradTask.cData[0] = const_cast<cuda::std::byte*>(packet);
                dA.tQ[DQ::next(qIdx, i)] = gradTask;
            }

            if constexpr (isGradient) {
                for (uint i = 0; i < tNx; ++i) {
                    const auto gateIdx = DQ::next(qIdx, tNx + i);
                    Task gateTask{
                        TaskType::gradGateCombine,
                        tokenIndices,
                        cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                        nTokens,
                        i,
                        expertIdx
                    };
                    gateTask.cData[0] = const_cast<cuda::std::byte*>(packet);
                    dA.tQ[gateIdx] = gateTask;
                }
            }

            constexpr auto totalTasks = tNx * (isGradient ? 2U : 1U);
            lTQHead += totalTasks;
            __threadfence();
            atomicAdd_block(tQHead, totalTasks);
        }
    };

    // Specialized remote gradient decoder for gradPreGEMM completion signals.
    // Only emits gradInputCombine - gradCombine/gradGateCombine are now
    // handled by the initial gradient decoder to align with the expected flow:
    // initial decoder → gradCombine + gradGateCombine → notifyGradient → gradPostGEMM → gradPreGEMM
    // final decoder (this) ← gradPreGEMM completion → gradInputCombine
    template<typename Element>
    struct Decoder<PacketStage::last, PeerConnectivity::remote, Element, JobMode::gradient> {
        __device__ __forceinline__
        void operator()(
            const DecoderArg& dA,
            cuda::std::byte* __restrict__ const& pGB, // post GEMM buffer (xM)
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            const unsigned int& nTokens,
            unsigned int& lTQHead,
            unsigned int* __restrict__ const& tQHead,
            const unsigned int& expertIdx,
            const unsigned int& peerIdx,          // peer that sent the packet
            const unsigned int& localExpertIdx,   // local expert index on that peer
            const unsigned int& batchIdx,         // batch index from signal
            const cuda::std::array<const cuda::std::byte*, GEMMs>& weights,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& savedActivations,
            flagsType* const& flags) const {
            constexpr auto P = ACC::P::value;
            constexpr auto pEC = ACC::pEC::value;
            constexpr auto TCM = ACC::TCM::value;
            constexpr auto TN = ACC::TN::value;
            constexpr auto tNx = ACC::TNx::value;

            const auto qIdx = DQ::sNext(lTQHead);

            const auto xMOffset = (peerIdx * dA.nLx + localExpertIdx) * pEC * P * sizeof(Element);
            auto* xMLocation = pGB + xMOffset;
            const auto sO = TCM * (peerIdx * dA.nLx + localExpertIdx) + batchIdx;
            const auto padM = Bookkeeping::pad<BLOCK_M>(nTokens);

            // Emit only gradInputCombine for gradPreGEMM completion signals (tNx tasks)
            // gradCombine and gradGateCombine are now handled by the initial decoder
            for (uint i = 0; i < tNx; ++i) {
                const auto inputSyncIdx = sO + (i / TN);
                Task inputTask{
                    TaskType::gradInputCombine,
                    tokenIndices,
                    cuda::std::array<const cuda::std::byte*, GEMMs>{},
                    nTokens,
                    i,
                    expertIdx
                };
                inputTask.cData[0] = xMLocation;
                inputTask.cData[1] = const_cast<cuda::std::byte*>(packet);  // heap location
                inputTask.syncIdx = inputSyncIdx;
                inputTask.M = padM;
                inputTask.peerIdx = peerIdx;
                inputTask.batchIdx = batchIdx;
                inputTask.isPeerRemote = true;
                dA.tQ[DQ::next(qIdx, i)] = inputTask;
            }

            // Only tNx gradInputCombine tasks
            constexpr auto totalTasks = tNx;
            lTQHead += totalTasks;
            __threadfence();
            atomicAdd_block(tQHead, totalTasks);
        }
    };
}
#endif //PACKET_CUH

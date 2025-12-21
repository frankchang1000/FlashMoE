/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_MOE_CUH
#define FLASHMOE_MOE_CUH

#include <cassert>
#include <cuda/std/array>

#include "../arch.cuh"
#include "../debug.cuh"
#include "../os/os.cuh"
#include "../os/processor/processor.cuh"
#include "fffn.cuh"
#include "gate.cuh"
#include "../telemetry.cuh"

namespace flashmoe::moe{
    __device__ const ACC::Element* hiddenStatesPtr = nullptr;
    __device__ const ACC::Element* gateWeightsPtr = nullptr;
    __device__ cuda::std::array<const cuda::std::byte*, GEMMs> savedActivationPtrs{};

    __host__ __forceinline__
    void exportSavedActivations(
        const cuda::std::array<const cuda::std::byte*, GEMMs>& activations
    ) {
        FLASHMOE_CHECK_CUDA(cudaMemcpyToSymbolAsync(
            savedActivationPtrs,
            &activations,
            sizeof(activations),
            0,
            cudaMemcpyHostToDevice,
            flashmoeStream
        ));
    }

    __host__ __forceinline__
    cuda::std::array<const cuda::std::byte*, GEMMs> makeBiasSavedActivations(
        const void* const& iP
    ) {
        const auto* __restrict__ base = CONST_CAST_TO(ACC::Element, iP);
        const auto lE = hostBookkeeping.nLx;
        const auto* __restrict__ gP = base + ACC::S::value * ACC::H::value;
        const auto* __restrict__ ePu = gP + ACC::PX::value * ACC::H::value;
        const auto* __restrict__ ePd = ePu + lE * ACC::P::value * ACC::H::value;
        const auto* __restrict__ bU = ePd + lE * ACC::H::value * ACC::P::value;
        const auto* __restrict__ bd = bU + lE * ACC::P::value;
        return cuda::std::array<const cuda::std::byte*, GEMMs>{
            CONST_CAST_TO(cuda::std::byte, bU),
            CONST_CAST_TO(cuda::std::byte, bd)
        };
    }
    template<
        uint threads,
        uint blocks,
        uint processors,
        CombineMode c,
        JobType jT,
        uint OZ,
        typename Element
    >
    __device__ __forceinline__
    void clearState(Element* __restrict__ const& outP) {
        // A barrier must occur after below otherwise, undefined behavior results.
        auto* __restrict__ pDB = bookkeeping.pDB();
        auto* __restrict__ sQ = bookkeeping.sQ();
        auto* __restrict__ gBp = bookkeeping.gBp();
        const auto gtQCl = bookkeeping.gtQCl;
        auto* __restrict__ tQH = bookkeeping.tQH();
        auto* __restrict__ tSA = bookkeeping.tSA();
        auto* __restrict__ pSA = bookkeeping.pSA();
        constexpr auto gBz = Bookkeeping::gBz();
        auto* __restrict__ eCSync = bookkeeping.eCSync();
        const auto idx = threads * blockIdx.x + threadIdx.x;
        if constexpr (c == CombineMode::multithreaded) {
            // clear output buffer
            for (uint i = idx; i < OZ; i += blocks * threads) {
                outP[i] = Element(0.0f);
            }
        }
        if constexpr (jT == JobType::training) {
            if (!idx) {
                const auto* gBase = reinterpret_cast<const BookType*>(gBp);
                const auto* tQHBase = bookkeeping.tQH();
                const auto diffEntries =
                    static_cast<long>(tQHBase - gBase);
            }
            __syncthreads();
            // clear loss buffers
            for (uint i = idx; i < gBz; i += blocks * threads) {
                gBp[i] = 0.0f;
            }
        }
        // clear processor doorbells
        for (uint i = idx; i < processors; i += blocks * threads) {
            pDB[i] = TQSignal{0U, 0U};
            sQ[i] = observed;
        }
        for (uint i = idx; i < gtQCl; i += blocks * threads) {
            tQH[i] = tQHeadGroundState;
            tSA[i] = 0U;
        }
        for (uint i = idx; i < ACC::E::value; i += blocks * threads) {
            pSA[i] = 0U;
        }
        if (!idx) {
            *eCSync = 0U;
        }
    }
    template<
        unsigned S = ACC::S::value,
        unsigned int P = ACC::P::value,
        unsigned int H = ACC::H::value,
        unsigned int PX = ACC::PX::value
    >
    __global__ __maxnreg__(ACC::PeakHardware::registers::value) void forward(
        const void* __restrict__ iP, /* A, G, B, D*/ void* __restrict__ oP /*G, O*/,
        const __grid_constant__ uint16_t sb) {
        using GPUType = ACC::PeakHardware;
        constexpr auto blocks = GPUType::blocks::value;
        constexpr auto processors = GPUType::OS::processorBlocks::value;
        constexpr auto sharedSize = ACC::sharedSize::value;
        constexpr auto threads = GPUType::OS::threads::value;
        constexpr auto d = ACC::DTK::value;
        constexpr auto c = ACC::CM::value;
        using Element = ACC::Element;
        using ElementC = ACC::ElementC;

        const auto lE = bookkeeping.nLx;
        // Salami slice pointers
        const auto* __restrict__ gP = CONST_CAST_TO(Element, iP) + S * H;
        const auto* __restrict__ ePu = gP + H * PX;
        const auto* __restrict__ ePd = ePu + lE * P * H;
        const auto* __restrict__ bU = ePd + lE * H * P;
        const auto* __restrict__ bd = bU + lE * P;
        auto* __restrict__ gOp = CAST_TO(Element, oP);
        auto* __restrict__ mOp = gOp + S * PX;
        __shared__ __align__(16) cuda::std::byte workspace[sharedSize];
        clearState<threads, blocks, processors, c, ACC::JT::value, S * H>(mOp);
        // Grid barrier ensures all blocks complete clearState before any proceeds.
        // This prevents race between pSA clearing and dispatch's atomicIncrement.
        gridBarrier();
#if FLASHMOE_DEBUG
        if (!threadIdx.x && !blockIdx.x) {
            const auto queueSpan = bookkeeping.queueSpanEntries();
            const auto gtQueueEntries = bookkeeping.gtQCl;
            const auto processorBlocks = ACC::PeakHardware::OS::processorBlocks::value;
            printf("DEBUG queue layout rank=%d sb=%u spanEntries=%lu gtQCl=%u procBlocks=%u eCount=%u\n",
                   nvshmem_my_pe(),
                   sb,
                   queueSpan,
                   gtQueueEntries,
                   processorBlocks,
                   ACC::E::value);
        }
#endif

        // prep tensors
        const auto activations = make_tensor(
            cute::make_gmem_ptr(CONST_CAST_TO(Element, iP)),
                    cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                            cute::Stride<cute::Int<H>, cute::_1>>{});
        const auto gateWeights = make_tensor(cute::make_gmem_ptr(gP),
            cute::Layout<cute::Shape<cute::Int<PX>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});
        if (!threadIdx.x && !blockIdx.x) {
            hiddenStatesPtr = CONST_CAST_TO(Element, activations.data().get());
            gateWeightsPtr = gP;
        }
        __syncthreads();
        // Experts Weights
        const auto expertsUp = make_tensor(cute::make_gmem_ptr(ePu),
            make_layout(make_shape(lE, cute::Shape<cute::Int<P>, cute::Int<H>>{}),
                cute::LayoutRight{}));
        const auto expertsDown = make_tensor(cute::make_gmem_ptr(ePd),
            make_layout(make_shape(lE, cute::Shape<cute::Int<H>, cute::Int<P>>{}),
                cute::LayoutRight{}));
        // Bias
        // Broadcast from vector to matrix
        const auto biasUp = make_tensor(cute::make_gmem_ptr(bU),
            make_layout(cute::make_shape(lE, P),
                cute::Stride<cute::Int<P>, cute::_1>{}));
        const auto biasDown = make_tensor(cute::make_gmem_ptr(bd),
            make_layout(make_shape(lE, cute::Int<H>{}),
                cute::Stride<cute::Int<H>, cute::_1>{}));

        // Output
        const auto gateOutput = make_tensor(cute::make_gmem_ptr(gOp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<PX>>,
                cute::Stride<cute::Int<PX>, cute::_1>>{});
        const auto moeOutput = make_tensor(cute::make_gmem_ptr(mOp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});

        gate::forward(activations, gateWeights, gateOutput, CAST_TO(ElementC, workspace));
        gridBarrier();
        if (!threadIdx.x && !blockIdx.x) {
            nvshmem_barrier_all(); // sync across ranks after gate forward
        }
        gridBarrier();

        // save expert counts for backward pass before they get cleared by dispatch
        if constexpr (ACC::JT::value == JobType::training) {
            const auto* __restrict__ eC = bookkeeping.eC();
            auto* __restrict__ sEC = bookkeeping.sEC();
            const auto idx = threads * blockIdx.x + threadIdx.x;
            for (uint i = idx; i < ACC::E::value; i += blocks * threads) {
                sEC[i] = eC[i];
            }
            gridBarrier();
        }

// #if FLASHMOE_DEBUG
//         if (!threadIdx.x && !blockIdx.x) {
//             const auto gateCount = static_cast<size_t>(ACC::S::value) * ACC::PX::value;
//             double gateSum = 0.0;
//             double gateMin = 0.0;
//             double gateMax = 0.0;
//             double gateAbsMax = 0.0;
//             bool gateInit = false;
//             const auto* gatePtr = gateOutput.data().get();
//             for (size_t i = 0; i < gateCount; ++i) {
//                 const double val = static_cast<double>(gatePtr[i]);
//                 gateSum += val;
//                 const double absVal = val >= 0.0 ? val : -val;
//                 if (!gateInit) {
//                     gateMin = gateMax = val;
//                     gateAbsMax = absVal;
//                     gateInit = true;
//                 } else {
//                     if (val < gateMin) {
//                         gateMin = val;
//                     }
//                     if (val > gateMax) {
//                         gateMax = val;
//                     }
//                     if (absVal > gateAbsMax) {
//                         gateAbsMax = absVal;
//                     }
//                 }
//             }
//             const double gateMean = gateCount ? gateSum / static_cast<double>(gateCount) : 0.0;
//             printf("DEBUG forward gateOutput stats rank=%d: sum=%.6e mean=%.6e min=%.6e max=%.6e absMax=%.6e\n",
//                    nvshmem_my_pe(),
//                    gateSum,
//                    gateMean,
//                    gateMin,
//                    gateMax,
//                    gateAbsMax);
//
//             const auto moeCount = static_cast<size_t>(ACC::S::value) * ACC::H::value;
//             double moeSum = 0.0;
//             double moeMin = 0.0;
//             double moeMax = 0.0;
//             double moeAbsMax = 0.0;
//             bool moeInit = false;
//             const auto* moePtr = moeOutput.data().get();
//             for (size_t i = 0; i < moeCount; ++i) {
//                 const double val = static_cast<double>(moePtr[i]);
//                 moeSum += val;
//                 const double absVal = val >= 0.0 ? val : -val;
//                 if (!moeInit) {
//                     moeMin = moeMax = val;
//                     moeAbsMax = absVal;
//                     moeInit = true;
//                 } else {
//                     if (val < moeMin) {
//                         moeMin = val;
//                     }
//                     if (val > moeMax) {
//                         moeMax = val;
//                     }
//                     if (absVal > moeAbsMax) {
//                         moeAbsMax = absVal;
//                     }
//                 }
//             }
//             const double moeMean = moeCount ? moeSum / static_cast<double>(moeCount) : 0.0;
//             printf("DEBUG forward moeOutput stats rank=%d: sum=%.6e mean=%.6e min=%.6e max=%.6e absMax=%.6e\n",
//                    nvshmem_my_pe(),
//                    moeSum,
//                    moeMean,
//                    moeMin,
//                    moeMax,
//                    moeAbsMax);
//         }
// #endif
        // For forward pass, savedActivations should point to the bias tensors
        const auto forwardSavedActivations = cuda::std::array<const cuda::std::byte*, GEMMs>{
            CONST_CAST_TO(cuda::std::byte, biasUp.data().get()),
            CONST_CAST_TO(cuda::std::byte, biasDown.data().get())
        };
// #if FLASHMOE_DEBUG
//         if (!threadIdx.x && blockIdx.x < 3) {
//             printf("DEBUG forward kernel block=%u/%u entering dispatch/processor section\n",
//                    blockIdx.x, blocks);
//         }
// #endif
        if (blockIdx.x + 1 < blocks) {
// #if FLASHMOE_DEBUG
//             if (!threadIdx.x && blockIdx.x < 3) {
//                 printf("DEBUG forward kernel block=%u calling dispatch (blockIdx < DBZ=%u: %d)\n",
//                        blockIdx.x, ACC::DBZ::value, blockIdx.x < ACC::DBZ::value);
//             }
// #endif
            if (blockIdx.x < ACC::DBZ::value) {
                packet::dispatch<ACC::DBZ::value, d, ACC::SBZ::value>(activations, workspace, sb);
// #if FLASHMOE_DEBUG
//                 if (!threadIdx.x && blockIdx.x < 3) {
//                     printf("DEBUG forward kernel block=%u dispatch complete\n", blockIdx.x);
//                 }
// #endif
            }
// #if FLASHMOE_DEBUG
//             if (!threadIdx.x && blockIdx.x < 3) {
//                 printf("DEBUG forward kernel block=%u calling processor::start\n", blockIdx.x);
//             }
// #endif
            processor::start(workspace, gateOutput, moeOutput, sb);
// #if FLASHMOE_DEBUG
//             if (!threadIdx.x && blockIdx.x < 3) {
//                 printf("DEBUG forward kernel block=%u processor::start complete\n", blockIdx.x);
//             }
// #endif
        }
        else {
// #if FLASHMOE_DEBUG
//             if (!threadIdx.x) {
//                 printf("DEBUG forward kernel block=%u calling os::start (last block)\n", blockIdx.x);
//             }
// #endif
            os::start<processors, d>(
                workspace,
                expertsUp,
                expertsDown,
                biasUp,
                biasDown,
                forwardSavedActivations,
                sb);
// #if FLASHMOE_DEBUG
//             if (!threadIdx.x) {
//                 printf("DEBUG forward kernel block=%u os::start complete\n", blockIdx.x);
//             }
// #endif
        }
    }

    template<
        unsigned S = ACC::S::value,
        unsigned int P = ACC::P::value,
        unsigned int H = ACC::H::value,
        unsigned int PX = ACC::PX::value
    >
    __global__ __maxnreg__(ACC::PeakHardware::registers::value) void backward(
        const void* __restrict__ gradOutputPtr,
        void* __restrict__ gradInputPtr,
        void* __restrict__ gradWeightsPtr,
        const __grid_constant__ uint16_t sb) {
        using GPUType = ACC::PeakHardware;
        constexpr auto blocks = GPUType::blocks::value;
        constexpr auto processors = GPUType::OS::processorBlocks::value;
        constexpr auto sharedSize = ACC::sharedSize::value;
        constexpr auto threads = GPUType::OS::threads::value;
        constexpr auto d = ACC::DTK::value;
        constexpr auto c = ACC::CM::value;
        using Element = ACC::Element;
        using ElementC = ACC::ElementC;

        const auto lE = bookkeeping.nLx;
        const auto* __restrict__ base = CONST_CAST_TO(Element, gradWeightsPtr);
        const auto* __restrict__ gP = base + S * H;
        const auto* __restrict__ ePu = gP + H * PX;
        const auto* __restrict__ ePd = ePu + lE * P * H;
        const auto* __restrict__ bU = ePd + lE * H * P;
        const auto* __restrict__ bd = bU + lE * P;
        auto* __restrict__ gIp = CAST_TO(Element, gradInputPtr);
        __shared__ __align__(16) cuda::std::byte workspace[sharedSize];
        clearState<threads, blocks, processors, c, ACC::JT::value, S * H>(gIp);
        // Grid barrier ensures all blocks complete clearState before any proceeds.
        // This prevents race between pSA clearing and dispatch's atomicIncrement.
        gridBarrier();

        const auto gradOutput = make_tensor(cute::make_gmem_ptr(CONST_CAST_TO(Element, gradOutputPtr)),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});
        const auto gradInput = make_tensor(cute::make_gmem_ptr(gIp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});

        const auto expertsUp = make_tensor(cute::make_gmem_ptr(CONST_CAST_TO(Element, ePu)),
            make_layout(make_shape(lE, cute::Shape<cute::Int<P>, cute::Int<H>>{}),
                cute::LayoutRight{}));
        const auto expertsDown = make_tensor(cute::make_gmem_ptr(CONST_CAST_TO(Element, ePd)),
            make_layout(make_shape(lE, cute::Shape<cute::Int<H>, cute::Int<P>>{}),
                cute::LayoutRight{}));
        const auto biasUp = make_tensor(cute::make_gmem_ptr(CONST_CAST_TO(Element, bU)),
            make_layout(cute::make_shape(lE, P),
                cute::Stride<cute::Int<P>, cute::_1>{}));
        const auto biasDown = make_tensor(cute::make_gmem_ptr(CONST_CAST_TO(Element, bd)),
            make_layout(make_shape(lE, cute::Int<H>{}),
                cute::Stride<cute::Int<H>, cute::_1>{}));

        if (!threadIdx.x && !blockIdx.x) {
            hiddenStatesPtr = CONST_CAST_TO(Element, savedActivationPtrs[0]);
            gateWeightsPtr = gP;
            printf("DEBUG backward kernel entry: sb=%u lE=%u base=%p gP=%p ePu=%p\n",
                   sb, lE, base, gP, ePu);
        }
        __syncthreads();

        // restore expert counts from forward pass (saved before they were cleared)
        {
            const auto* __restrict__ sEC = bookkeeping.sEC();
            auto* __restrict__ eC = bookkeeping.eC();
            const auto idx = threads * blockIdx.x + threadIdx.x;
            for (uint i = idx; i < ACC::E::value; i += blocks * threads) {
                eC[i] = sEC[i];
            }
        }
        gridBarrier();

        if constexpr (ACC::E::value > 1) {
            if (blockIdx.x + 1 < blocks) {
                if (blockIdx.x < ACC::DBZ::value) {
                    packet::dispatch<ACC::DBZ::value, d, ACC::SBZ::value>(gradOutput, workspace, sb);
                    if (!threadIdx.x && !blockIdx.x) {
                        printf("DEBUG backward kernel dispatcher done (sb=%u)\n", sb);
                    }
                }
                processor::start(workspace, gradOutput, gradInput, sb);
                if (!threadIdx.x && !blockIdx.x) {
                    printf("DEBUG backward kernel processor start invoked (sb=%u)\n", sb);
                }
            }
            else {
                os::start<processors, d>(
                    workspace,
                    expertsUp,
                    expertsDown,
                    biasUp,
                    biasDown,
                    savedActivationPtrs,
                    sb);
                if (!threadIdx.x && !blockIdx.x) {
                    printf("DEBUG backward kernel os start invoked (sb=%u)\n", sb);
                }
            }
        }
    }

    template<uint skip = 50, uint trials = 100>
    __host__ __forceinline__
    void forwardHostBench(const void* const& __restrict__ iP, void* __restrict__ const& oP, float& duration){
        #if FLASHMOE_NVTX
        flashmoeRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        exportSavedActivations(makeBiasSavedActivations(iP));
        cudaEvent_t start, stop;
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&start));
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&stop));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        #pragma unroll
        for (uint i = 0; i < skip; ++i) {
            forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        // Call forward pass
        FLASHMOE_CHECK_CUDA(cudaEventRecord(start, flashmoeStream));
        if constexpr (ACC::E::value > 1) {
            #pragma unroll
            for (uint i = 0; i < trials; ++i) {
                forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
                seqBit = sbs::next(seqBit);
            }
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, flashmoeStream>>>(iP, oP);
        }
        FLASHMOE_CHECK_CUDA(cudaEventRecord(stop, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaEventElapsedTime(&duration, start, stop));
        duration = duration / trials;
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(start));
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(stop));
    }

    __host__ __forceinline__
    void forwardHost(const void* const& __restrict__ iP, void* const& __restrict__ oP){
        #if FLASHMOE_NVTX
        flashmoeRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        exportSavedActivations(makeBiasSavedActivations(iP));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        // Call forward pass
        if constexpr (ACC::E::value > 1) {
            forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, flashmoeStream>>>(iP, oP);
        }
    }

    template<uint skip = 50, uint trials = 100>
    __host__ __forceinline__
    void forwardHostTrainingBench(
        const void* const& __restrict__ iP,
        void* __restrict__ const& oP,
        const cuda::std::array<const cuda::std::byte*, GEMMs>& savedActivations,
        float& duration
    ) {
        #if FLASHMOE_NVTX
        flashmoeRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        exportSavedActivations(savedActivations);
        cudaEvent_t start, stop;
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&start));
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&stop));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        #pragma unroll
        for (uint i = 0; i < skip; ++i) {
            forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        // Call forward pass
        FLASHMOE_CHECK_CUDA(cudaEventRecord(start, flashmoeStream));
        if constexpr (ACC::E::value > 1) {
            #pragma unroll
            for (uint i = 0; i < trials; ++i) {
                forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
                seqBit = sbs::next(seqBit);
            }
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, flashmoeStream>>>(iP, oP);
        }
        FLASHMOE_CHECK_CUDA(cudaEventRecord(stop, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaEventElapsedTime(&duration, start, stop));
        duration = duration / trials;
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(start));
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(stop));
    }

    __host__ __forceinline__
    void forwardHostTraining(
        const void* const& __restrict__ iP,
        void* const& __restrict__ oP,
        const cuda::std::array<const cuda::std::byte*, GEMMs>& savedActivations
    ) {
        #if FLASHMOE_NVTX
        flashmoeRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        exportSavedActivations(savedActivations);
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        // Call forward pass
        if constexpr (ACC::E::value > 1) {
            forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, flashmoeStream>>>(iP, oP);
        }
    }

    __host__ __forceinline__
    void backwardHost(
        const void* const& __restrict__ gradOutputPtr,
        const void* const& __restrict__ savedActivations,
        void* __restrict__ const& gradInputPtr,
        void* __restrict__ const& gradWeightsPtr,
        float& duration
    ) {
        #if FLASHMOE_NVTX
        flashmoeRange backwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        if (savedActivations != nullptr) {
            const auto* saved = reinterpret_cast<const cuda::std::array<const cuda::std::byte*, GEMMs>*>(savedActivations);
            exportSavedActivations(*saved);
        }
        cudaEvent_t start, stop;
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&start));
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&stop));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        uint16_t gradSeqBit = static_cast<uint16_t>(SignalConstants::gradSequenceStart);
        printf("DEBUG backwardHost: launching kernel seqBit=%u saved=%p\n",
               gradSeqBit, savedActivations);
        FLASHMOE_CHECK_CUDA(cudaEventRecord(start, flashmoeStream));
        backward<<<blocks, threads, 0, flashmoeStream>>>(
            gradOutputPtr, gradInputPtr, gradWeightsPtr, gradSeqBit);
        gradSeqBit = sbs::nextGrad(gradSeqBit);
        FLASHMOE_CHECK_CUDA(cudaEventRecord(stop, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoeStream));
        printf("DEBUG backwardHost: kernel completed seqBit=%u\n", gradSeqBit);
        FLASHMOE_CHECK_CUDA(cudaEventElapsedTime(&duration, start, stop));
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(start));
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(stop));
    }
}
#endif //FLASHMOE_MOE_CUH

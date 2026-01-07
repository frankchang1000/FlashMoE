/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */
//
// Created by osayamen on 7/13/24.
//

#ifndef FLASHMOE_COMPUTE_CUH
#define FLASHMOE_COMPUTE_CUH

#include <cutlass/array.h>
#include <cute/tensor.hpp>
#include <cuda/std/limits>
#include <nvshmem.h>

#include "gemm.cuh"
#include "../../types.cuh"

namespace flashmoe::moe {
    extern __device__ const ACC::Element* hiddenStatesPtr;
    extern __device__ const ACC::Element* gateWeightsPtr;
    extern __device__ const ACC::Element* gradOutputBasePtr;
    extern __device__ ACC::Element* gradInputBasePtr;
}

namespace flashmoe::processor{
    enum class ReleaseType {
        stable,
        experimental
    };
    template<
        CombineMode c = CombineMode::single,
        unsigned int gM = BLOCK_M,
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::H::value,
        class ScaleWeights,
        typename Element,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        unsigned int sharedSize = ACC::PeakHardware::sharedMemory::value + ACC::PeakHardware::spare::value,
        unsigned int wS = WARP_SIZE
    >
    requires(TensorValueType<Element> &&
            elems % wS == 0 && // guarantees warp convergence
            flashmoe::isMatrix<ScaleWeights> &&
            cuda::std::is_same_v<typename ScaleWeights::value_type, Element>)
    __device__ __forceinline__
    void combine(cuda::std::byte* __restrict__ const& workspace,
            const TPS* __restrict__ const& tokenIndices,
            const Element* __restrict__ const& inputs,
            Element* __restrict__ const& moeOutput,
            ScaleWeights const& scale,
            const unsigned int& tileIdx,
            const uint16_t& tileSize,
            const uint16_t& expertIdx) {
        using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
        constexpr BlockTiler tiler{};
        constexpr auto bM = cute::get<0>(tiler);
        constexpr auto bN = cute::get<1>(tiler);
        cutlass::Array<Element, bN> registers{};
        constexpr auto mTe = cutlass::NumericConverter<Element, mp_t>{};
        constexpr auto eTm = cutlass::NumericConverter<mp_t, Element>{};
        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            cute::Layout<cute::Shape<cute::Int<gM>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mC = make_tensor(cute::make_gmem_ptr(moeOutput),
            cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});

        // We assert the below prior to this point
        static_assert(gM % bM == 0);
        constexpr auto tilesM = gM / bM;
        constexpr auto tilesN = N / bN;

        const auto tileCoord = idx2crd(tileIdx,
            cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord));
        const auto gA = cute::local_tile(mA, tiler, ctaCoord);

        const auto tileCoordOut = idx2crd(tileIdx,
            cute::Shape<cute::_1, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto gC = cute::local_tile(mC,
            cute::Shape<cute::Int<M>, cute::Int<bN>>{},
                cute::make_coord(cute::get<0>(tileCoordOut),
                    cute::get<1>(tileCoordOut)));
        static_assert(bN % elems == 0);
        constexpr auto trips = bN / elems;
        // ensures we have enough shared memory
        static_assert(sizeof(TPS) * bM <= sharedSize);
        static_assert(bM % elems == 0);
        // slice and dice token indices
        constexpr auto phases = bM / elems;
        const auto phaseIdx = threadIdx.x / elems;
        static_assert(elems % wS == 0);
        constexpr auto wE = elems / wS;
        auto* __restrict__ sTPS = CAST_TO(TPS, workspace);
        static_assert(bM == threads);
        if (threadIdx.x < tileSize) {
            sTPS[threadIdx.x] = tokenIndices[threadIdx.x];
        } else {
            sTPS[threadIdx.x] = TPS{0, mp_t(0)};
        }
        __syncthreads();
#if FLASHMOE_DEBUG
        // Debug: verify TPS data and input heap data
        if (!threadIdx.x && !blockIdx.x) {
            // Sample input data from heap (first 4 elements of first row)
            const auto inputSample0 = static_cast<float>(inputs[0]);
            const auto inputSample1 = static_cast<float>(inputs[1]);
            const auto inputSample2 = static_cast<float>(inputs[2]);
            const auto inputSample3 = static_cast<float>(inputs[3]);
            // Sample TPS probabilities
            const auto prob0 = static_cast<float>(sTPS[0].probability);
            const auto prob1 = static_cast<float>(sTPS[1].probability);
            printf("DEBUG combine-input rank=%d tile=%u expert=%u tileSize=%u "
                   "heapData[0..3]=[%.2f,%.2f,%.2f,%.2f] "
                   "TPS[0]={tok=%u,prob=%.4f} TPS[1]={tok=%u,prob=%.4f}\n",
                   nvshmem_my_pe(), tileIdx, expertIdx, tileSize,
                   inputSample0, inputSample1, inputSample2, inputSample3,
                   sTPS[0].tokenIdx, prob0, sTPS[1].tokenIdx, prob1);
        }
#endif
        // Eagerly prefetch inputs to registers
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            // global -> registers
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto rIdx = phaseIdx + j * phases;
                const auto cIdx =  threadIdx.x % elems + i * elems;
                registers[j + i * elems] = gA(rIdx, cIdx);
            }
        }
        if constexpr (c == CombineMode::multithreaded) {
            TPS wT[wE];
            Element rS[wE];
            #pragma unroll
            for (uint i = 0; i < wE; ++i) {
                const auto tid = threadIdx.x % wS;
                wT[i] = sTPS[phaseIdx + (tid + i * wS) * phases];
                rS[i] = scale(wT[i].tokenIdx, expertIdx);
            }
            __syncwarp();
            cutlass::Array<uint, elems> tIds{};
            using CDxT = typename ToCDx<Element>::T;
            constexpr auto cTCx = cutlass::NumericConverter<CDxT, Element>{};
            const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(CDxT, registers.data())),
                cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>, cute::Stride<cute::Int<bN>, cute::_1>>{});
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = wT[j / wS];
                const auto* __restrict__ mP = CONST_CAST_TO(ull_t, &msg);
                const auto rM = __shfl_sync(0xffffffff, *mP, j % wS);
                const auto [tokenIdx, probability] = *CONST_CAST_TO(TPS, &rM);
                tIds[j] = tokenIdx;
                // apply division operation
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    registers[i + j * elems] = mTe(__fdividef(eTm(registers[i + j * elems]), probability));
                }
            }
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = rS[j / wS];
                const auto* __restrict__ mP = CONST_CAST_TO(CDxT, &msg);
                const auto rM = __shfl_sync(0xffffffff, *mP, j % wS);
                // apply scale
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    rC(j + i * elems) = cTCx(*CONST_CAST_TO(Element, &rM) * registers[j + i * elems]);
                }
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                // do atomic addition
                if (tileSize < gM) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        if (phaseIdx + j * phases < tileSize) {
                            atomicAdd(CAST_TO(CDxT, &gC(tIds[j], cIdx)), rC(j + i * elems));
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        atomicAdd(CAST_TO(CDxT, &gC(tIds[j], cIdx)), rC(j + i * elems));
                    }
                }
            }
        }
        else {
            uint wT[wE];
            const auto tid = threadIdx.x % wS;
            #pragma unroll
            for (uint i = 0; i < wE; ++i) {
                wT[i] = sTPS[phaseIdx + (tid + i * wS) * phases].tokenIdx;
            }
            // vector copy from registers to global directly and call it a day
            cutlass::AlignedArray<uint, elems> tIds{};
            __syncwarp();
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = wT[j / wS];
                tIds[j] = __shfl_sync(0xffffffff, msg, j % wS);
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                const auto cIdx = threadIdx.x % elems + i * elems;
                if (tileSize < gM) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        // predicated writes
                        if (phaseIdx + j * phases < tileSize) {
                            gC(tIds[j], cIdx) = registers[j + i * elems];
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        gC(tIds[j], cIdx) = registers[j + i * elems];
                    }
                }
            }
        }
    }

    template<
        CombineMode c = CombineMode::single,
        unsigned int gM = BLOCK_M,
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::H::value,
        class ScaleWeights,
        typename Element,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        unsigned int sharedSize = ACC::PeakHardware::sharedMemory::value + ACC::PeakHardware::spare::value,
        unsigned int wS = WARP_SIZE
    >
    requires(TensorValueType<Element> &&
            elems % wS == 0 &&
            flashmoe::isMatrix<ScaleWeights> &&
            cuda::std::is_same_v<typename ScaleWeights::value_type, Element>)
    __device__ __forceinline__
    void splitGradients(cuda::std::byte* __restrict__ const& workspace,
            const TPS* __restrict__ const& tokenIndices,
            const Element* __restrict__ const& gradOutput,
            Element* __restrict__ const& expertGradients,
            ScaleWeights const& scale,
            const unsigned int& tileIdx,
            const uint16_t& tileSize,
            const uint16_t& expertIdx) {
        using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
        constexpr BlockTiler tiler{};
        constexpr auto bM = cute::get<0>(tiler);
        constexpr auto bN = cute::get<1>(tiler);
        cutlass::Array<Element, bN> registers{};
        constexpr auto mTe = cutlass::NumericConverter<Element, mp_t>{};
        constexpr auto eTm = cutlass::NumericConverter<mp_t, Element>{};
        const auto mGradOut = make_tensor(cute::make_gmem_ptr(gradOutput),
            cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mExpertGrad = make_tensor(cute::make_gmem_ptr(expertGradients),
            cute::Layout<cute::Shape<cute::Int<gM>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});

        static_assert(gM % bM == 0);
        constexpr auto tilesM = gM / bM;
        constexpr auto tilesN = N / bN;

        const auto tileCoordOut = idx2crd(tileIdx,
            cute::Shape<cute::_1, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto gGradOut = cute::local_tile(mGradOut,
            cute::Shape<cute::Int<M>, cute::Int<bN>>{},
                cute::make_coord(cute::get<0>(tileCoordOut),
                    cute::get<1>(tileCoordOut)));

        const auto tileCoord = idx2crd(tileIdx,
            cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord));
        const auto gExpertGrad = cute::local_tile(mExpertGrad, tiler, ctaCoord);

        static_assert(bN % elems == 0);
        constexpr auto trips = bN / elems;
        static_assert(sizeof(TPS) * bM <= sharedSize);
        static_assert(bM % elems == 0);
        constexpr auto phases = bM / elems;
        const auto phaseIdx = threadIdx.x / elems;
        static_assert(elems % wS == 0);
        constexpr auto wE = elems / wS;
        auto* __restrict__ sTPS = CAST_TO(TPS, workspace);
        static_assert(bM == threads);
        if (threadIdx.x < tileSize) {
            sTPS[threadIdx.x] = tokenIndices[threadIdx.x];
        } else {
            sTPS[threadIdx.x] = TPS{0, mp_t(0)};
        }
        __syncthreads();

#if FLASHMOE_DEBUG
        // Debug: Find max tokenIdx and check for OOB (tokenIdx >= S)
        {
            __shared__ unsigned int sMaxTokenIdx;
            __shared__ unsigned int sOOBCount;
            if (!threadIdx.x) {
                sMaxTokenIdx = 0;
                sOOBCount = 0;
            }
            __syncthreads();

            if (threadIdx.x < tileSize) {
                const auto myTokenIdx = sTPS[threadIdx.x].tokenIdx;
                atomicMax(&sMaxTokenIdx, myTokenIdx);
                if (myTokenIdx >= M) {
                    atomicAdd(&sOOBCount, 1U);
                }
            }
            __syncthreads();

            if (!threadIdx.x) {
                if (sMaxTokenIdx >= M || sOOBCount > 0) {
                    printf("ERROR splitGradients OOB: rank=%d block=%u tileIdx=%u tileSize=%u expert=%u "
                           "maxTokenIdx=%u >= S=%u oobCount=%u\n",
                           nvshmem_my_pe(), blockIdx.x, tileIdx, tileSize, expertIdx,
                           sMaxTokenIdx, M, sOOBCount);
                    printf("ERROR splitGradients PTRS: aData(tokenIndices)=%p gradOutput=%p expertGradients=%p\n",
                           tokenIndices, gradOutput, expertGradients);
                    printf("ERROR splitGradients tokenIdx[0..7]=[%u,%u,%u,%u,%u,%u,%u,%u]\n",
                           sTPS[0].tokenIdx, sTPS[1].tokenIdx, sTPS[2].tokenIdx, sTPS[3].tokenIdx,
                           sTPS[4].tokenIdx, sTPS[5].tokenIdx, sTPS[6].tokenIdx, sTPS[7].tokenIdx);
                }
            }
            __syncthreads();
        }
//         // DEBUG: Print sample input/output values to verify correctness
//         // Print for rank 0, first few blocks to limit spam
//         if (!threadIdx.x && nvshmem_my_pe() == 0 && blockIdx.x < 4) {
//             // Sample TPS entries (tokenIdx, probability)
//             printf("DEBUG splitGradients VALUES: rank=%d block=%u tileIdx=%u tileSize=%u expert=%u\n",
//                    nvshmem_my_pe(), blockIdx.x, tileIdx, tileSize, expertIdx);
//             for (uint i = 0; i < min(4u, static_cast<uint>(tileSize)); ++i) {
//                 printf("  TPS[%u]: tokenIdx=%u prob=%.6f\n", i, sTPS[i].tokenIdx, static_cast<float>(sTPS[i].probability));
//             }
//             // Sample gradOutput values for first token
//             if (tileSize > 0) {
//                 const auto tok0 = sTPS[0].tokenIdx;
//                 printf("  gradOutput[tok=%u, 0..3]: %.6e %.6e %.6e %.6e\n",
//                        tok0,
//                        static_cast<float>(gradOutput[tok0 * N + 0]),
//                        static_cast<float>(gradOutput[tok0 * N + 1]),
//                        static_cast<float>(gradOutput[tok0 * N + 2]),
//                        static_cast<float>(gradOutput[tok0 * N + 3]));
//                 // Expected output: (gradOutput[tok, col] / probability) * scale
//                 const auto prob0 = static_cast<float>(sTPS[0].probability);
//                 const auto scale0 = static_cast<float>(scale(tok0, expertIdx));
//                 const auto expectedVal = static_cast<float>(gradOutput[tok0 * N + 0]) / prob0 * scale0;
//                 printf("  expected expertGrad[0,0] = %.6e / %.6e * %.6e = %.6e\n",
//                        static_cast<float>(gradOutput[tok0 * N + 0]),
//                        prob0,
//                        scale0,
//                        expectedVal);
//             }
//         }
//         __syncthreads();
#endif

        if constexpr (c == CombineMode::multithreaded) {
            TPS wT[wE];
            Element rS[wE];
            #pragma unroll
            for (uint i = 0; i < wE; ++i) {
                const auto tid = threadIdx.x % wS;
                wT[i] = sTPS[phaseIdx + (tid + i * wS) * phases];
                rS[i] = scale(wT[i].tokenIdx, expertIdx);
            }
            __syncwarp();
            cutlass::Array<uint, elems> tIds{};
            using CDxT = typename ToCDx<Element>::T;
            constexpr auto cTCx = cutlass::NumericConverter<CDxT, Element>{};
            const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(CDxT, registers.data())),
                cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>, cute::Stride<cute::Int<bN>, cute::_1>>{});
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = wT[j / wS];
                const auto* __restrict__ mP = CONST_CAST_TO(ull_t, &msg);
                const auto rM = __shfl_sync(0xffffffff, *mP, j % wS);
                const auto [tokenIdx, probability] = *CONST_CAST_TO(TPS, &rM);
                tIds[j] = tokenIdx;
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto cIdx = threadIdx.x % elems + i * elems;
                    registers[j + i * elems] = gGradOut(tIds[j], cIdx);
                }
            }
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = wT[j / wS];
                const auto* __restrict__ mP = CONST_CAST_TO(ull_t, &msg);
                const auto rM = __shfl_sync(0xffffffff, *mP, j % wS);
                const auto [tokenIdx, probability] = *CONST_CAST_TO(TPS, &rM);
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    registers[i + j * elems] = mTe(__fdividef(eTm(registers[i + j * elems]), probability));
                }
            }
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = rS[j / wS];
                const auto* __restrict__ mP = CONST_CAST_TO(CDxT, &msg);
                const auto rM = __shfl_sync(0xffffffff, *mP, j % wS);
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    rC(j + i * elems) = cTCx(*CONST_CAST_TO(Element, &rM) * registers[j + i * elems]);
                }
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                if (tileSize < gM) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        if (phaseIdx + j * phases < tileSize) {
                            *CAST_TO(CDxT, &gExpertGrad(phaseIdx + j * phases, cIdx)) = rC(j + i * elems);
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        *CAST_TO(CDxT, &gExpertGrad(phaseIdx + j * phases, cIdx)) = rC(j + i * elems);
                    }
                }
            }
        }
        else {
            uint wT[wE];
            const auto tid = threadIdx.x % wS;
            #pragma unroll
            for (uint i = 0; i < wE; ++i) {
                wT[i] = sTPS[phaseIdx + (tid + i * wS) * phases].tokenIdx;
            }
            cutlass::AlignedArray<uint, elems> tIds{};
            __syncwarp();
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = wT[j / wS];
                tIds[j] = __shfl_sync(0xffffffff, msg, j % wS);
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto cIdx = threadIdx.x % elems + i * elems;
                    registers[j + i * elems] = gGradOut(tIds[j], cIdx);
                }
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                const auto cIdx = threadIdx.x % elems + i * elems;
                if (tileSize < gM) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        if (phaseIdx + j * phases < tileSize) {
                            gExpertGrad(phaseIdx + j * phases, cIdx) = registers[j + i * elems];
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        gExpertGrad(phaseIdx + j * phases, cIdx) = registers[j + i * elems];
                    }
                }
            }
        }

// #if FLASHMOE_DEBUG
//         // DEBUG: Print sample output values after computation
//         __syncthreads();
//         if (!threadIdx.x && nvshmem_my_pe() == 0 && blockIdx.x < 4) {
//             // tileIdx maps to 2D coord: rowTile = tileIdx / tilesN, colTile = tileIdx % tilesN
//             // Each tile writes to [rowTile * bM, colTile * bN] offset
//             const uint rowTile = tileIdx / tilesN;
//             const uint colTile = tileIdx % tilesN;
//             const uint outRowOffset = rowTile * bM;
//             const uint outColOffset = colTile * bN;
//             printf("  OUTPUT expertGrad[row=%u, col=%u..%u]: %.6e %.6e %.6e %.6e (block=%u tile=%u rowT=%u colT=%u)\n",
//                    outRowOffset, outColOffset, outColOffset + 3,
//                    static_cast<float>(expertGradients[outRowOffset * N + outColOffset + 0]),
//                    static_cast<float>(expertGradients[outRowOffset * N + outColOffset + 1]),
//                    static_cast<float>(expertGradients[outRowOffset * N + outColOffset + 2]),
//                    static_cast<float>(expertGradients[outRowOffset * N + outColOffset + 3]),
//                    blockIdx.x, tileIdx, rowTile, colTile);
//             // Compare with expected - use column from THIS tile's range
//             if (tileSize > 0) {
//                 const auto tok0 = sTPS[0].tokenIdx;
//                 const auto prob0 = static_cast<float>(sTPS[0].probability);
//                 const auto scale0 = static_cast<float>(scale(tok0, expertIdx));
//                 // gradOutput is indexed by tokenIdx (full row), so we read the col that this tile writes to
//                 // expected = (gradOutput / prob) * scale
//                 printf("  VERIFY: gradOut[%u,%u]=%.6e / prob=%.6e * scale=%.6e -> expected=%.6e, actual=%.6e\n",
//                        tok0, outColOffset,
//                        static_cast<float>(gradOutput[tok0 * N + outColOffset]),
//                        prob0,
//                        scale0,
//                        static_cast<float>(gradOutput[tok0 * N + outColOffset]) / prob0 * scale0,
//                        static_cast<float>(expertGradients[outRowOffset * N + outColOffset]));
//             }
//         }
// #endif
    }

    // fused GEMM, epilogue and data transfer, with static M, N and K
    template<
        typename BlockGEMM,
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::H::value,
        unsigned int K = ACC::P::value,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __forceinline__ __device__
    void sfGET(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& tileIdx) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) % elems == 0);
        cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<K>>,
            cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
        cute::Layout<cute::Shape<cute::Int<N>, cute::Int<K>>,
            cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(output),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
            cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mD = make_tensor(cute::make_gmem_ptr(bias),
            cute::Layout<cute::Shape<cute::_1, cute::Int<N>>,
                cute::Stride<cute::_0, cute::_1>>{});

        // M is padded, such that the below is correct
        constexpr auto tilesM = M / bM;
        // We assert the below prior to this point
        constexpr auto tilesN = N / bN;
        constexpr auto tilesK = K / bK;

        const auto tileCoord = idx2crd(tileIdx, cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
            cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});

        const auto k_tile_iter = cute::make_coord_iterator(tilesK);
        using Element = typename BlockGEMM::MatrixDType;

        // prefetch bias from global memory
        static_assert(ACC::sharedSize::value >= ACC::GSM::value + sizeof(Element) * bN);
        const auto biasCoord = idx2crd(tileIdx, cute::Shape<cute::_1, cute::Int<tilesN>>{},
            cute::Stride<cute::Int<bN>, cute::_1>{});
        const auto gD = cute::local_tile(mD,
            cute::Shape<cute::_1, cute::Int<bN>>{}, cute::get<1>(biasCoord));
        static_assert(threads % bN == 0);
        if (threadIdx.x < bN) {
            using LT =  cuda::std::conditional_t<sizeof(Element) == 2, uint16_t, uint32_t>;
            CAST_TO(LT, workspace)[threadIdx.x] = __ldg(CONST_CAST_TO(LT, &gD(threadIdx.x)));
        }
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, workspace));
        /// There is a block-wide barrier at the end of the above ^

        // Epilogue
        using ElementC = typename decltype(accumulator)::value_type;
        typename BlockGEMM::MMA tiledMMA{};
        constexpr auto gCStoreOp = cutlass::NumericConverter<Element, ElementC>{};
        constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, Element>{};
        // Assume elementwise operator
        typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / elems;
        // copy single bias value
        ElementC rB[trips];
        #pragma unroll
        for (uint i = 0 ; i < trips; ++i) {
            rB[i] = gDLoadOp(workspace[threadIdx.x % elems + i * elems]);
        }
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, workspace + bN)), sCLay);
        const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(Element, accumulator.data())),
            cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>,
                cute::Stride<cute::Int<bN>, cute::_1>>{});
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);

        // Reorder to striped arrangement
        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = accumulator(j + i * elems);
            }
            __syncthreads();
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                accumulator(j + i * elems) = sC(rIdx + j, cIdx);
            }
        }

        // apply epilogue
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                rC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), rB[i]));
            }
        }

        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        // Coalesced copy from registers to global memory
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = rC(j + i * elems);
            }
        }
    }

    // fused GEMM, epilogue and data transfer, with dynamic M and static N and K
    // Optional preActivationOutput parameter for saving z values (pre-activation) during forward pass
    template<
        typename BlockGEMM,
        unsigned int N,
        unsigned int K,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __forceinline__ __device__
    void fGET(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& M,
        const unsigned int& tileIdx,
        typename BlockGEMM::MatrixDType* __restrict__ const& preActivationOutput = nullptr) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) % elems == 0);
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        static_assert(cute::size(accumulator) == bN);
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::Stride<cute::Int<K>, cute::_1>{}));
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
            cute::Layout<cute::Shape<cute::Int<N>, cute::Int<K>>,
                cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(output),
            make_layout(cute::make_shape(M, N), cute::Stride<cute::Int<N>, cute::_1>{}));

        // M is padded, such that the below is correct
        const auto tilesM = M / bM;
        // We assert the below prior to this point
        constexpr auto tilesN = N / bN;
        constexpr auto tilesK = K / bK;

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN),
            cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto k_tile_iter = cute::make_coord_iterator(tilesK);
        using Element = typename BlockGEMM::MatrixDType;
        constexpr bool isGradientMode = isActivationDerivative<typename BlockGEMM::FusedEpilogue>::value;

        if constexpr (!isGradientMode) {
            static_assert(ACC::sharedSize::value >= ACC::GSM::value + sizeof(Element) * bN);
            const auto mD = make_tensor(cute::make_gmem_ptr(bias),
                cute::Layout<cute::Shape<cute::_1, cute::Int<N>>,
                    cute::Stride<cute::_0, cute::_1>>{});
            const auto biasCoord = idx2crd(tileIdx, cute::Shape<cute::_1, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<bN>, cute::_1>{});
            const auto gD = cute::local_tile(mD,
                cute::Shape<cute::_1, cute::Int<bN>>{}, cute::get<1>(biasCoord));
            static_assert(threads % bN == 0);
            if (threadIdx.x < bN) {
                using LT =  cuda::std::conditional_t<sizeof(Element) == 2, uint16_t, uint32_t>;
                CAST_TO(LT, workspace)[threadIdx.x] = __ldg(CONST_CAST_TO(LT, &gD(threadIdx.x)));
            }
        }
        using ElementC = typename decltype(accumulator)::value_type;
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, workspace + bN));
        /// There is a block-wide barrier at the end of the above ^

        // Epilogue
        typename BlockGEMM::MMA tiledMMA{};
        constexpr auto gCStoreOp = cutlass::NumericConverter<Element, ElementC>{};
        constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, Element>{};
        // Assume elementwise operator
        typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / elems;
 
        ElementC rB[trips];
        if constexpr (!isGradientMode) {
            #pragma unroll
            for (uint i = 0 ; i < trips; ++i) {
                rB[i] = gDLoadOp(workspace[threadIdx.x % elems + i * elems]);
            }
        }

        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, workspace + bN)), sCLay);
        const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(Element, accumulator.data())),
            cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>,
                cute::Stride<cute::Int<bN>, cute::_1>>{});
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);

        // Reorder to striped arrangement
        // TODO(Jonathan): Do the below reordering without shared memory. It would make my day (decade actually) to solve this.
        // A lot of performance badness happens down there.
        // I haven't sat down to think about a solution yet.
        // First idea that comes to mind is some form of warp register shuffling.
        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = accumulator(j + i * elems);
            }
            __syncthreads();
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                accumulator(j + i * elems) = sC(rIdx + j, cIdx);
            }
        }

        // After reordering, compute row/column indices for this thread
        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;

        // create tensor view of saved activation (2D layout) for gradients
        // and load values directly from global memory per-element
        if constexpr (isGradientMode) {
            // bias parameter contains the saved activation tensor with shape (M, N)
            const auto mSavedAct = make_tensor(cute::make_gmem_ptr(bias),
                make_layout(cute::make_shape(M, N), cute::Stride<cute::Int<N>, cute::_1>{}));
            const auto gSavedAct = cute::local_tile(mSavedAct, typename BlockGEMM::BlockTiler{}, ctaCoord,
                cute::Step<cute::_1,cute::_1, cute::X>{});

#if FLASHMOE_DEBUG
            if (!threadIdx.x && blockIdx.x < 2) {
                const auto tileRow = cute::get<0>(tileCoord);
                const auto tileCol = cute::get<1>(tileCoord);
                printf("DEBUG fGET gradient mode: block=%u M=%u N=%u K=%u tileIdx=%u tileCoord=(%u,%u) bias=%p tilesM=%u tilesN=%u\n",
                       blockIdx.x, M, N, K, tileIdx, tileRow, tileCol, bias, tilesM, tilesN);
            }
#endif
            // apply epilogue with per-element saved activation
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    const auto savedVal = gDLoadOp(gSavedAct(rIdx + j, cIdx + i * elems));
                    rC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), savedVal));
                }
            }
        } else {
            // apply epilogue with broadcast bias
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    rC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), rB[i]));
                }
            }
        }

        // Save pre-activation values (z = accumulator + bias) if buffer is provided
        // This is needed for gradient computation in backward pass
        // Only applies to forward mode (gradient mode reads saved values, doesn't save new ones)
        if constexpr (!isGradientMode) {
            if (preActivationOutput != nullptr) {
                // Create tensor view for z-buffer with same layout as output
                const auto mZ = make_tensor(cute::make_gmem_ptr(preActivationOutput),
                    make_layout(cute::make_shape(M, N), cute::Stride<cute::Int<N>, cute::_1>{}));
                const auto gZ = cute::local_tile(mZ, typename BlockGEMM::BlockTiler{}, ctaCoord,
                    cute::Step<cute::_1,cute::_1, cute::X>{});

                // Compute and save pre-activation values: z = accumulator + bias
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    #pragma unroll
                    for (unsigned int j = 0; j < elems; ++j) {
                        // Pre-activation = accumulator (after reordering) + bias
                        // Note: accumulator here has already been reordered to striped arrangement
                        const auto preActivation = accumulator(j + i * elems) + rB[i];
                        gZ(rIdx + j, cIdx + i * elems) = gCStoreOp(preActivation);
                    }
                }
            }
        }

        // Coalesced copy from registers to global memory
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = rC(j + i * elems);
            }
        }
    }
    template<
        typename BlockGEMM,
        unsigned int N,
        unsigned int K,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __forceinline__ __device__
    void fGETGrad(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& savedActivation,
        const unsigned int& M,
        const unsigned int& tileIdx) {
        using Element = typename BlockGEMM::MatrixDType;
        using MMA = typename BlockGEMM::MMA;
        using ElementC = ACC::ElementC;

        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});

        constexpr auto gCStoreOp = cutlass::NumericConverter<Element, ElementC>{};
        constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, Element>{};
        typename BlockGEMM::FusedEpilogue epilogueOp{};

        // Row-major input [M, K]
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::Stride<cute::Int<K>, cute::_1>{}));
        // Saved activation [M, K] - same shape as input
        const auto mSavedAct = make_tensor(cute::make_gmem_ptr(savedActivation),
            make_layout(cute::make_shape(M, K), cute::Stride<cute::Int<K>, cute::_1>{}));
        // Interpret weights as [K, N] row-major for A @ B semantics
        // Physical storage for this op is [K, N] row-major, so mB(k, n) = weights[k*N + n]
        // This allows us to transpose during load: sB(n, k) = mB(k, n)
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
            cute::Layout<cute::Shape<cute::Int<K>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});
        // Row-major output [M, N]
        const auto mC = make_tensor(cute::make_gmem_ptr(output),
            make_layout(cute::make_shape(M, N), cute::Stride<cute::Int<N>, cute::_1>{}));

        const auto tilesM = M / bM;
        constexpr auto tilesN = N / bN;
        constexpr auto tilesK = K / bK;

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN),
            cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);

#if FLASHMOE_DEBUG
        if (!threadIdx.x && blockIdx.x < 2) {
            const auto tileRow = cute::get<0>(tileCoord);
            const auto tileCol = cute::get<1>(tileCoord);
            printf("DEBUG fGETGrad: block=%u M=%u N=%u K=%u tileIdx=%u tileCoord=(%u,%u) savedAct=%p tilesM=%u tilesN=%u\n",
                   blockIdx.x, M, N, K, tileIdx, tileRow, tileCol, savedActivation, tilesM, tilesN);
        }
#endif

        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::_1, cute::X>{});

        // Shared memory layout for A and B tiles - use same layouts as BlockGEMM
        using sLayAAtom = typename BlockGEMM::Parameters::sLayA;
        using sLayBAtom = typename BlockGEMM::Parameters::sLayB;
        using sLayA = decltype(cute::tile_to_shape(sLayAAtom{}, cute::Shape<cute::Int<bM>, cute::Int<bK>>{}));
        using sLayB = decltype(cute::tile_to_shape(sLayBAtom{}, cute::Shape<cute::Int<bN>, cute::Int<bK>>{}));

        constexpr auto smemAElements = cute::size(sLayA{});
        constexpr auto smemBElements = cute::size(sLayB{});
        static_assert(ACC::sharedSize::value >= sizeof(Element) * (smemAElements + smemBElements));

        auto* __restrict__ sARaw = CAST_TO(Element, workspace);
        auto* __restrict__ sBRaw = sARaw + smemAElements;

        const auto sA = cute::make_tensor(cute::make_smem_ptr(sARaw), sLayA{});
        const auto sB = cute::make_tensor(cute::make_smem_ptr(sBRaw), sLayB{});

        MMA tiledMMA{};
        auto thr_mma = tiledMMA.get_slice(threadIdx.x);

        auto tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K)
        auto tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K)

        auto tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA, MMA_M, MMA_K)
        auto tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA, MMA_N, MMA_K)

        using sCLayC = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>,
            cute::Stride<cute::Int<bN>, cute::_1>>;
        const auto sC_for_partition = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, workspace)), sCLayC{});
        auto accumulator = thr_mma.partition_fragment_C(sC_for_partition);  // (MMA, MMA_M, MMA_N)
        cute::clear(accumulator);
        static_assert(cute::size(accumulator) % elems == 0);

        // K-tile loop: load tiles to smem, apply act' to A, then gemm via tensor cores
        for (uint kTile = 0; kTile < tilesK; ++kTile) {
            const auto mTileRow = cute::get<0>(tileCoord) * bM;
            const auto nTileCol = cute::get<1>(tileCoord) * bN;

            constexpr auto elementsPerThreadA = (bM * bK + threads - 1) / threads;
            #pragma unroll
            for (uint e = 0; e < elementsPerThreadA; ++e) {
                const auto flatIdx = threadIdx.x + e * threads;
                if (flatIdx < bM * bK) {
                    const auto localRow = flatIdx / bK;
                    const auto localCol = flatIdx % bK;
                    const auto globalRow = mTileRow + localRow;
                    const auto globalCol = kTile * bK + localCol;

                    if (globalRow < M && globalCol < K) {
                        const auto inVal = gDLoadOp(mA(globalRow, globalCol));
                        const auto savedVal = gDLoadOp(mSavedAct(globalRow, globalCol));
                        // grad_z = grad * act'(z) - apply activation derivative before GEMM
                        sA(localRow, localCol) = gCStoreOp(epilogueOp(inVal, savedVal));
                    } else {
                        sA(localRow, localCol) = Element(0);
                    }
                }
            }

            // Load B tile (weights) with transpose: sB(n,k) = mB(k,n)
            // sLayB matches BlockGEMM layout, so index sB(n,k) directly
            // mB is [K, N] row-major, so mB(k,n) reads from weights[k*N + n]
            // This achieves A @ B semantics (MMA computes A @ sB^T = A @ B)
            constexpr auto elementsPerThreadB = (bN * bK + threads - 1) / threads;
            #pragma unroll
            for (uint e = 0; e < elementsPerThreadB; ++e) {
                const auto flatIdx = threadIdx.x + e * threads;
                if (flatIdx < bN * bK) {
                    // [bN, bK] row-major indexing for sB
                    const auto localN = flatIdx / bK;
                    const auto localK = flatIdx % bK;
                    const auto globalN = nTileCol + localN;
                    const auto globalK = kTile * bK + localK;

                    if (globalK < K && globalN < N) {
                        // Transpose: sB(n,k) = mB(k,n) = weights[k*N + n]
                        sB(localN, localK) = mB(globalK, globalN);
                    } else {
                        sB(localN, localK) = Element(0);
                    }
                }
            }
            __syncthreads();

            constexpr int K_BLOCK_MAX = cute::size<2>(tCrA);
            #pragma unroll
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                #pragma unroll
                for (int m = 0; m < cute::size<1>(tCrA); ++m) {
                    #pragma unroll
                    for (int i = 0; i < cute::size<0>(tCrA); ++i) {
                        tCrA(i, m, k_block) = tCsA(i, m, k_block);
                    }
                }
                #pragma unroll
                for (int n = 0; n < cute::size<1>(tCrB); ++n) {
                    #pragma unroll
                    for (int i = 0; i < cute::size<0>(tCrB); ++i) {
                        tCrB(i, n, k_block) = tCsB(i, n, k_block);
                    }
                }
                cute::gemm(thr_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), accumulator);
            }
            __syncthreads();
        }

        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, workspace)), sCLay);
        const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(Element, accumulator.data())),
            cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>,
                cute::Stride<cute::Int<bN>, cute::_1>>{});
        const auto tCsC = thr_mma.partition_C(sC);

        constexpr auto trips = cute::size(accumulator) / elems;

        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = accumulator(j + i * elems);
            }
            __syncthreads();
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                accumulator(j + i * elems) = sC(rIdx + j, cIdx);
            }
        }

        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                rC(j + i * elems) = gCStoreOp(accumulator(j + i * elems));
            }
        }
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = rC(j + i * elems);
            }
        }
    }

    struct __align__(16) ProcessorArgs{
        // sensible sentinel values
        unsigned int* __restrict__ sQ = nullptr;
        TQSignal* __restrict__ pDB = nullptr;
        unsigned int* __restrict__ tQH = nullptr;
        Task* __restrict__ tQ = nullptr;
        Task* __restrict__ ptQ = nullptr;
        unsigned int* __restrict__ tQS = nullptr;

        ProcessorArgs() = default;
        __device__
        ProcessorArgs(unsigned int* const& _sQ,
            TQSignal* const& _pDB,
            unsigned int* const& _tQH,
            Task* const& _tQ,
            Task* const& _ptQ,
            unsigned int* const& _tQS) :
        sQ(_sQ), pDB(_pDB), tQH(_tQH), tQ(_tQ), ptQ(_ptQ), tQS(_tQS) {}
    };

    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TNx::value
    >
    __device__ __forceinline__
    void notifyNext(uint* __restrict__ const& workspace, const Task& rCurrentTask, const ProcessorArgs& pA) {
        static_assert(sizeof(Task) == 128);
        constexpr auto eS = sizeof(Task) / sizeof(uint);
        static_assert(eS == WARP_SIZE);
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        constexpr auto sharedSize = ACC::sharedSize::value;
        static_assert(sharedSize % sizeof(Task) == 0);
        static_assert(sharedSize / sizeof(Task) >= threads);
        constexpr auto capacity = threads;
        constexpr auto trips = tasks / capacity;
        static_assert(threads % eS == 0);
        static_assert(capacity % threads == 0);
        constexpr auto elems = capacity * eS / threads;
        constexpr unsigned int preIndex = 0;

        const auto offset = ACC::TNx::value * rCurrentTask.batchIdx;
        constexpr auto ptQSlotSize = ACC::TN::value + ACC::TNx::value;  // Forward pass: TN + TNx
        auto* __restrict__ tQ = CAST_TO(uint, pA.ptQ + (rCurrentTask.syncIdx * ptQSlotSize));
        const auto cIdx = threadIdx.x % eS;
        // prep memory-view tensors
        const auto sTQ = make_tensor(cute::make_smem_ptr(workspace),
            cute::Layout<cute::Shape<cute::Int<threads>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        const auto gTQ = make_tensor(cute::make_gmem_ptr(tQ),
            cute::Layout<cute::Shape<cute::Int<tasks>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        // copy from registers to shared memory using swizzle
        if constexpr (trips) {
            const auto rIdx = threadIdx.x / eS * eS;
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                // each thread does a copy from registers to shared memory
                const auto taskIdx = threadIdx.x + i * capacity;
                const auto tileIdx = offset + taskIdx;
                const auto nextTask = Task {
                    TaskType::postGEMM,
                    rCurrentTask.cData[preIndex],
                    rCurrentTask.bData,
                    rCurrentTask.cData,
                    rCurrentTask.dData,
                    rCurrentTask.rcData,
                    rCurrentTask.flags + offset + (p == PeerConnectivity::p2p ? taskIdx : 0),
                    rCurrentTask.syncIdx,
                    tileIdx,
                    rCurrentTask.M,
                    rCurrentTask.tileSize,
                    rCurrentTask.peerIdx,
                    rCurrentTask.batchIdx,
                    rCurrentTask.isPeerRemote,
                    rCurrentTask.expertIdx,
                };
                // Directive to the compiler to reinterpret the Task structure as a stream of 4-byte blocks
                const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
                #pragma unroll
                for (uint j = 0; j < eS; ++j) {
                    // temporal shift of indices to eliminate bank conflicts
                    const auto swizzleIdx = (j + threadIdx.x) % eS;
                    sTQ(threadIdx.x, swizzleIdx) = uT[j];
                }
                __syncthreads();
                // now copy from shared memory to global memory
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    gTQ(rIdx + (j + i * capacity), cIdx) = sTQ(rIdx + j, (cIdx + j) % eS);
                }
            }
            // before reusing shared memory below
            __syncthreads();
        }
        if constexpr (constexpr auto residue = tasks - trips * capacity; residue) {
            if (threadIdx.x < residue) {
                const auto taskIdx = threadIdx.x + trips * capacity;
                const auto tileIdx = offset + taskIdx;
                const auto nextTask = Task {
                    TaskType::postGEMM,
                    rCurrentTask.cData[preIndex],
                    rCurrentTask.bData,
                    rCurrentTask.cData,
                    rCurrentTask.dData,
                    rCurrentTask.rcData,
                    rCurrentTask.flags + offset + (p == PeerConnectivity::p2p ? taskIdx : 0),
                    rCurrentTask.syncIdx,
                    tileIdx,
                    rCurrentTask.M,
                    rCurrentTask.tileSize,
                    rCurrentTask.peerIdx,
                    rCurrentTask.batchIdx,
                    rCurrentTask.isPeerRemote,
                    rCurrentTask.expertIdx,
                };
                // Directive to the compiler to reinterpret the Task structure as a stream of 4-byte blocks
                const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
                #pragma unroll
                for (uint j = 0; j < eS; ++j) {
                    // temporal shift of indices to eliminate bank conflicts
                    const auto swizzleIdx = (j + threadIdx.x) % eS;
                    sTQ(threadIdx.x, swizzleIdx) = uT[j];
                }
            }
            __syncthreads();
            constexpr auto stride = threads / eS;
            const auto pIdx = threadIdx.x / eS;
            constexpr auto length = residue / stride;
            // now copy from shared memory to global memory by multiplexing each row across available warps
            #pragma unroll
            for (uint j = 0; j < length; ++j) {
                const auto idx = j * stride + pIdx;
                gTQ(idx + trips * capacity, cIdx) = sTQ(idx, (cIdx + idx) % eS);
            }
            if constexpr (constexpr auto rS = residue % stride; rS) {
                if (pIdx < rS) {
                    const auto idx = length * stride + pIdx;
                    gTQ(idx + trips * capacity, cIdx) = sTQ(idx, (cIdx + idx) % eS);
                }
            }
        }

        __syncthreads();
        if (!threadIdx.x) {
            __threadfence();
            // notify scheduler
            atomicAdd(pA.tQH + rCurrentTask.syncIdx, tasks);
        }
    }

    template<TaskType TaskT,
        PeerConnectivity p,
        unsigned int tasks = ACC::TNx::value,
        unsigned int gradIndex = 0,  // Which cData index to use as new task's aData
        bool preserveAData = false,
        unsigned int slotOffset = 0, // Offset within ptQ slot (e.g., TN for gradGateGEMM to write after gradPostGEMM)
        bool useSecondaryTQH = false> // Use secondary tQH domain (syncIdx + gtQCl) for gradPreGEMM tasks
    __device__ __forceinline__
    void notifyGradientImpl(uint* __restrict__ const& workspace, const Task& rCurrentTask, const ProcessorArgs& pA) {
        static_assert(sizeof(Task) == 128);
        constexpr auto eS = sizeof(Task) / sizeof(uint);
        static_assert(eS == WARP_SIZE);
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        constexpr auto sharedSize = ACC::sharedSize::value;
        static_assert(sharedSize % sizeof(Task) == 0);
        static_assert(sharedSize / sizeof(Task) >= threads);
        constexpr auto capacity = threads;
        constexpr auto trips = tasks / capacity;
        static_assert(threads % eS == 0);
        static_assert(capacity % threads == 0);
        constexpr auto elems = capacity * eS / threads;

        constexpr auto ptQSlotSize = ACC::TN::value + 2 * ACC::TNx::value;
        const auto ptQOffset = rCurrentTask.syncIdx * ptQSlotSize + slotOffset;
        auto* __restrict__ tQ = CAST_TO(uint, pA.ptQ + ptQOffset);

        constexpr auto flagsStride = ACC::TNx::value;
        constexpr bool applyFlagsOffset = (TaskT == TaskType::gradPreGEMM);
        const auto flagsOffset = applyFlagsOffset ? (flagsStride * rCurrentTask.batchIdx) : 0U;

        // Bounds checks for ptQ secondary queue overflow
        {
            const auto gtQCl = bookkeeping.gtQCl;
            const auto ptQCapacity = gtQCl * ptQSlotSize;
            // Check syncIdx is within gtQCl (gradient task queue capacity limit)
            if (!threadIdx.x && rCurrentTask.syncIdx >= gtQCl) {
                printf("ERROR notifyGradientImpl syncIdx OOB: rank=%u block=%u TaskT=%u "
                       "syncIdx=%u >= gtQCl=%u expert=%u batchIdx=%u\n",
                       nvshmem_my_pe(), blockIdx.x, static_cast<unsigned>(TaskT),
                       rCurrentTask.syncIdx, gtQCl, rCurrentTask.expertIdx, rCurrentTask.batchIdx);
            }
            // Check ptQOffset won't exceed ptQ capacity
            if (!threadIdx.x && ptQOffset >= ptQCapacity) {
                printf("ERROR notifyGradientImpl ptQOffset OOB: rank=%u block=%u TaskT=%u "
                       "ptQOffset=%u >= ptQCapacity=%u syncIdx=%u gtQCl=%u\n",
                       nvshmem_my_pe(), blockIdx.x, static_cast<unsigned>(TaskT),
                       ptQOffset, ptQCapacity, rCurrentTask.syncIdx, gtQCl);
            }
            // Check ptQOffset + tasks won't exceed ptQ capacity (last task index)
            if (!threadIdx.x && (ptQOffset + tasks) > ptQCapacity) {
                printf("ERROR notifyGradientImpl ptQ end OOB: rank=%u block=%u TaskT=%u "
                       "ptQOffset=%u tasks=%u end=%u > ptQCapacity=%u syncIdx=%u\n",
                       nvshmem_my_pe(), blockIdx.x, static_cast<unsigned>(TaskT),
                       ptQOffset, tasks, ptQOffset + tasks, ptQCapacity, rCurrentTask.syncIdx);
            }
            // Check ptQ base pointer is not null
            if (!threadIdx.x && pA.ptQ == nullptr) {
                printf("ERROR notifyGradientImpl ptQ NULL: rank=%u block=%u TaskT=%u syncIdx=%u\n",
                       nvshmem_my_pe(), blockIdx.x, static_cast<unsigned>(TaskT), rCurrentTask.syncIdx);
            }
        }

#if FLASHMOE_DEBUG
        if (!threadIdx.x && blockIdx.x < 2) {
            printf("DEBUG notifyGradientImpl ENTER: block=%u TaskT=%u tasks=%u batchIdx=%u syncIdx=%u "
                   "ptQ=%p ptQ_offset=%u write_addr=%p\n",
                   blockIdx.x, static_cast<unsigned>(TaskT), tasks, rCurrentTask.batchIdx,
                   rCurrentTask.syncIdx, pA.ptQ, ptQOffset, pA.ptQ + ptQOffset);
        }
#endif
        const auto cIdx = threadIdx.x % eS;
        const auto sTQ = make_tensor(cute::make_smem_ptr(workspace),
            cute::Layout<cute::Shape<cute::Int<threads>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        const auto gTQ = make_tensor(cute::make_gmem_ptr(tQ),
            cute::Layout<cute::Shape<cute::Int<tasks>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        if constexpr (trips) {
            const auto rIdx = threadIdx.x / eS * eS;
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                const auto taskIdx = threadIdx.x + i * capacity;
                // tileIdx is column tile only - cData pointers are already row-offset
                // so fGETGrad/fGET should see tileIdx in [0, tilesN) to avoid double row offset
                const auto tileIdx = taskIdx;
                const auto nextAData = preserveAData ? rCurrentTask.aData : rCurrentTask.cData[gradIndex];
                const auto nextTask = Task {
                    TaskT,
                    nextAData,
                    rCurrentTask.bData,
                    rCurrentTask.cData,
                    rCurrentTask.dData,
                    rCurrentTask.rcData,
                    rCurrentTask.flags + flagsOffset + (applyFlagsOffset && p == PeerConnectivity::p2p ? taskIdx : 0),
                    rCurrentTask.syncIdx,
                    tileIdx,
                    rCurrentTask.M,
                    rCurrentTask.tileSize,
                    rCurrentTask.peerIdx,
                    rCurrentTask.batchIdx,
                    rCurrentTask.isPeerRemote,
                    rCurrentTask.expertIdx,
                };
                const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
                #pragma unroll
                for (uint j = 0; j < eS; ++j) {
                    const auto swizzleIdx = (j + threadIdx.x) % eS;
                    sTQ(threadIdx.x, swizzleIdx) = uT[j];
                }
                __syncthreads();
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    gTQ(rIdx + (j + i * capacity), cIdx) = sTQ(rIdx + j, (cIdx + j) % eS);
                }
            }
            __syncthreads();
        }
        if constexpr (constexpr auto residue = tasks - trips * capacity; residue) {
            if (threadIdx.x < residue) {
                const auto taskIdx = threadIdx.x + trips * capacity;
                // tileIdx is column tile only - cData pointers are already row-offset
                const auto tileIdx = taskIdx;
                const auto nextAData = preserveAData ? rCurrentTask.aData : rCurrentTask.cData[gradIndex];
                const auto nextTask = Task {
                    TaskT,
                    nextAData,
                    rCurrentTask.bData,
                    rCurrentTask.cData,
                    rCurrentTask.dData,
                    rCurrentTask.rcData,
                    rCurrentTask.flags + flagsOffset + (applyFlagsOffset && p == PeerConnectivity::p2p ? taskIdx : 0),
                    rCurrentTask.syncIdx,
                    tileIdx,
                    rCurrentTask.M,
                    rCurrentTask.tileSize,
                    rCurrentTask.peerIdx,
                    rCurrentTask.batchIdx,
                    rCurrentTask.isPeerRemote,
                    rCurrentTask.expertIdx,
                };
                const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
                #pragma unroll
                for (uint j = 0; j < eS; ++j) {
                    const auto swizzleIdx = (j + threadIdx.x) % eS;
                    sTQ(threadIdx.x, swizzleIdx) = uT[j];
                }
            }
            __syncthreads();
            constexpr auto stride = threads / eS;
            const auto pIdx = threadIdx.x / eS;
            constexpr auto length = residue / stride;
            #pragma unroll
            for (uint j = 0; j < length; ++j) {
                const auto idx = j * stride + pIdx;
                gTQ(idx + trips * capacity, cIdx) = sTQ(idx, (cIdx + idx) % eS);
            }
            if constexpr (constexpr auto rS = residue % stride; rS) {
                if (pIdx < rS) {
                    const auto idx = length * stride + pIdx;
                    gTQ(idx + trips * capacity, cIdx) = sTQ(idx, (cIdx + idx) % eS);
                }
            }
        }

        __syncthreads();
        if (!threadIdx.x) {
            __threadfence();
            // For gradPreGEMM (useSecondaryTQH=true), write to tQH[syncIdx + gtQCl] so scheduler
            // can poll it separately from the already-visited primary domain (gradPostGEMM/gradGateGEMM)
            const auto tQHSyncIdx = useSecondaryTQH ? (rCurrentTask.syncIdx + bookkeeping.gtQCl) : rCurrentTask.syncIdx;
#if FLASHMOE_DEBUG
            const auto tQH_before = atomicAdd(pA.tQH + tQHSyncIdx, 0U);
#endif
            atomicAdd(pA.tQH + tQHSyncIdx, tasks);
// #if FLASHMOE_DEBUG
//             if (blockIdx.x < 2) {
//                 printf("DEBUG notifyGradientImpl DONE: block=%u TaskT=%u syncIdx=%u tQHSyncIdx=%u tQH_before=%u adding=%u tQH_after=%u\n",
//                        blockIdx.x, static_cast<unsigned>(TaskT), rCurrentTask.syncIdx, tQHSyncIdx,
//                        tQH_before, tasks, tQH_before + tasks);
//             }
// #endif
        }
    }

    // gradPostGEMM output is [M, P], needs TN = ceil(P/BLOCK_N) column tiles
    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TN::value
    >
    __device__ __forceinline__
    void notifyGradient(uint* __restrict__ const& workspace, const Task& rCurrentTask, const ProcessorArgs& pA) {
        notifyGradientImpl<TaskType::gradPostGEMM, p, tasks>(workspace, rCurrentTask, pA);
    }

    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TNx::value
    >
    __device__ __forceinline__
    void notifyGateGradient(uint* __restrict__ const& workspace, const Task& rCurrentTask, const ProcessorArgs& pA) {
        notifyGradientImpl<TaskType::gradGateGEMM, p, tasks, 0, true, ACC::TN::value>(workspace, rCurrentTask, pA);
    }

    // gradPostGEMM output (grad_intermediate in cData[1]) becomes gradPreGEMM input
    // gradPreGEMM output is [M, H], needs TNx = ceil(H/BLOCK_N) column tiles
    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TNx::value
    >
    __device__ __forceinline__
    void notifyGradPreGEMM(uint* __restrict__ const& workspace, const Task& rCurrentTask, const ProcessorArgs& pA) {
        // Use gradIndex=1 because gradPostGEMM output is in cData[1]
        // Use useSecondaryTQH=true to write to tQH[syncIdx + gtQCl], giving gradPreGEMM tasks
        // a fresh scheduler polling domain (since the primary domain is already marked "visited")
        notifyGradientImpl<TaskType::gradPreGEMM, p, tasks, 1, false, ACC::TN::value + ACC::TNx::value, true>(workspace, rCurrentTask, pA);
    }

    __device__ __forceinline__
    void enqueueTask(const Task& task, const ProcessorArgs& pA) {
        static_assert(sizeof(Task) == 128);
        constexpr auto eS = sizeof(Task) / sizeof(uint);

// #if FLASHMOE_DEBUG
//         if (!blockIdx.x) {
//             printf("DEBUG enqueueTask: thread=%u block=%u - single-thread direct copy\n",
//                    threadIdx.x, blockIdx.x);
//         }
// #endif
        // Direct copy from task to global memory (thread 0 only)
        const auto taskIdx = atomicAdd(pA.tQH + task.syncIdx, 1U);
        if (taskIdx >= bookkeeping.sT) {
            printf("ERROR tQ->ptQ OVERFLOW (enqueueTask): rank=%u taskIdx=%u sT=%u syncIdx=%u taskType=%u\n",
                   nvshmem_my_pe(), taskIdx, bookkeeping.sT, task.syncIdx, static_cast<unsigned>(task.taskType));
            return;
        }
        auto* __restrict__ tQ = CAST_TO(uint, pA.tQ + taskIdx);
        const auto* __restrict__ uT = CONST_CAST_TO(uint, &task);
        #pragma unroll
        for (uint i = 0; i < eS; ++i) {
            tQ[i] = uT[i];
        }
        __threadfence();
    }

    template<
        unsigned int K = ACC::P::value,
        unsigned int N = ACC::H::value,
        typename Element,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    requires(TensorValueType<Element>)
    __device__ __forceinline__
    void computeWeightGradients(cuda::std::byte* __restrict__ const& workspace,
            const Element* __restrict__ const& activations,
            const Element* __restrict__ const& gradients,
            Element* __restrict__ const& weightGradBuffer,
            const uint16_t& tileSize,
            const uint16_t& expertIdx) {
        constexpr auto bM = BLOCK_M;

        const auto mAct = make_tensor(cute::make_gmem_ptr(activations),
            cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<K>>,
                cute::Stride<cute::Int<K>, cute::_1>>{});
        const auto mGrad = make_tensor(cute::make_gmem_ptr(gradients),
            cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mWGrad = make_tensor(cute::make_gmem_ptr(weightGradBuffer),
            cute::Layout<cute::Shape<cute::Int<K>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});

        constexpr uint totalKN = K * N;
        constexpr uint elemsPerThread = (totalKN + threads - 1) / threads;

#if FLASHMOE_DEBUG
        if (!threadIdx.x && blockIdx.x < 2) {
            printf("DEBUG computeWeightGradients: rank=%d block=%u expert=%u tileSize=%u K=%u N=%u bM=%u threads=%u act=%p grad=%p buffer=%p act[0]=%.6f grad[0]=%.6f buffer[0]=%.6f totalKN=%u elemsPerThread=%u\n",
                   nvshmem_my_pe(), blockIdx.x, expertIdx, tileSize, K, N, bM, threads,
                   activations, gradients, weightGradBuffer,
                   static_cast<float>(activations[0]), static_cast<float>(gradients[0]),
                   static_cast<float>(weightGradBuffer[0]),
                   totalKN, elemsPerThread);
        }
#endif

        using NativeElement = typename ToCDx<Element>::T;
        constexpr auto convert = cutlass::NumericConverter<NativeElement, Element>{};

        const uint startIdx = threadIdx.x * elemsPerThread;
        const uint endIdx = min(startIdx + elemsPerThread, totalKN);

#if FLASHMOE_DEBUG
        Element debugSum{0};
        bool isDebugThread = (threadIdx.x == 0 && blockIdx.x == 0);
#endif

        for (uint idx = startIdx; idx < endIdx; ++idx) {
            const uint k = idx / N;
            const uint n = idx % N;

            Element sum{0};
            for (uint m = 0; m < tileSize; ++m) {
                sum += mAct(m, k) * mGrad(m, n);
            }

            auto* nativePtr = CAST_TO(NativeElement, &mWGrad(k, n));
            atomicAdd(nativePtr, convert(sum));

#if FLASHMOE_DEBUG
            if (isDebugThread && idx == startIdx) {
                debugSum = sum;
            }
#endif
        }

        __syncthreads();

#if FLASHMOE_DEBUG
        if (!threadIdx.x && blockIdx.x < 2) {
            printf("DEBUG computeWeightGradients EXIT: rank=%d block=%u expert=%u buffer[0]=%.6f firstSum=%.6f startIdx=%u endIdx=%u\n",
                   nvshmem_my_pe(), blockIdx.x, expertIdx, static_cast<float>(weightGradBuffer[0]),
                   static_cast<float>(debugSum), startIdx, endIdx);
        }
        if (threadIdx.x == threads - 1 && blockIdx.x < 2) {
            printf("DEBUG computeWeightGradients LAST_THREAD: rank=%d block=%u thread=%u startIdx=%u endIdx=%u elemsComputed=%u\n",
                   nvshmem_my_pe(), blockIdx.x, threadIdx.x, startIdx, endIdx, endIdx - startIdx);
        }
#endif
    }

    template<
        typename ScaleWeights,
        typename Output
    >
    __device__ __forceinline__
    void start(cuda::std::byte* const& workspace,
        ScaleWeights const& sW, Output const& moeOutput,
        const uint16_t& _seqBit){
        using Element = ACC::Element;
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(flagsType)
            && alignof(SignalPayload<PacketStage::last>) == alignof(flagsType));

        static_assert(sizeof(Task) == 128);
        __shared__ Task currentTask;
        __shared__ uint globalInterrupt;
        __shared__ uint enqueue;

        // Register allocations
        const auto rSeqBit = _seqBit;
        Task rCurrentTask{};
        TQSignal tqs{0U, 0U};
        const auto pA = ProcessorArgs{
            bookkeeping.sQ() + blockIdx.x,
            bookkeeping.pDB() + blockIdx.x,
            bookkeeping.tQH(),
            bookkeeping.tQ(),
            bookkeeping.ptQ(),
            bookkeeping.tSA()
        };

        if (!threadIdx.x) {
            atomicExch_block(&globalInterrupt, 0U);
            atomicExch_block(&enqueue, 0U);
        }
        using PreGEMM = BlockMM<ACC::ActivationOp, Element>;
        using PostGEMM = BlockMM<ACC::ActivationOpX, Element>;
        using GradPreGEMM = BlockMM<flashmoe::ActivationDerivative<ACC::ElementC, ACC::ActivationOp>, Element>;
        using GradPostGEMM = BlockMM<flashmoe::ActivationDerivative<ACC::ElementC, ACC::ActivationOpX>, Element>;
        constexpr uint H = ACC::H::value;
        constexpr uint P = ACC::P::value;
        constexpr auto tN = ACC::TN::value;
        constexpr auto tNx = ACC::TNx::value;
        __syncthreads();
#if FLASHMOE_DEBUG
        static __shared__ uint processorIterCount;
        static __shared__ uint processorStuckCount;
        if (!threadIdx.x) {
            processorIterCount = 0U;
            processorStuckCount = 0U;
        }
        __syncthreads();
#endif
        while (!tqs.interrupt) {
            if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                if (!threadIdx.x) {
#if FLASHMOE_DEBUG
                    processorIterCount++;
                    processorStuckCount++;
                    // if (blockIdx.x < 3 && processorIterCount <= 10) {
                    //     printf("DEBUG processor-loop rank=%d block=%u sb=%u iter=%u waiting for task\n",
                    //            nvshmem_my_pe(),
                    //            blockIdx.x,
                    //            _seqBit,
                    //            processorIterCount);
                    // }
                    if (processorStuckCount >= 100000 && processorStuckCount % 100000 == 0) {
                        printf("DEBUG processor-stuck rank=%d block=%u sb=%u stuck_iter=%u total_iter=%u\n",
                               nvshmem_my_pe(),
                               blockIdx.x,
                               _seqBit,
                               processorStuckCount,
                               processorIterCount);
                    }
#endif
                    auto* __restrict__ tQSignal = pA.pDB;
                    // Grabs next task
                    awaitNotification(tQSignal, &tqs, tqs.signal);
#if FLASHMOE_DEBUG
                    // if (blockIdx.x < 3 && processorIterCount <= 10) {
                    //     printf("DEBUG processor-loop rank=%d block=%u sb=%u iter=%u got notification signal=%u interrupt=%u\n",
                    //            nvshmem_my_pe(),
                    //            blockIdx.x,
                    //            _seqBit,
                    //            processorIterCount,
                    //            tqs.signal,
                    //            tqs.interrupt);
                    // }
                    if (processorStuckCount >= 100000) {
                        printf("DEBUG processor-unstuck rank=%d block=%u sb=%u was_stuck=%u signal=%u interrupt=%u\n",
                               nvshmem_my_pe(),
                               blockIdx.x,
                               _seqBit,
                               processorStuckCount,
                               tqs.signal,
                               tqs.interrupt);
                        processorStuckCount = 0U;
                    }
#endif
                    __threadfence();
                    // Eagerly indicate readiness for the next task as the above fence allows us to do so correctly
                    globalInterrupt = tqs.interrupt;
                    atomicExch(pA.sQ, ready);
                }
                // The below is necessary as it guarantees memory ordering
                __syncwarp();
                auto* __restrict__ tqsP = CAST_TO(ull_t, &tqs);
                *tqsP = __shfl_sync(0xffffffff, *tqsP, 0);
                const auto* __restrict__ gtQ = pA.tQ + tqs.decodeSig();
                if (!tqs.interrupt) {
                    // coalesced copy from global to shared memory
                    CAST_TO(uint, &currentTask)[threadIdx.x] = __ldg(CONST_CAST_TO(uint, gtQ) + threadIdx.x);
                }
            }
            __syncthreads();
            tqs.interrupt = globalInterrupt;
            // if we received an interrupt, there is nothing to do next
            if (!tqs.interrupt) {
                // shared -> registers
                rCurrentTask = currentTask;
// #if FLASHMOE_DEBUG
//                 const bool isGradTask = rCurrentTask.taskType == TaskType::gradPreGEMM ||
//                     rCurrentTask.taskType == TaskType::gradPostGEMM ||
//                     rCurrentTask.taskType == TaskType::gradCombine ||
//                     rCurrentTask.taskType == TaskType::gradWeights ||
//                     rCurrentTask.taskType == TaskType::gradGateCombine ||
//                     rCurrentTask.taskType == TaskType::gradGateGEMM;
//                 const bool isCombineTask = rCurrentTask.taskType == TaskType::combine;
//                 if (!threadIdx.x && !blockIdx.x && (isGradTask || isCombineTask)) {
//                     printf("DEBUG processor task sb=%u type=%u tile=%u tileSize=%u peer=%u remote=%u expert=%u M=%u\n",
//                            rSeqBit,
//                            static_cast<unsigned>(rCurrentTask.taskType),
//                            rCurrentTask.tileIdx,
//                            rCurrentTask.tileSize,
//                            rCurrentTask.peerIdx,
//                            static_cast<unsigned>(rCurrentTask.isPeerRemote),
//                            rCurrentTask.expertIdx,
//                            rCurrentTask.M);
//                 }
// #endif
#if FLASHMOE_DEBUG
                // Log gradient task types being processed and check for corruption
                if (!threadIdx.x && blockIdx.x < 3) {
                    constexpr const char* taskNames[] = {
                        "preGEMM", "postGEMM", "combine", "gradPreGEMM", "gradPostGEMM",
                        "gradWeights", "gradCombine", "gradGateCombine", "gradGateGEMM", "gradInputCombine"
                    };
                    const auto taskTypeVal = static_cast<unsigned>(rCurrentTask.taskType);
                    if (taskTypeVal > 9) {
                        printf("ERROR processor CORRUPT_TASK: block=%u type=%u (INVALID>9!) syncIdx=%u tileIdx=%u batchIdx=%u aData=%p\n",
                               blockIdx.x, taskTypeVal, rCurrentTask.syncIdx, rCurrentTask.tileIdx,
                               rCurrentTask.batchIdx, rCurrentTask.aData);
                    } else if (taskTypeVal >= 3) { // gradient tasks start at 3 (gradPreGEMM)
                        printf("DEBUG processor TASK: rank=%d block=%u type=%s tile=%u syncIdx=%u batchIdx=%u\n",
                               nvshmem_my_pe(), blockIdx.x, taskNames[taskTypeVal], rCurrentTask.tileIdx, rCurrentTask.syncIdx,
                               rCurrentTask.batchIdx);
                    }
                }
#endif
                switch (rCurrentTask.taskType) {
                    case TaskType::preGEMM: {
                        constexpr unsigned int preIndex = 0;
                        // Compute z1 save pointer for training mode
                        // z1 has the same layout as xM (intermediate buffer)
                        Element* z1Save = nullptr;
                        if constexpr (ACC::JT::value == JobType::training) {
                            auto* xMBase = CAST_TO(Element, bookkeeping.xM());
                            auto* cDataPtr = CAST_TO(Element, rCurrentTask.cData[preIndex]);
                            const auto xMOffset = cDataPtr - xMBase;
                            z1Save = bookkeeping.z1() + xMOffset;
                        }
                        fGET<PreGEMM, ACC::P::value, ACC::H::value>(
                            CAST_TO(typename PreGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename PreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PreGEMM::MatrixBType, rCurrentTask.bData[preIndex]),
                            CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.cData[preIndex]),
                            CONST_CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.dData[preIndex]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx,
                            z1Save);
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
                            enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) + 1 == tN;
                        }
                        __syncthreads();
                        if (enqueue) {
                            if (!rCurrentTask.isPeerRemote) {
                                notifyNext<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyNext<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                        }
                    }
                    break;
                    case TaskType::postGEMM: {
                        constexpr unsigned int postIndex = 1;
#if FLASHMOE_DEBUG
                        // Verify tileIdx consistency - currentTask and rCurrentTask should match
                        if (!threadIdx.x && !blockIdx.x) {
                            if (currentTask.tileIdx != rCurrentTask.tileIdx) {
                                printf("BUG DETECTED: postGEMM tileIdx mismatch! currentTask=%u rCurrentTask=%u\n",
                                       currentTask.tileIdx, rCurrentTask.tileIdx);
                            }
                        }
#endif
                        // Compute z2 save pointer for training mode
                        // z2 layout: [world * nLx * pEC, H]
                        // aData points to xM (intermediate buffer from preGEMM)
                        Element* z2Save = nullptr;
                        if constexpr (ACC::JT::value == JobType::training) {
                            auto* xMBase = CAST_TO(Element, bookkeeping.xM());
                            auto* aDataPtr = CONST_CAST_TO(Element, rCurrentTask.aData);
                            // Compute row index in xM (which has P columns)
                            const auto xMOffset = aDataPtr - xMBase;
                            const auto rowIdx = xMOffset / P;
                            // z2 has H columns, so offset is rowIdx * H
                            z2Save = bookkeeping.z2() + rowIdx * H;
                        }
                        fGET<PostGEMM, ACC::H::value, ACC::P::value>(
                            CAST_TO(typename PostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixBType, rCurrentTask.bData[postIndex]),
                            CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.cData[postIndex]),
                            CONST_CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.dData[postIndex]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx,
                            z2Save);  // was prev currentTask.tileIdx. look into this
                        __syncthreads();
                        if (!threadIdx.x) {
                            // Pack payload into single signal word of 8 bytes
                            const auto flagSignal = SignalPayload<PacketStage::last>{
                                rCurrentTask.batchIdx,
                                rCurrentTask.tileSize,
                                rSeqBit,
                            };
                            if (rCurrentTask.isPeerRemote) {
                                // Remote; check if we need to do the transfer
                                __threadfence();
                                const auto syncCount = atomicIncrement(pA.tQS + rCurrentTask.syncIdx) + 1;
                                if (syncCount == tN + tNx) {
#if FLASHMOE_DEBUG
                                    printf("DEBUG postGEMM NVSHMEM transfer rank=%d block=%u sb=%u syncIdx=%u syncCount=%u tN=%u tNx=%u tileSize=%u peer=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, rSeqBit,
                                           rCurrentTask.syncIdx, syncCount, tN, tNx,
                                           rCurrentTask.tileSize, rCurrentTask.peerIdx);
#endif
                                    nvshmem_putmem_signal_nbi(rCurrentTask.rcData,
                                        rCurrentTask.cData[postIndex],
                                        // Batched remote network transfer to avoid overwhelming the NIC
                                        rCurrentTask.tileSize * H * sizeof(Element),
                                        rCurrentTask.flags,
                                        *CONST_CAST_TO(flagsType, &flagSignal), NVSHMEM_SIGNAL_SET,
                                        rCurrentTask.peerIdx);
                                }
                            }
                            else {
                                // individual tile, no batching here
                                // Already did the network transfer,
                                // so set signal only
                                __threadfence_system();
                                atomicExch_system(CAST_TO(ull_t, rCurrentTask.flags),
                                    *CONST_CAST_TO(ull_t, &flagSignal));
                            }
                        }
                    }
                    break;
                    case TaskType::combine: {
                        constexpr unsigned int combineIndex = 0;
                        combine<ACC::CM::value>(
                            workspace,
                            CONST_CAST_TO(TPS, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.bData[combineIndex]),
                            moeOutput.data().get(),
                            sW,
                            rCurrentTask.tileIdx,
                            rCurrentTask.tileSize,
                            rCurrentTask.expertIdx);
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && !blockIdx.x) {
                            // Sample first 4 token indices from this combine task
                            const auto* tps = CONST_CAST_TO(TPS, rCurrentTask.aData);
                            printf("DEBUG combine rank=%d sb=%u tile=%u tileSize=%u expert=%u peer=%u remote=%u tokenIdx[0..3]=[%u,%u,%u,%u]\n",
                                   nvshmem_my_pe(),
                                   rSeqBit,
                                   rCurrentTask.tileIdx,
                                   rCurrentTask.tileSize,
                                   rCurrentTask.expertIdx,
                                   rCurrentTask.peerIdx,
                                   static_cast<unsigned>(rCurrentTask.isPeerRemote),
                                   tps[0].tokenIdx, tps[1].tokenIdx, tps[2].tokenIdx, tps[3].tokenIdx);
                        }
#endif
                    }
                    break;
                    case TaskType::gradCombine: {
                        constexpr unsigned int gradIndex = 0;
                        constexpr auto H = ACC::H::value;
                        constexpr auto S = ACC::S::value;

                        if (!threadIdx.x) {
                            if (rCurrentTask.aData == nullptr) {
                                printf("ERROR gradCombine: rank=%u block=%u aData (tokenIndices) is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (rCurrentTask.cData[gradIndex] == nullptr) {
                                printf("ERROR gradCombine: rank=%u block=%u cData[0] (dest buffer) is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (rCurrentTask.cData[1] == nullptr) {
                                printf("ERROR gradCombine: rank=%u block=%u cData[1] (xM) is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (flashmoe::moe::gradOutputBasePtr == nullptr) {
                                printf("ERROR gradCombine: rank=%u block=%u gradOutputBasePtr is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (rCurrentTask.tileSize == 0) {
                                printf("ERROR gradCombine: rank=%u block=%u tileSize is ZERO!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            // ERROR: Validate expertIdx bounds
                            if (rCurrentTask.expertIdx >= ACC::E::value) {
                                printf("ERROR gradCombine: rank=%u block=%u expertIdx=%u >= E=%u OUT OF BOUNDS!\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.expertIdx, ACC::E::value);
                            }
                        }

                        // Bounds check: tileIdx must be < tNx to avoid OOB in splitGradients
                        if (rCurrentTask.tileIdx >= tNx) {
                            if (!threadIdx.x) {
                                printf("ERROR gradCombine OOB: rank=%u block=%u tileIdx=%u >= tNx=%u expert=%u syncIdx=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tNx,
                                       rCurrentTask.expertIdx, rCurrentTask.syncIdx);
                            }
                        }

                        if (!threadIdx.x) {
                            if (rCurrentTask.tileSize > BLOCK_M) {
                                printf("ERROR gradCombine: rank=%u block=%u tileSize=%u > BLOCK_M=%u - WRITE OVERFLOW!\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileSize, BLOCK_M);
                            }
                        }

                        // TPS range check before splitGradients
                        {
                            auto* tPBase = bookkeeping.tP();
                            auto* tPEnd  = tPBase + ACC::E::value * ACC::pEC::value;
                            const auto* rowTokenIndices = CONST_CAST_TO(TPS, rCurrentTask.aData);
                            if (!threadIdx.x && (reinterpret_cast<const cuda::std::byte*>(rowTokenIndices) < reinterpret_cast<const cuda::std::byte*>(tPBase) ||
                                            reinterpret_cast<const cuda::std::byte*>(rowTokenIndices + rCurrentTask.tileSize) > reinterpret_cast<const cuda::std::byte*>(tPEnd))) {
                                printf("ERROR TPS OOB gradCombine: expert=%u tileSize=%u "
                                       "rowPtr=%p tPBase=%p tPEnd=%p\n",
                                       rCurrentTask.expertIdx, rCurrentTask.tileSize,
                                       rowTokenIndices, tPBase, tPEnd);
                            }
                        }
                        // Use global gradOutputBasePtr instead of bData[0] (which is packet, not full buffer)
                        splitGradients<ACC::CM::value>(
                            workspace,
                            CONST_CAST_TO(TPS, rCurrentTask.aData),
                            flashmoe::moe::gradOutputBasePtr,
                            CAST_TO(Element, rCurrentTask.cData[gradIndex]),
                            sW,
                            rCurrentTask.tileIdx,
                            rCurrentTask.tileSize,
                            rCurrentTask.expertIdx);
// #if FLASHMOE_DEBUG
//                         if (!threadIdx.x) {
//                             printf("DEBUG gradCombine DONE: block=%u tile=%u\n", blockIdx.x, rCurrentTask.tileIdx);
//                         }
// #endif
// #if FLASHMOE_DEBUG
//                         if (!threadIdx.x && !blockIdx.x) {
//                             printf("DEBUG gradCombine split sb=%u tile=%u tileSize=%u expert=%u peer=%u remote=%u\n",
//                                    rSeqBit,
//                                    rCurrentTask.tileIdx,
//                                    rCurrentTask.tileSize,
//                                    rCurrentTask.expertIdx,
//                                    rCurrentTask.peerIdx,
//                                    static_cast<unsigned>(rCurrentTask.isPeerRemote));
//                         }
// #endif
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
                            const auto combineSyncIdx = rCurrentTask.syncIdx + bookkeeping.gtQCl;
                            // P2P: tNx gradCombine + 1 gradGateCombine = tNx+1 tasks per syncIdx
                            // Remote: tNx gradCombine + 1 gradGateCombine = tNx+1 tasks per syncIdx
                            const auto threshold = tNx + 1;
// #if FLASHMOE_DEBUG
//                             const auto priorCount = atomicAdd(pA.tQS + combineSyncIdx, 0U);
//                             printf("DEBUG gradCombine SYNC: block=%u syncIdx=%u combineSyncIdx=%u gtQCl=%u "
//                                    "priorCount=%u threshold=%u remote=%u expert=%u tile=%u\n",
//                                    blockIdx.x, rCurrentTask.syncIdx, combineSyncIdx, bookkeeping.gtQCl,
//                                    priorCount, threshold, rCurrentTask.isPeerRemote, rCurrentTask.expertIdx,
//                                    rCurrentTask.tileIdx);
//                             printf("DEBUG gradCombine METADATA: cData[0]=%p cData[1]=%p bData[0]=%p bData[1]=%p "
//                                    "dData[0]=%p dData[1]=%p M=%u tileSize=%u peerIdx=%u\n",
//                                    rCurrentTask.cData[0], rCurrentTask.cData[1],
//                                    rCurrentTask.bData[0], rCurrentTask.bData[1],
//                                    rCurrentTask.dData[0], rCurrentTask.dData[1],
//                                    rCurrentTask.M, rCurrentTask.tileSize, rCurrentTask.peerIdx);
// #endif
                            enqueue = atomicAdd(pA.tQS + combineSyncIdx, 1U) + 1 == threshold;
                        }
                        __syncthreads();
                        if (enqueue) {
                            // Clear sync counter so the same syncIdx can trigger again for the next packet
                            if (!threadIdx.x) {
                                const auto combineSyncIdx = rCurrentTask.syncIdx + bookkeeping.gtQCl;
                                atomicExch(pA.tQS + combineSyncIdx, 0U);
#if FLASHMOE_DEBUG
                                printf("DEBUG gradCombine TRIGGERED: rank=%u block=%u combineSyncIdx=%u "
                                       "threshold=%u -> emit gradPostGEMM/gradGateGEMM with syncIdx=%u\n", 
                                        nvshmem_my_pe(), blockIdx.x, combineSyncIdx,
                                       tNx + 1, rCurrentTask.syncIdx);
#endif
                            }
                            // Task cData layout from decoder:
                            //   - cData[0] = rowPacket (grad_output split destination, gradPreGEMM output)
                            //   - cData[1] = rowXM (xM row pointer for z1/z2 offset derivation)
                            //   - bData = [W1, W2] weights
                            //   - dData = [z1, z2] saved activations
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradCombine ENQUEUE: block=%u syncIdx=%u combineSyncIdx=%u remote=%u CALLING notifyGradient...\n",
                                       blockIdx.x, rCurrentTask.syncIdx, rCurrentTask.syncIdx + bookkeeping.gtQCl, rCurrentTask.isPeerRemote);
                            }
#endif
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }

                            rCurrentTask.cData[0] = CAST_TO(cuda::std::byte, bookkeeping.gGateCombine());
                            rCurrentTask.bData[0] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::hiddenStatesPtr);
                            rCurrentTask.bData[1] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::gateWeightsPtr);
                            rCurrentTask.dData[0] = CAST_TO(cuda::std::byte, flashmoe::moe::gradInputBasePtr);

#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradCombine ENQUEUE: block=%u syncIdx=%u CALLING notifyGateGradient...\n",
                                       blockIdx.x, rCurrentTask.syncIdx);
                            }
#endif
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGateGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGateGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradCombine ENQUEUE: block=%u syncIdx=%u DONE both notifications\n",
                                       blockIdx.x, rCurrentTask.syncIdx);
                            }
#endif
                        }
                    }
                    break;
                    case TaskType::gradGateCombine: {
                        constexpr unsigned int gradIndex = 0;
                        using ComputeElement = ACC::ElementC;
                        constexpr auto H = ACC::H::value;
                        constexpr auto E = ACC::E::value;
                        constexpr auto S = ACC::S::value;
                        const auto tileSize = rCurrentTask.tileSize;
                        // expertIdx is local; compute global for gateBuffer indexing ([S, E] buffer)
                        const auto localExpertIdx = rCurrentTask.expertIdx;
                        const auto globalExpertIdx = bookkeeping.lX()[localExpertIdx].expertIndex;
                        // Use global gradOutputBasePtr instead of cData[0] (which is packet, not full buffer)
                        const auto* __restrict__ const gradOut = flashmoe::moe::gradOutputBasePtr;
                        const auto mGradOut = make_tensor(cute::make_gmem_ptr(gradOut),
                            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                                cute::Stride<cute::Int<H>, cute::_1>>{});
                        // Use global gateBuffer - write directly to correct token positions using tokenIdx from TPS
                        auto* __restrict__ const gateBuffer = bookkeeping.gGateCombine();
                        auto* __restrict__ const tokenIds = CONST_CAST_TO(TPS, rCurrentTask.aData);
                        // Source y_e from savedZ2 instead of packet (cData[0]) to avoid race with gradCombine.
                        // Safe when ActivationOpX is identity (typical FFN): z2 == y_e.
                        // Derive z2 offset from xM pointer (cData[1] = rowXM) which includes expert/peer offset.
                        constexpr auto P = ACC::P::value;
                        auto* __restrict__ const z2Base = bookkeeping.z2();
                        auto* __restrict__ const xMBase = CAST_TO(Element, bookkeeping.xM());
                        auto* __restrict__ const xMPtr = CAST_TO(Element, rCurrentTask.cData[1]);  // rowXM
                        const auto xMOffset = xMPtr - xMBase;
                        const auto baseRowIdx = xMOffset / P;  // Includes expert/peer offset
                        auto* __restrict__ const z2RowBase = z2Base + baseRowIdx * H;
                        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
                        constexpr auto toCompute = cutlass::NumericConverter<ComputeElement, Element>{};
                        constexpr auto toElement = cutlass::NumericConverter<Element, ComputeElement>{};
                        using NativeElement = typename ToCDx<Element>::T;
                        constexpr auto convertNative = cutlass::NumericConverter<NativeElement, Element>{};
                        constexpr auto hiddenStride = static_cast<size_t>(H);

                        if (!threadIdx.x) {
                            if (gradOut == nullptr) {
                                printf("ERROR gradGateCombine: rank=%u block=%u gradOut is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (gateBuffer == nullptr) {
                                printf("ERROR gradGateCombine: rank=%u block=%u gateBuffer is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (tokenIds == nullptr) {
                                printf("ERROR gradGateCombine: rank=%u block=%u tokenIds (aData) is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (z2Base == nullptr) {
                                printf("ERROR gradGateCombine: rank=%u block=%u z2Base is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (xMBase == nullptr) {
                                printf("ERROR gradGateCombine: rank=%u block=%u xMBase is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (xMPtr == nullptr) {
                                printf("ERROR gradGateCombine: rank=%u block=%u xMPtr (cData[1]) is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (xMOffset < 0) {
                                printf("ERROR gradGateCombine: rank=%u block=%u NEGATIVE xMOffset=%lld "
                                       "(xMPtr=%p < xMBase=%p)\n",
                                       nvshmem_my_pe(), blockIdx.x, static_cast<long long>(xMOffset),
                                       xMPtr, xMBase);
                            }
                            if (baseRowIdx < 0) {
                                printf("ERROR gradGateCombine: rank=%u block=%u NEGATIVE baseRowIdx=%lld\n",
                                       nvshmem_my_pe(), blockIdx.x, static_cast<long long>(baseRowIdx));
                            }
                            if (localExpertIdx >= bookkeeping.nLx) {
                                printf("ERROR gradGateCombine: rank=%u block=%u localExpertIdx=%u >= nLx=%u OUT OF BOUNDS!\n",
                                       nvshmem_my_pe(), blockIdx.x, localExpertIdx, bookkeeping.nLx);
                            }
                            if (globalExpertIdx >= E) {
                                printf("ERROR gradGateCombine: rank=%u block=%u globalExpertIdx=%u >= E=%u OUT OF BOUNDS!\n",
                                       nvshmem_my_pe(), blockIdx.x, globalExpertIdx, E);
                            }
                            if (tileSize == 0) {
                                printf("ERROR gradGateCombine: rank=%u block=%u tileSize is ZERO!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                        }
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 2) {
                            printf("DEBUG gradGateCombine ENTER: rank=%u block=%u tile=%u tileSize=%u S=%u E=%u H=%u localExpert=%u globalExpert=%u "
                                   "gradOut=%p gateBuffer=%p tokenIds=%p z2RowBase=%p xMOffset=%lld baseRowIdx=%lld cData[1]=%p\n",
                                      nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tileSize, S, E, H, localExpertIdx, globalExpertIdx,
                                   gradOut, gateBuffer, tokenIds, z2RowBase,
                                   static_cast<long long>(xMOffset), static_cast<long long>(baseRowIdx),
                                   rCurrentTask.cData[1]);
                        }
#endif
                        // No per-tile zeroing needed - gateBuffer is pre-zeroed in bootstrap
                        // Write directly to global positions using tokenIdx from TPS array
                        for (uint idx = threadIdx.x; idx < tileSize; idx += threads) {
                            const auto tokenEntry = tokenIds[idx];
                            const auto tokenIdx = tokenEntry.tokenIdx;
                            if (tokenIdx >= S) {
#if FLASHMOE_DEBUG
                                if (!threadIdx.x) {
                                    printf("DEBUG gradGateCombine: rank=%u block=%u SKIP tokenIdx=%u >= S=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, tokenIdx, S);
                                }
#endif
                                continue;
                            }
                            const auto rowOffset = static_cast<size_t>(idx) * hiddenStride;
                            const auto* __restrict__ const expertRow = z2RowBase + rowOffset;
                            ComputeElement gradSum = ComputeElement(0);
                            #pragma unroll
                            for (uint n = 0; n < H; ++n) {
                                gradSum = fmaf(toCompute(mGradOut(tokenIdx, n)), toCompute(expertRow[n]), gradSum);
                            }
                            const auto routingVal = toElement(gradSum);

                            {
                                const float gradSumF = static_cast<float>(gradSum);
                                if (isnan(gradSumF) || isinf(gradSumF)) {
                                    printf("ERROR gradGateCombine: rank=%u block=%u idx=%u tokenIdx=%u globalExpert=%u "
                                           "gradSum is %s (%.8e)! Sampling inputs: gradOut[0]=%.8e expertRow[0]=%.8e\n",
                                           nvshmem_my_pe(), blockIdx.x, idx, tokenIdx, globalExpertIdx,
                                           isnan(gradSumF) ? "NaN" : "Inf", gradSumF,
                                           static_cast<float>(mGradOut(tokenIdx, 0)),
                                           static_cast<float>(expertRow[0]));
                                }
                            }

                            const auto writeIdx = static_cast<size_t>(tokenIdx) * E + globalExpertIdx;
                            constexpr auto gateBufferSize = static_cast<size_t>(S) * E;
                            if (writeIdx >= gateBufferSize) {
                                printf("ERROR gradGateCombine: rank=%u block=%u WRITE OUT OF BOUNDS! "
                                       "writeIdx=%lu >= gateBufferSize=%lu (S=%u E=%u tokenIdx=%u globalExpertIdx=%u)\n",
                                       nvshmem_my_pe(), blockIdx.x,
                                       static_cast<unsigned long>(writeIdx),
                                       static_cast<unsigned long>(gateBufferSize),
                                       S, E, tokenIdx, globalExpertIdx);
                                continue;
                            }
#if FLASHMOE_DEBUG
                            if (blockIdx.x < 2 && idx < 2) {
                                const float go0 = static_cast<float>(mGradOut(tokenIdx, 0));
                                const float go1 = static_cast<float>(mGradOut(tokenIdx, 1));
                                const float go2 = static_cast<float>(mGradOut(tokenIdx, 2));
                                const float go3 = static_cast<float>(mGradOut(tokenIdx, 3));
                                const float er0 = static_cast<float>(expertRow[0]);
                                const float er1 = static_cast<float>(expertRow[1]);
                                const float er2 = static_cast<float>(expertRow[2]);
                                const float er3 = static_cast<float>(expertRow[3]);
                                printf("DEBUG gradGateCombine DOT: rank= %u block=%u idx=%u tokenIdx=%u rowOffset=%lu globalExpert=%u\n",
                                      nvshmem_my_pe(), blockIdx.x, idx, tokenIdx, rowOffset, globalExpertIdx);
                                printf("  gradOut[%u,0..3]=[%.8e,%.8e,%.8e,%.8e]\n",
                                       tokenIdx, go0, go1, go2, go3);
                                printf("  expertRow[0..3]=[%.8e,%.8e,%.8e,%.8e]\n",
                                       er0, er1, er2, er3);
                                printf("  gradSum=% .8e\n routingVal=%.8e slot=%p\n",
                                       static_cast<float>(gradSum), static_cast<float>(routingVal),
                                       gateBuffer + tokenIdx * E + globalExpertIdx);
                            }
#endif
                            // Write to global position: gateBuffer[tokenIdx, globalExpertIdx]
                            auto* const slot = gateBuffer + tokenIdx * E + globalExpertIdx;
                            atomicAdd(CAST_TO(NativeElement, slot), convertNative(routingVal));
                        }
#if FLASHMOE_DEBUG
                        __syncthreads();
                        if (!threadIdx.x && blockIdx.x < 2) {
                            const auto firstTokenIdx = tokenIds[0].tokenIdx;
                            if (firstTokenIdx < S) {
                                auto* slot = gateBuffer + firstTokenIdx * E + globalExpertIdx;
                                printf("DEBUG gradGateCombine SUMMARY: block=%u tile=%u firstToken=%u localExpert=%u globalExpert=%u\n",
                                       blockIdx.x, rCurrentTask.tileIdx, firstTokenIdx, localExpertIdx, globalExpertIdx);
                                printf("  gateBuffer[%u, %u] slot=%p val=%.9e\n",
                                       firstTokenIdx, globalExpertIdx, slot, static_cast<float>(*slot));
                            }
                        }
#endif
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
                            // gradPostGEMM uses tQS[0, gtQCl), combine tasks use tQS[gtQCl, 2*gtQCl)
                            const auto combineSyncIdx = rCurrentTask.syncIdx + bookkeeping.gtQCl;
                            // P2P: tNx gradCombine + 1 gradGateCombine = tNx+1 tasks per syncIdx
                            // Remote: tNx gradCombine + 1 gradGateCombine = tNx+1 tasks per syncIdx
                            const auto threshold = tNx + 1;
// #if FLASHMOE_DEBUG
//                             const auto priorCount = atomicAdd(pA.tQS + combineSyncIdx, 0U);
//                             printf("DEBUG gradGateCombine SYNC: block=%u syncIdx=%u combineSyncIdx=%u "
//                                    "priorCount=%u threshold=%u remote=%u localExpert=%u globalExpert=%u tile=%u\n",
//                                    blockIdx.x, rCurrentTask.syncIdx, combineSyncIdx,
//                                    priorCount, threshold, rCurrentTask.isPeerRemote, localExpertIdx, globalExpertIdx,
//                                    rCurrentTask.tileIdx);
//                             printf("DEBUG gradGateCombine METADATA: cData[0]=%p cData[1]=%p bData[0]=%p bData[1]=%p "
//                                    "dData[0]=%p dData[1]=%p M=%u tileSize=%u peerIdx=%u\n",
//                                    rCurrentTask.cData[0], rCurrentTask.cData[1],
//                                    rCurrentTask.bData[0], rCurrentTask.bData[1],
//                                    rCurrentTask.dData[0], rCurrentTask.dData[1],
//                                    rCurrentTask.M, rCurrentTask.tileSize, rCurrentTask.peerIdx);
// #endif
                            enqueue = atomicAdd(pA.tQS + combineSyncIdx, 1U) + 1 == threshold;
                        }
                        __syncthreads();
                        if (enqueue) {
                            // Clear sync counter so the same syncIdx can trigger again for the next packet
                            if (!threadIdx.x) {
                                const auto combineSyncIdx = rCurrentTask.syncIdx + bookkeeping.gtQCl;
                                atomicExch(pA.tQS + combineSyncIdx, 0U);
                            }
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradGateCombine TRIGGERED: block=%u combineSyncIdx=%u "
                                       "threshold=%u -> emit gradPostGEMM/gradGateGEMM with syncIdx=%u\n",
                                       blockIdx.x, rCurrentTask.syncIdx + bookkeeping.gtQCl,
                                       tNx + 1, rCurrentTask.syncIdx);
                            }
#endif
                            // Task cData layout from decoder:
                            //   - cData[0] = rowPacket (grad_output split destination, gradPreGEMM output)
                            //   - cData[1] = rowXM (xM row pointer for z1/z2 offset derivation)
                            //   - bData = [W1, W2] weights
                            //   - dData = [z1, z2] saved activations
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            rCurrentTask.cData[gradIndex] = CAST_TO(cuda::std::byte, gateBuffer);
                            rCurrentTask.bData[0] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::hiddenStatesPtr);
                            rCurrentTask.bData[1] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::gateWeightsPtr);
                            rCurrentTask.dData[0] = CAST_TO(cuda::std::byte, flashmoe::moe::gradInputBasePtr);

                            if (!rCurrentTask.isPeerRemote) {
                                notifyGateGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGateGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradGateCombine ENQUEUE: block=%u syncIdx=%u DONE notifications\n",
                                       blockIdx.x, rCurrentTask.syncIdx);
                            }
#endif
                        }
                    }
                    break;
                    case TaskType::gradGateGEMM: {
                        constexpr auto H = ACC::H::value;
                        constexpr auto E = ACC::E::value;
                        constexpr auto S = ACC::S::value;
                        const auto tileSize = rCurrentTask.tileSize;
                        const auto* __restrict__ const tokenIds = CONST_CAST_TO(TPS, rCurrentTask.aData);
                        const auto* __restrict__ const hiddenStates = CONST_CAST_TO(Element, rCurrentTask.bData[0]);
                        const auto* __restrict__ const gateWeights = CONST_CAST_TO(Element, rCurrentTask.bData[1]);
                        const auto* __restrict__ const gradRouting = CONST_CAST_TO(Element, rCurrentTask.cData[0]);
                        auto* __restrict__ const gradInput = const_cast<Element*>(CONST_CAST_TO(Element, rCurrentTask.dData[0]));
                        auto* __restrict__ const routingScores = bookkeeping.gateRoutingScores();
                        auto* __restrict__ const gateWeightGrad = bookkeeping.gGateW();
                        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
                        constexpr auto toCompute = cutlass::NumericConverter<ACC::ElementC, Element>{};
                        constexpr auto toElement = cutlass::NumericConverter<Element, ACC::ElementC>{};
                        using NativeElement = typename ToCDx<Element>::T;
                        constexpr auto convertNative = cutlass::NumericConverter<NativeElement, Element>{};
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            printf("DEBUG gradGateGEMM rank=%d block=%u tile=%u tileSize=%u syncIdx=%u "
                                   "gradRouting=%p gradInput=%p gateWeightGrad=%p tokenIdx[0]=%u\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tileSize, rCurrentTask.syncIdx,
                                   gradRouting, gradInput, gateWeightGrad,
                                   tileSize > 0 ? tokenIds[0].tokenIdx : 0u);
                        }
#endif

                        for (uint t = threadIdx.x; t < tileSize; t += threads) {
                            const auto tokenEntry = tokenIds[t];
                            const auto tokenIdx = tokenEntry.tokenIdx;
                            if (tokenIdx >= S) {
                                continue;
                            }
                            // Read gradRouting at correct global position (gateBuffer[tokenIdx, :])
                            const auto* __restrict__ const gradRoutingRow = gradRouting + tokenIdx * E;
                            const auto* __restrict__ const scoreRow = routingScores + tokenIdx * E;
                            ACC::ElementC maxScore = -cuda::std::numeric_limits<ACC::ElementC>::infinity();
                            ACC::ElementC expRow[E];
                            #pragma unroll
                            for (uint e = 0; e < E; ++e) {
                                const auto val = toCompute(scoreRow[e]);
                                if (val > maxScore) {
                                    maxScore = val;
                                }
                                expRow[e] = val;
                            }
                            ACC::ElementC sumExp = ACC::ElementC(0);
                            #pragma unroll
                            for (uint e = 0; e < E; ++e) {
                                expRow[e] = __expf(expRow[e] - maxScore);
                                sumExp = fmaf(expRow[e], ACC::ElementC(1), sumExp);
                            }
                            if (sumExp == ACC::ElementC(0)) {
                                sumExp = ACC::ElementC(1);
                            }
                            ACC::ElementC softmaxRow[E];
                            #pragma unroll
                            for (uint e = 0; e < E; ++e) {
                                softmaxRow[e] = expRow[e] / sumExp;
                            }
                            ACC::ElementC dot = ACC::ElementC(0);
                            #pragma unroll
                            for (uint e = 0; e < E; ++e) {
                                dot = fmaf(softmaxRow[e], toCompute(gradRoutingRow[e]), dot);
                            }
                            ACC::ElementC gradRoutingSoftmax[E];
                            #pragma unroll
                            for (uint e = 0; e < E; ++e) {
                                gradRoutingSoftmax[e] = softmaxRow[e] * (toCompute(gradRoutingRow[e]) - dot);
                            }
                            // Access hiddenStates and gradInput at correct global positions using tokenIdx
                            const auto* __restrict__ const hiddenRow = hiddenStates + tokenIdx * H;
                            auto* __restrict__ const inputRow = gradInput + tokenIdx * H;
                            #pragma unroll
                            for (uint h = 0; h < H; ++h) {
                                const auto hiddenVal = toCompute(hiddenRow[h]);
                                #pragma unroll
                                for (uint e = 0; e < E; ++e) {
                                    const auto product = hiddenVal * gradRoutingSoftmax[e];
                                    const auto productElem = toElement(product);
                                    auto* const gradWeightPtr = gateWeightGrad + h * E + e;
                                    atomicAdd(CAST_TO(NativeElement, gradWeightPtr), convertNative(productElem));
                                }
                                ACC::ElementC inputAcc = ACC::ElementC(0);
                                #pragma unroll
                                for (uint e = 0; e < E; ++e) {
                                    const auto weightVal = toCompute(gateWeights[e * H + h]);
                                    inputAcc = fmaf(weightVal, gradRoutingSoftmax[e], inputAcc);
                                }
                                const auto inputElem = toElement(inputAcc);
                                atomicAdd(CAST_TO(NativeElement, inputRow + h), convertNative(inputElem));
                            }
                        }
#if FLASHMOE_DEBUG
                        __syncthreads();
                        if (!threadIdx.x && blockIdx.x < 3) {
                            printf("DEBUG gradGateGEMM COMPLETE: rank=%d block=%u tile=%u tileSize=%u syncIdx=%u "
                                   "gateWeightGrad[0]=%.6f gradInput=%p\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tileSize, rCurrentTask.syncIdx,
                                   static_cast<float>(gateWeightGrad[0]), gradInput);
                        }
#endif
                    }
                    break;
                    case TaskType::gradPostGEMM: {
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            printf("DEBUG gradPostGEMM CASE ENTERED: rank=%d block=%u tile=%u aData=%p\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.aData);
                        }
#endif
                        if (!threadIdx.x) {
                            if (rCurrentTask.aData == nullptr) {
                                printf("ERROR gradPostGEMM: rank=%u block=%u aData (grad_output) is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (rCurrentTask.bData[1] == nullptr) {
                                printf("ERROR gradPostGEMM: rank=%u block=%u bData[1] (W2 weights) is NULL!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            if (rCurrentTask.tileSize == 0) {
                                printf("ERROR gradPostGEMM: rank=%u block=%u tileSize is ZERO!\n",
                                       nvshmem_my_pe(), blockIdx.x);
                            }
                            // ERROR: Validate expertIdx bounds for gW buffer access
                            if (rCurrentTask.expertIdx >= bookkeeping.nLx) {
                                printf("ERROR gradPostGEMM: rank=%u block=%u expertIdx=%u >= nLx=%u OUT OF BOUNDS!\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.expertIdx, bookkeeping.nLx);
                            }
                        }

                        if (!threadIdx.x) {
                            if (rCurrentTask.tileSize > BLOCK_M) {
                                printf("ERROR gradPostGEMM: rank=%u block=%u tileSize=%u > BLOCK_M=%u - WRITE OVERFLOW!\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileSize, BLOCK_M);
                            }
                        }

                        // Bounds check: tileIdx must be < tN to avoid OOB
                        if (rCurrentTask.tileIdx >= tN) {
                            if (!threadIdx.x) {
                                printf("ERROR gradPostGEMM OOB: rank=%u block=%u tileIdx=%u >= tN=%u expert=%u syncIdx=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tN,
                                       rCurrentTask.expertIdx, rCurrentTask.syncIdx);
                            }
                            break;
                        }
                        constexpr unsigned int w2Index = 1;
                        constexpr unsigned int w1Index = 0;
                        const Element* z2Activation = nullptr;
                        const Element* a1Activation = nullptr;
                        auto* xMBase = CAST_TO(Element, bookkeeping.xM());
                        auto* xMPtr = CAST_TO(Element, rCurrentTask.cData[w2Index]);  // cData[1] = rowXM
                        const auto xMOffset2 = xMPtr - xMBase;
                        const auto rowIdx2 = xMOffset2 / P;
                        z2Activation = bookkeeping.z2() + rowIdx2 * H;
                        a1Activation = xMPtr;
                        // Bounds checks for cData pointers and derived z2Activation
                        {
                            const auto xMTotalRows = bookkeeping.world * bookkeeping.nLx * ACC::pEC::value;
                            auto* z2Base = bookkeeping.z2();
                            auto* z2End = z2Base + xMTotalRows * H;
                            auto* xMEnd = xMBase + xMTotalRows * P;
                            // Check cData[1] (xM row) is within xM buffer
                            if (!threadIdx.x && (xMPtr < xMBase || xMPtr >= xMEnd)) {
                                printf("ERROR gradPostGEMM cData[1] OOB: rank=%u block=%u tile=%u "
                                       "cData[1]=%p xMBase=%p xMEnd=%p expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       xMPtr, xMBase, xMEnd, rCurrentTask.expertIdx);
                            }
                            // Check cData[0] (packet row) is not null
                            if (!threadIdx.x && rCurrentTask.cData[w1Index] == nullptr) {
                                printf("ERROR gradPostGEMM cData[0] NULL: rank=%u block=%u tile=%u expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.expertIdx);
                            }
                            // Check z2Activation is within z2 buffer
                            if (!threadIdx.x && (z2Activation < z2Base || z2Activation >= z2End)) {
                                printf("ERROR gradPostGEMM z2Activation OOB: rank=%u block=%u tile=%u "
                                       "z2Act=%p z2Base=%p z2End=%p rowIdx=%lld expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       z2Activation, z2Base, z2End, rowIdx2, rCurrentTask.expertIdx);
                            }
                            // Check rowIdx2 is non-negative
                            if (!threadIdx.x && rowIdx2 < 0) {
                                printf("ERROR gradPostGEMM rowIdx2 negative: rank=%u block=%u tile=%u "
                                       "rowIdx=%lld xMOffset=%lld expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       rowIdx2, xMOffset2, rCurrentTask.expertIdx);
                            }
                            // Check dData (savedZ2) pointers are valid (propagated from gradCombine)
                            if (!threadIdx.x && (rCurrentTask.dData[0] == nullptr || rCurrentTask.dData[1] == nullptr)) {
                                printf("ERROR gradPostGEMM dData NULL: rank=%u block=%u tile=%u "
                                       "dData[0]=%p dData[1]=%p expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       rCurrentTask.dData[0], rCurrentTask.dData[1], rCurrentTask.expertIdx);
                            }
                            constexpr auto expertStride = 2 * P * H + P + H;
                            auto* gWBase = CAST_TO(Element, bookkeeping.gW());
                            auto* gWEnd = gWBase + bookkeeping.nLx * expertStride;
                            auto* dW2Start = gWBase + rCurrentTask.expertIdx * expertStride + P * H;
                            auto* dW2End = dW2Start + H * P;
                            if (!threadIdx.x && (dW2Start < gWBase || dW2End > gWEnd)) {
                                printf("ERROR gradPostGEMM gW OOB: rank=%u block=%u tile=%u expert=%u "
                                       "dW2Start=%p dW2End=%p gWBase=%p gWEnd=%p stride=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       rCurrentTask.expertIdx, dW2Start, dW2End, gWBase, gWEnd, expertStride);
                            }
                        }
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            auto* z2Base = bookkeeping.z2();
                            printf("DEBUG gradPostGEMM ENTER rank=%d block=%u tile=%u M=%u: xMOffset=%lld rowIdx=%lld z2Base=%p z2Act=%p cData[0]=%p\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.M,
                                   xMOffset2, rowIdx2, z2Base, z2Activation, rCurrentTask.cData[w1Index]);
                        }
//                         // DEBUG: Print INPUT values for gradPostGEMM correctness verification
//                         // Math: grad_intermediate = (grad_output * act'(z2)) @ W2^T  (W2 stored as [H,P])
//                         // For identity activation: grad_intermediate = grad_output @ W2
//                         if (!threadIdx.x && nvshmem_my_pe() == 0 && blockIdx.x < 4) {
//                             auto* gradOut = CONST_CAST_TO(Element, rCurrentTask.aData);  // grad_output [M, H]
//                             auto* W2 = CONST_CAST_TO(Element, rCurrentTask.bData[w2Index]);  // W2 [H, P]
//                             printf("DEBUG gradPostGEMM VALUES: rank=%d block=%u tileIdx=%u M=%u expert=%u syncIdx=%u\n",
//                                    nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.M,
//                                    rCurrentTask.expertIdx, rCurrentTask.syncIdx);
//                             // Sample grad_output values (first row, first 4 cols)
//                             printf("  grad_output[0, 0..3]: %.6e %.6e %.6e %.6e\n",
//                                    static_cast<float>(gradOut[0 * H + 0]),
//                                    static_cast<float>(gradOut[0 * H + 1]),
//                                    static_cast<float>(gradOut[0 * H + 2]),
//                                    static_cast<float>(gradOut[0 * H + 3]));
//                             // Sample z2 values (activation derivative input)
//                             printf("  z2[0, 0..3]: %.6e %.6e %.6e %.6e\n",
//                                    static_cast<float>(z2Activation[0 * H + 0]),
//                                    static_cast<float>(z2Activation[0 * H + 1]),
//                                    static_cast<float>(z2Activation[0 * H + 2]),
//                                    static_cast<float>(z2Activation[0 * H + 3]));
//                             // Sample W2 values (first row = W2[0,:], which contributes to output[:,0])
//                             // W2 is [H, P], so W2[h, p] = W2[h * P + p]
//                             printf("  W2[0..3, 0]: %.6e %.6e %.6e %.6e (first 4 rows of col 0)\n",
//                                    static_cast<float>(W2[0 * P + 0]),
//                                    static_cast<float>(W2[1 * P + 0]),
//                                    static_cast<float>(W2[2 * P + 0]),
//                                    static_cast<float>(W2[3 * P + 0]));
//                             // Compute partial expected output[0,0] = sum_h (grad_output[0,h] * act'(z2[0,h])) * W2[h,0]
//                             // For identity activation: act'(z2) = 1, so output[0,0] = sum_h grad_output[0,h] * W2[h,0]
//                             float partial_sum = 0.0f;
//                             for (uint h = 0; h < min(4u, H); ++h) {
//                                 partial_sum += static_cast<float>(gradOut[0 * H + h]) * static_cast<float>(W2[h * P + 0]);
//                             }
//                             printf("  partial expected output[0,0] (first 4 h terms): %.6e\n", partial_sum);
//                         }
#endif
                        // grad_output [M, H], z2 [M, H] (K=H), W2_stored [H, P] (K=H, N=P), output [M, P]
                        fGETGrad<GradPostGEMM, ACC::P::value, ACC::H::value>(
                            CAST_TO(typename GradPostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename GradPostGEMM::MatrixAType, rCurrentTask.aData),  // grad_output [M, H]
                            CONST_CAST_TO(typename GradPostGEMM::MatrixBType, rCurrentTask.bData[w2Index]),  // W2 [H, P]
                            CAST_TO(typename GradPostGEMM::MatrixDType, rCurrentTask.cData[w2Index]),  // output [M, P]
                            CONST_CAST_TO(typename GradPostGEMM::MatrixDType, z2Activation),  // z2 [M, H] (K=H)
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
// #if FLASHMOE_DEBUG
//                         // DEBUG: Print OUTPUT values after fGETGrad to verify correctness
//                         __syncthreads();
//                         if (!threadIdx.x && nvshmem_my_pe() == 0 && blockIdx.x < 4) {
//                             auto* output = CAST_TO(Element, rCurrentTask.cData[w2Index]);  // grad_intermediate [M, P]
//                             auto* gradOut = CONST_CAST_TO(Element, rCurrentTask.aData);    // grad_output [M, H]
//                             auto* W2 = CONST_CAST_TO(Element, rCurrentTask.bData[w2Index]); // W2 [H, P]
//                             // tileIdx encodes 2D position: tileIdx = rowTile * tilesP + colTile
//                             // tilesP = P / BLOCK_N (number of column tiles)
//                             constexpr uint tilesP = P / BLOCK_N;
//                             const uint rowTile = rCurrentTask.tileIdx / tilesP;
//                             const uint colTile = rCurrentTask.tileIdx % tilesP;
//                             const uint outRowOffset = rowTile * BLOCK_M;
//                             const uint outColOffset = colTile * BLOCK_N;
//                             printf("  OUTPUT grad_intermediate[row=%u, col=%u..%u]: %.6e %.6e %.6e %.6e (tile=%u rowT=%u colT=%u)\n",
//                                    outRowOffset, outColOffset, outColOffset + 3,
//                                    static_cast<float>(output[outRowOffset * P + outColOffset + 0]),
//                                    static_cast<float>(output[outRowOffset * P + outColOffset + 1]),
//                                    static_cast<float>(output[outRowOffset * P + outColOffset + 2]),
//                                    static_cast<float>(output[outRowOffset * P + outColOffset + 3]),
//                                    rCurrentTask.tileIdx, rowTile, colTile);
//                             // Full expected calculation for output[0, outColOffset]
//                             // output[row, col] = sum_h (grad_output[row, h] * act'(z2[row, h])) * W2[h, col]
//                             // For identity: output[0, outColOffset] = sum_h grad_output[0, h] * W2[h, outColOffset]
//                             if (outRowOffset == 0) {
//                                 float full_sum = 0.0f;
//                                 for (uint h = 0; h < H; ++h) {
//                                     full_sum += static_cast<float>(gradOut[0 * H + h]) * static_cast<float>(W2[h * P + outColOffset]);
//                                 }
//                                 printf("  VERIFY: expected output[0,%u]=%.6e, actual=%.6e, diff=%.6e\n",
//                                        outColOffset, full_sum,
//                                        static_cast<float>(output[0 * P + outColOffset]),
//                                        full_sum - static_cast<float>(output[0 * P + outColOffset]));
//                             }
//                         }
// #endif
                        {
                            const auto localExpertIdx = rCurrentTask.expertIdx;
                            constexpr auto expertStride = 2 * P * H + P + H;
                            auto* const weightGradBuffer = CAST_TO(Element, bookkeeping.gW()) + localExpertIdx * expertStride;
                            auto* const dW2Buffer = weightGradBuffer + P * H;
                            computeWeightGradients<H, P>(
                                workspace,
                                CONST_CAST_TO(Element, rCurrentTask.aData),  // grad_z2 [M, H]
                                CONST_CAST_TO(Element, a1Activation),        // a1 [M, P] from xM()
                                dW2Buffer,
                                rCurrentTask.tileSize,
                                static_cast<uint16_t>(localExpertIdx));
                        }
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
// #if FLASHMOE_DEBUG
//                             const auto priorCount = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 0U);
//                             printf("DEBUG gradPostGEMM SYNC: block=%u syncIdx=%u (combine used %u+gtQCl) "
//                                    "priorCount=%u tN=%u expert=%u tile=%u\n",
//                                    blockIdx.x, rCurrentTask.syncIdx, rCurrentTask.syncIdx,
//                                    priorCount, tN, rCurrentTask.expertIdx, rCurrentTask.tileIdx);
// #endif
                            enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) + 1 == tN;
#if FLASHMOE_DEBUG
                            if (enqueue && blockIdx.x < 3) {
                                printf("DEBUG gradPostGEMM TRIGGERED: rank=%d block=%u syncIdx=%u reached tN=%u -> notifyGradPreGEMM\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.syncIdx, tN);
                            }
#endif
                        }
                        __syncthreads();
                        if (enqueue) {
// #if FLASHMOE_DEBUG
//                             if (!threadIdx.x) {
//                                 printf("DEBUG gradPostGEMM block=%u CALLING notifyGradPreGEMM remote=%u\n",
//                                        blockIdx.x, rCurrentTask.isPeerRemote ? 1U : 0U);
//                             }
// #endif
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGradPreGEMM<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGradPreGEMM<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                        }
                    }
                    break;
                    case TaskType::gradPreGEMM: {
                        // Bounds check: tileIdx must be < tNx to avoid OOB
                        // gradPreGEMM output is [M, H], so column tiles = TNx = ceil(H/BLOCK_N)
                        if (rCurrentTask.tileIdx >= tNx) {
                            if (!threadIdx.x) {
                                printf("ERROR gradPreGEMM OOB: rank=%u block=%u tileIdx=%u >= tNx=%u expert=%u syncIdx=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tNx,
                                       rCurrentTask.expertIdx, rCurrentTask.syncIdx);
                            }
                            break;
                        }
                        constexpr unsigned int w1Index = 0;   // Output destination (packet buffer)
                        constexpr unsigned int xMIndex = 1;   // xM row pointer (same as w2Index in gradPostGEMM)
                        // Compute z1 offset for saved activation (same layout as forward preGEMM)
                        // z1 was stored at same offset as cData[1] (xM row pointer) relative to xM base
                        // Note: cData[0] is packet (output dest), cData[1] is xM (for offset calculation)
                        const Element* z1Activation = nullptr;
                        long long xMOffset = 0;
                        auto* xMBase = CAST_TO(Element, bookkeeping.xM());
                        {
                            auto* xMPtr = CAST_TO(Element, rCurrentTask.cData[xMIndex]);
                            xMOffset = xMPtr - xMBase;
                            z1Activation = bookkeeping.z1() + xMOffset;
                        }
                        // Bounds checks for aData, z1Activation, cData, and dData
                        {
                            const auto xMTotalSize = bookkeeping.world * bookkeeping.nLx * ACC::pEC::value * P;
                            auto* z1Base = bookkeeping.z1();
                            auto* z1End = z1Base + xMTotalSize;
                            auto* xMEnd = xMBase + xMTotalSize;
                            auto* xMDataPtr = CAST_TO(Element, rCurrentTask.cData[xMIndex]);
                            // Check xMOffset is non-negative
                            if (!threadIdx.x && xMOffset < 0) {
                                printf("ERROR gradPreGEMM xMOffset negative: rank=%u block=%u tile=%u "
                                       "xMOffset=%lld cData[1]=%p xMBase=%p expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       xMOffset, xMDataPtr, xMBase, rCurrentTask.expertIdx);
                            }
                            // Check cData[1] (xM pointer) is within xM buffer
                            if (!threadIdx.x && (xMDataPtr < xMBase || xMDataPtr >= xMEnd)) {
                                printf("ERROR gradPreGEMM cData[1] OOB: rank=%u block=%u tile=%u "
                                       "cData[1]=%p xMBase=%p xMEnd=%p expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       xMDataPtr, xMBase, xMEnd, rCurrentTask.expertIdx);
                            }
                            // Check z1Activation is within z1 buffer
                            if (!threadIdx.x && (z1Activation < z1Base || z1Activation >= z1End)) {
                                printf("ERROR gradPreGEMM z1Activation OOB: rank=%u block=%u tile=%u "
                                       "z1Act=%p z1Base=%p z1End=%p xMOffset=%lld expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       z1Activation, z1Base, z1End, xMOffset, rCurrentTask.expertIdx);
                            }
                            // Check aData (grad_a1 from gradPostGEMM) is not null
                            if (!threadIdx.x && rCurrentTask.aData == nullptr) {
                                printf("ERROR gradPreGEMM aData NULL: rank=%u block=%u tile=%u expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.expertIdx);
                            }
                            // Check dData (savedZ1) pointers are valid
                            if (!threadIdx.x && (rCurrentTask.dData[0] == nullptr || rCurrentTask.dData[1] == nullptr)) {
                                printf("ERROR gradPreGEMM dData NULL: rank=%u block=%u tile=%u "
                                       "dData[0]=%p dData[1]=%p expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       rCurrentTask.dData[0], rCurrentTask.dData[1], rCurrentTask.expertIdx);
                            }
                        }
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            auto* z1Base = bookkeeping.z1();
                            printf("DEBUG gradPreGEMM rank=%d block=%u tile=%u M=%u: xMOffset=%lld z1Base=%p z1Act=%p aData=%p cData[0]=%p cData[1]=%p\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.M,
                                   xMOffset, z1Base, z1Activation,
                                   rCurrentTask.aData, rCurrentTask.cData[w1Index], rCurrentTask.cData[xMIndex]);
                        }
                        // DEBUG: Print INPUT values for gradPreGEMM correctness verification
                        // Math: grad_input = (grad_a1 * act'(z1)) @ W1  (W1 stored as [P,H])
                        // act'(z1) is ReLU' (HIDDEN_ACT=0) or GELU' (HIDDEN_ACT=1)
                        if (!threadIdx.x && nvshmem_my_pe() == 0 && blockIdx.x < 4) {
                            auto* gradA1 = CONST_CAST_TO(Element, rCurrentTask.aData);  // grad_a1 [M, P]
                            auto* W1 = CONST_CAST_TO(Element, rCurrentTask.bData[w1Index]);  // W1 [P, H]
                            printf("DEBUG gradPreGEMM VALUES: rank=%d block=%u tileIdx=%u M=%u expert=%u syncIdx=%u\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.M,
                                   rCurrentTask.expertIdx, rCurrentTask.syncIdx);
                            // Sample grad_a1 values (first row, first 4 cols)
                            printf("  grad_a1[0, 0..3]: %.6e %.6e %.6e %.6e\n",
                                   static_cast<float>(gradA1[0 * P + 0]),
                                   static_cast<float>(gradA1[0 * P + 1]),
                                   static_cast<float>(gradA1[0 * P + 2]),
                                   static_cast<float>(gradA1[0 * P + 3]));
                            // Sample z1 values (activation derivative input)
                            printf("  z1[0, 0..3]: %.6e %.6e %.6e %.6e\n",
                                   static_cast<float>(z1Activation[0 * P + 0]),
                                   static_cast<float>(z1Activation[0 * P + 1]),
                                   static_cast<float>(z1Activation[0 * P + 2]),
                                   static_cast<float>(z1Activation[0 * P + 3]));
                            // Sample W1 values (W1 is [P, H], so W1[p, h] = W1[p * H + h])
                            printf("  W1[0..3, 0]: %.6e %.6e %.6e %.6e (first 4 rows of col 0)\n",
                                   static_cast<float>(W1[0 * H + 0]),
                                   static_cast<float>(W1[1 * H + 0]),
                                   static_cast<float>(W1[2 * H + 0]),
                                   static_cast<float>(W1[3 * H + 0]));
                            // Compute partial expected output[0,0] with activation derivative
                            // ReLU: act'(z) = z > 0 ? 1 : 0
                            float partial_sum = 0.0f;
                            for (uint p = 0; p < min(4u, P); ++p) {
                                float grad_val = static_cast<float>(gradA1[0 * P + p]);
                                float z1_val = static_cast<float>(z1Activation[0 * P + p]);
                                float w1_val = static_cast<float>(W1[p * H + 0]);
                                // ReLU derivative
                                float act_deriv = z1_val > 0.0f ? 1.0f : 0.0f;
                                partial_sum += grad_val * act_deriv * w1_val;
                            }
                            printf("  partial expected output[0,0] (first 4 p terms, ReLU deriv): %.6e\n", partial_sum);
                        }
#endif
                        // grad_a1 [M, P], z1 [M, P] (K=P), W1_stored [P, H] (K=P, N=H), output [M, H]
                        fGETGrad<GradPreGEMM, ACC::H::value, ACC::P::value>(
                            CAST_TO(typename GradPreGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename GradPreGEMM::MatrixAType, rCurrentTask.aData),  // grad_a1 [M, P]
                            CONST_CAST_TO(typename GradPreGEMM::MatrixBType, rCurrentTask.bData[w1Index]),  // W1 [P, H]
                            CAST_TO(typename GradPreGEMM::MatrixDType, rCurrentTask.cData[w1Index]),  // output [M, H]
                            CONST_CAST_TO(typename GradPreGEMM::MatrixDType, z1Activation),  // z1 [M, P] (K=P)
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
#if FLASHMOE_DEBUG
                        // DEBUG: Print OUTPUT values after fGETGrad to verify correctness
                        if (!threadIdx.x && nvshmem_my_pe() == 0 && blockIdx.x < 4) {
                            auto* output = CAST_TO(Element, rCurrentTask.cData[w1Index]);  // grad_input [M, H]
                            auto* gradA1 = CONST_CAST_TO(Element, rCurrentTask.aData);     // grad_a1 [M, P]
                            auto* W1 = CONST_CAST_TO(Element, rCurrentTask.bData[w1Index]); // W1 [P, H]
                            // tileIdx encodes 2D position: tileIdx = rowTile * tilesH + colTile
                            // tilesH = H / BLOCK_N (number of column tiles)
                            constexpr uint tilesH = H / BLOCK_N;
                            const uint rowTile = rCurrentTask.tileIdx / tilesH;
                            const uint colTile = rCurrentTask.tileIdx % tilesH;
                            const uint outRowOffset = rowTile * BLOCK_M;
                            const uint outColOffset = colTile * BLOCK_N;
                            printf("  OUTPUT grad_input[row=%u, col=%u..%u]: %.6e %.6e %.6e %.6e (tile=%u rowT=%u colT=%u)\n",
                                   outRowOffset, outColOffset, outColOffset + 3,
                                   static_cast<float>(output[outRowOffset * H + outColOffset + 0]),
                                   static_cast<float>(output[outRowOffset * H + outColOffset + 1]),
                                   static_cast<float>(output[outRowOffset * H + outColOffset + 2]),
                                   static_cast<float>(output[outRowOffset * H + outColOffset + 3]),
                                   rCurrentTask.tileIdx, rowTile, colTile);
                            // Full expected calculation for output[0, outColOffset]
                            // output[row, col] = sum_p (grad_a1[row, p] * act'(z1[row, p])) * W1[p, col]
                            if (outRowOffset == 0) {
                                float full_sum = 0.0f;
                                for (uint p = 0; p < P; ++p) {
                                    float grad_val = static_cast<float>(gradA1[0 * P + p]);
                                    float z1_val = static_cast<float>(z1Activation[0 * P + p]);
                                    float w1_val = static_cast<float>(W1[p * H + outColOffset]);
                                    // ReLU derivative
                                    float act_deriv = z1_val > 0.0f ? 1.0f : 0.0f;
                                    full_sum += grad_val * act_deriv * w1_val;
                                }
                                printf("  VERIFY: expected output[0,%u]=%.6e, actual=%.6e, diff=%.6e\n",
                                       outColOffset, full_sum,
                                       static_cast<float>(output[0 * H + outColOffset]),
                                       full_sum - static_cast<float>(output[0 * H + outColOffset]));
                            }
                        }
#endif
                        {
                            const auto peer = rCurrentTask.peerIdx;
                            const auto localExpertIdx = rCurrentTask.expertIdx;

                            // Get original input from heap (forward pass stored it at stage 0, cell 1)
                            const auto tokenOffset = rCurrentTask.batchIdx * BLOCK_M;
                            const auto* heapPtr = heap::advance<0, 1>(bookkeeping.sHeap, peer, localExpertIdx, tokenOffset);
                            const auto* originalInput = CONST_CAST_TO(typename PreGEMM::MatrixAType, heapPtr);

                            constexpr auto expertStride = 2 * P * H + P + H;
                            auto* const weightGradBuffer = CAST_TO(Element, bookkeeping.gW()) + localExpertIdx * expertStride;

                            // Bounds checks for originalInput heap pointer and computeWeightGradients inputs
                            {
                                // Check originalInput (heap pointer) is not null
                                if (!threadIdx.x && heapPtr == nullptr) {
                                    printf("ERROR gradPreGEMM heapPtr NULL: rank=%u block=%u tile=%u "
                                           "peer=%u expert=%u batchIdx=%u tokenOffset=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                           peer, localExpertIdx, rCurrentTask.batchIdx, tokenOffset);
                                }
                                // Check peer index is within world size
                                if (!threadIdx.x && peer >= bookkeeping.world) {
                                    printf("ERROR gradPreGEMM peer OOB: rank=%u block=%u tile=%u "
                                           "peer=%u >= world=%u expert=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                           peer, bookkeeping.world, localExpertIdx);
                                }
                                // Check localExpertIdx is within nLx
                                if (!threadIdx.x && localExpertIdx >= bookkeeping.nLx) {
                                    printf("ERROR gradPreGEMM expertIdx OOB: rank=%u block=%u tile=%u "
                                           "expert=%u >= nLx=%u peer=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                           localExpertIdx, bookkeeping.nLx, peer);
                                }
                                // Check tileSize is valid (> 0 and <= BLOCK_M)
                                if (!threadIdx.x && (rCurrentTask.tileSize == 0 || rCurrentTask.tileSize > BLOCK_M)) {
                                    printf("ERROR gradPreGEMM tileSize invalid: rank=%u block=%u tile=%u "
                                           "tileSize=%u BLOCK_M=%u expert=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                           rCurrentTask.tileSize, BLOCK_M, localExpertIdx);
                                }
                                // Check batchIdx won't cause tokenOffset overflow
                                const auto maxTokens = ACC::pEC::value;
                                if (!threadIdx.x && tokenOffset >= maxTokens) {
                                    printf("ERROR gradPreGEMM tokenOffset OOB: rank=%u block=%u tile=%u "
                                           "tokenOffset=%u >= pEC=%u batchIdx=%u expert=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                           tokenOffset, maxTokens, rCurrentTask.batchIdx, localExpertIdx);
                                }
                                // Check weightGradBuffer is not null
                                if (!threadIdx.x && weightGradBuffer == nullptr) {
                                    printf("ERROR gradPreGEMM weightGradBuffer NULL: rank=%u block=%u tile=%u "
                                           "gW=%p expert=%u stride=%u\n",
                                           nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                           bookkeeping.gW(), localExpertIdx, expertStride);
                                }
                            }

                            if (!threadIdx.x && blockIdx.x < 3) {
                                printf("DEBUG gradWeights: rank=%d block=%u peer=%u expert=%u batchIdx=%u tileSize=%u tokenOff=%u stride=%u heap=%p orig=%p aData=%p buf=%p\n",
                                       nvshmem_my_pe(), blockIdx.x, peer, localExpertIdx, rCurrentTask.batchIdx, rCurrentTask.tileSize,
                                       tokenOffset, expertStride, heapPtr, originalInput, rCurrentTask.aData, weightGradBuffer);
                            }
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 3) {
                                printf("DEBUG gradPreGEMM->gradWeights DIRECT: rank=%d block=%u peer=%u localExpert=%u tileSize=%u tileIdx=%u tokenOffset=%u gW=%p buffer=%p aData=%p origInput=%p\n",
                                       nvshmem_my_pe(), blockIdx.x, peer, localExpertIdx, rCurrentTask.tileSize, rCurrentTask.tileIdx, tokenOffset,
                                       bookkeeping.gW(), weightGradBuffer, rCurrentTask.aData, originalInput);
                            }
#endif
                            // dW1 = grad_intermediate^T @ original_input
                            // Shape: [P, H] = [P, M] @ [M, H]
                            // activations = grad_intermediate [M, P], gradients = original_input [M, H]
                            computeWeightGradients<P, H>(
                                workspace,
                                CONST_CAST_TO(typename PreGEMM::MatrixAType, rCurrentTask.aData),  // grad_intermediate [M, P]
                                originalInput,  // original input [M, H] from forward pass
                                weightGradBuffer,
                                rCurrentTask.tileSize,
                                static_cast<uint16_t>(localExpertIdx));
                            __syncthreads();
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 3) {
                                printf("DEBUG gradPreGEMM->gradWeights COMPLETE: rank=%d block=%u localExpert=%u buffer[0]=%.6f\n",
                                       nvshmem_my_pe(), blockIdx.x, localExpertIdx, static_cast<float>(weightGradBuffer[0]));
                            }
#endif
                        }
                        // gradPreGEMM is the last GEMM in backward chain, so send grad_input back
                        if (!threadIdx.x) {
                            const auto flagSignal = SignalPayload<PacketStage::last>{
                                rCurrentTask.batchIdx,
                                rCurrentTask.tileSize,
                                rSeqBit,
                            };
                            if (rCurrentTask.isPeerRemote) {
                                __threadfence();
                                if (atomicIncrement(pA.tQS + rCurrentTask.syncIdx) + 1 == tN + tNx) {
#if FLASHMOE_DEBUG
                                    printf("DEBUG gradPreGEMM NVSHMEM transfer block=%u syncIdx=%u tileSize=%u peer=%u\n",
                                           blockIdx.x, rCurrentTask.syncIdx, rCurrentTask.tileSize, rCurrentTask.peerIdx);
#endif
                                    nvshmem_putmem_signal_nbi(rCurrentTask.rcData,
                                        rCurrentTask.cData[w1Index],
                                        rCurrentTask.tileSize * H * sizeof(Element),
                                        rCurrentTask.flags,
                                        *CONST_CAST_TO(flagsType, &flagSignal), NVSHMEM_SIGNAL_SET,
                                        rCurrentTask.peerIdx);
                                }
                            }
                            else {
                                __threadfence_system();
                                atomicExch_system(CAST_TO(ull_t, rCurrentTask.flags),
                                    *CONST_CAST_TO(ull_t, &flagSignal));
                            }
                        }
                    }
                    break;
                    // This case is kept for reference but should never be reached since we no longer enqueue gradWeights tasks.
                    case TaskType::gradWeights: {
                        if (!threadIdx.x) {
                            printf("ERROR gradWeights TASK RECEIVED (should not happen - computed inline now): block=%u expert=%u\n",
                                   blockIdx.x, rCurrentTask.expertIdx);
                        }
                        computeWeightGradients<ACC::P::value, ACC::H::value>(
                            workspace,
                            CONST_CAST_TO(typename PreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.cData[0]),
                            CAST_TO(Element, rCurrentTask.rcData),
                            rCurrentTask.tileSize,
                            rCurrentTask.expertIdx);
                    }
                    break;
                    case TaskType::gradInputCombine: {
                        // accumulate grad_input from expert computation back to token positions
                        // no probability scaling needed - gradCombine already applied it
                        constexpr auto H = ACC::H::value;
                        constexpr auto S = ACC::S::value;
                        constexpr auto threads = ACC::PeakHardware::OS::threads::value;

                        const auto tileSize = rCurrentTask.tileSize;
                        // aData already points to this tile's TPS
                        const auto* __restrict__ tokenIds = CONST_CAST_TO(TPS, rCurrentTask.aData);

                        // Source: grad_input from xM (P2P: cData[0]) or heap (remote: cData[1])
                        const auto* __restrict__ gradInputSrc = CONST_CAST_TO(Element,
                            rCurrentTask.isPeerRemote ? rCurrentTask.cData[1] : rCurrentTask.cData[0]);

                        // Destination: global gradInputBasePtr
                        auto* __restrict__ gradInputDst = flashmoe::moe::gradInputBasePtr;

                        // Bounds checks for tokenIds, gradInputSrc, and tileSize
                        {
                            // Check tokenIds (aData) is within tP bounds
                            auto* tPBase = bookkeeping.tP();
                            auto* tPEnd = tPBase + ACC::E::value * ACC::pEC::value;
                            if (!threadIdx.x && (tokenIds < tPBase ||
                                reinterpret_cast<const cuda::std::byte*>(tokenIds + tileSize) >
                                reinterpret_cast<const cuda::std::byte*>(tPEnd))) {
                                printf("ERROR gradInputCombine tokenIds OOB: rank=%u block=%u tile=%u "
                                       "tokenIds=%p tileSize=%u tPBase=%p tPEnd=%p expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       tokenIds, tileSize, tPBase, tPEnd, rCurrentTask.expertIdx);
                            }
                            // Check gradInputSrc is not null
                            if (!threadIdx.x && gradInputSrc == nullptr) {
                                printf("ERROR gradInputCombine gradInputSrc NULL: rank=%u block=%u tile=%u "
                                       "isPeerRemote=%u cData[0]=%p cData[1]=%p expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       static_cast<unsigned>(rCurrentTask.isPeerRemote),
                                       rCurrentTask.cData[0], rCurrentTask.cData[1], rCurrentTask.expertIdx);
                            }
                            // Check gradInputDst is not null
                            if (!threadIdx.x && gradInputDst == nullptr) {
                                printf("ERROR gradInputCombine gradInputDst NULL: rank=%u block=%u tile=%u expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.expertIdx);
                            }
                            // Check tileSize is valid
                            if (!threadIdx.x && (tileSize == 0 || tileSize > BLOCK_M)) {
                                printf("ERROR gradInputCombine tileSize invalid: rank=%u block=%u tile=%u "
                                       "tileSize=%u BLOCK_M=%u expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx,
                                       tileSize, BLOCK_M, rCurrentTask.expertIdx);
                            }
                        }

                        using NativeElement = typename ToCDx<Element>::T;
                        constexpr auto convertNative = cutlass::NumericConverter<NativeElement, Element>{};

                        // Load TPS to shared memory for coalesced access
                        auto* __restrict__ sTPS = CAST_TO(TPS, workspace);
                        if (threadIdx.x < tileSize) {
                            sTPS[threadIdx.x] = tokenIds[threadIdx.x];
                        }
                        __syncthreads();

                        // Validate tokenIdx < S before scatter (check in shared memory after load)
                        {
                            __shared__ unsigned int sMaxTokenIdx;
                            __shared__ unsigned int sOOBCount;
                            if (!threadIdx.x) {
                                sMaxTokenIdx = 0;
                                sOOBCount = 0;
                            }
                            __syncthreads();
                            // Each thread checks its assigned tokens
                            if (threadIdx.x < tileSize) {
                                const auto myTokenIdx = sTPS[threadIdx.x].tokenIdx;
                                atomicMax(&sMaxTokenIdx, myTokenIdx);
                                if (myTokenIdx >= S) {
                                    atomicAdd(&sOOBCount, 1U);
                                }
                            }
                            __syncthreads();
                            if (!threadIdx.x && (sMaxTokenIdx >= S || sOOBCount > 0)) {
                                printf("ERROR gradInputCombine tokenIdx OOB: rank=%u block=%u tile=%u tileSize=%u "
                                       "maxTokenIdx=%u >= S=%u oobCount=%u expert=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tileSize,
                                       sMaxTokenIdx, S, sOOBCount, rCurrentTask.expertIdx);
                            }
                        }

                        // Parallel scatter-accumulate
                        for (uint t = 0; t < tileSize; ++t) {
                            const auto tokenIdx = sTPS[t].tokenIdx;
                            auto* __restrict__ const dstRow = gradInputDst + tokenIdx * H;
                            const auto* __restrict__ const srcRow = gradInputSrc + t * H;
                            for (uint h = threadIdx.x; h < H; h += threads) {
                                const auto srcVal = srcRow[h];
                                atomicAdd(CAST_TO(NativeElement, dstRow + h), convertNative(srcVal));
                            }
                        }
                        __syncthreads();
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            printf("DEBUG gradInputCombine rank=%d block=%u tile=%u tileSize=%u expert=%u peer=%u remote=%u tokenIdx[0]=%u src=%p dst=%p\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tileSize, rCurrentTask.expertIdx,
                                   rCurrentTask.peerIdx, static_cast<unsigned>(rCurrentTask.isPeerRemote),
                                   sTPS[0].tokenIdx, gradInputSrc, gradInputDst);
                        }
#endif
                    }
                    break;
                }
            }
        }
    }
}
#endif //FLASHMOE_COMPUTE_CUH

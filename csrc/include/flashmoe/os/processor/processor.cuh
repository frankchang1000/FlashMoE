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
                            atomicAdd(CAST_TO(CDxT, &gExpertGrad(phaseIdx + j * phases, cIdx)), rC(j + i * elems));
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        atomicAdd(CAST_TO(CDxT, &gExpertGrad(phaseIdx + j * phases, cIdx)), rC(j + i * elems));
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
        auto* __restrict__ tQ = CAST_TO(uint, pA.ptQ + (rCurrentTask.syncIdx * ACC::TNx::value));
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
        unsigned int gradIndex = 0>  // Which cData index to use as new task's aData
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

        const auto offset = ACC::TNx::value * rCurrentTask.batchIdx;
        const auto ptQOffset = rCurrentTask.syncIdx * ACC::TNx::value;
        auto* __restrict__ tQ = CAST_TO(uint, pA.ptQ + ptQOffset);
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
                const auto tileIdx = offset + taskIdx;
                const auto nextTask = Task {
                    TaskT,
                    rCurrentTask.cData[gradIndex],
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
                const auto tileIdx = offset + taskIdx;
                const auto nextTask = Task {
                    TaskT,
                    rCurrentTask.cData[gradIndex],
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
#if FLASHMOE_DEBUG
            const auto tQH_before = atomicAdd(pA.tQH + rCurrentTask.syncIdx, 0U);
#endif
            atomicAdd(pA.tQH + rCurrentTask.syncIdx, tasks);
#if FLASHMOE_DEBUG
            if (blockIdx.x < 2) {
                printf("DEBUG notifyGradientImpl DONE: block=%u TaskT=%u syncIdx=%u tQH_before=%u adding=%u tQH_after=%u\n",
                       blockIdx.x, static_cast<unsigned>(TaskT), rCurrentTask.syncIdx,
                       tQH_before, tasks, tQH_before + tasks);
            }
#endif
        }
    }

    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TNx::value
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
        notifyGradientImpl<TaskType::gradGateGEMM, p, tasks>(workspace, rCurrentTask, pA);
    }

    // gradPostGEMM output (grad_intermediate in cData[1]) becomes gradPreGEMM input
    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TN::value
    >
    __device__ __forceinline__
    void notifyGradPreGEMM(uint* __restrict__ const& workspace, const Task& rCurrentTask, const ProcessorArgs& pA) {
        // Use gradIndex=1 because gradPostGEMM output is in cData[1]
        notifyGradientImpl<TaskType::gradPreGEMM, p, tasks, 1>(workspace, rCurrentTask, pA);
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

#if FLASHMOE_DEBUG
        if (!threadIdx.x && blockIdx.x < 2) {
            printf("DEBUG computeWeightGradients ENTRY: block=%u expert=%u tileSize=%u K=%u N=%u bM=%u threads=%u\n",
                   blockIdx.x, expertIdx, tileSize, K, N, bM, threads);
            printf("DEBUG computeWeightGradients PTRS: act=%p grad=%p buffer=%p\n",
                   activations, gradients, weightGradBuffer);
            printf("DEBUG computeWeightGradients INPUT: act[0]=%.6f grad[0]=%.6f buffer[0]=%.6f\n",
                   static_cast<float>(activations[0]), static_cast<float>(gradients[0]),
                   static_cast<float>(weightGradBuffer[0]));
        }
#endif
        constexpr uint totalKN = K * N;
        constexpr uint elemsPerThread = (totalKN + threads - 1) / threads;

#if FLASHMOE_DEBUG
        if (!threadIdx.x && blockIdx.x < 2) {
            printf("DEBUG computeWeightGradients WORK: totalKN=%u elemsPerThread=%u\n",
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
            printf("DEBUG computeWeightGradients EXIT: block=%u expert=%u buffer[0]=%.6f firstSum=%.6f startIdx=%u endIdx=%u\n",
                   blockIdx.x, expertIdx, static_cast<float>(weightGradBuffer[0]),
                   static_cast<float>(debugSum), startIdx, endIdx);
        }
        if (threadIdx.x == threads - 1 && blockIdx.x < 2) {
            printf("DEBUG computeWeightGradients LAST_THREAD: block=%u thread=%u startIdx=%u endIdx=%u elemsComputed=%u\n",
                   blockIdx.x, threadIdx.x, startIdx, endIdx, endIdx - startIdx);
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
                    const auto taskTypeVal = static_cast<unsigned>(rCurrentTask.taskType);
                    if (taskTypeVal > 9) {
                        printf("DEBUG processor CORRUPT_TASK: block=%u type=%u (INVALID>9!) syncIdx=%u tileIdx=%u batchIdx=%u aData=%p\n",
                               blockIdx.x, taskTypeVal, rCurrentTask.syncIdx, rCurrentTask.tileIdx,
                               rCurrentTask.batchIdx, rCurrentTask.aData);
                    } else if (taskTypeVal >= 3) { // gradient tasks start at 3 (gradPreGEMM)
                        printf("DEBUG processor TASK: block=%u type=%u tile=%u syncIdx=%u batchIdx=%u\n",
                               blockIdx.x, taskTypeVal, rCurrentTask.tileIdx, rCurrentTask.syncIdx,
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
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            printf("DEBUG gradCombine ENTER: rank=%d block=%u tile=%u tileSize=%u expert=%u peer=%u remote=%u syncIdx=%u\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.tileSize, rCurrentTask.expertIdx,
                                   rCurrentTask.peerIdx, static_cast<unsigned>(rCurrentTask.isPeerRemote), rCurrentTask.syncIdx);
                        }
#endif
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
                            // tNx gradCombine + 1 gradGateCombine = tNx + 1 tasks per syncIdx
                            constexpr auto threshold = tNx + 1;
                            const auto countBefore = atomicAdd(pA.tQS + combineSyncIdx, 1U);
                            enqueue = countBefore + 1 == threshold;
#if FLASHMOE_DEBUG
                            if (blockIdx.x < 3) {
                                printf("DEBUG gradCombine SYNC: rank=%d block=%u syncIdx=%u combineSyncIdx=%u threshold=%u countBefore=%u countAfter=%u enqueue=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.syncIdx, combineSyncIdx, threshold, countBefore, countBefore + 1, enqueue ? 1U : 0U);
                            }
#endif
                        }
                        __syncthreads();
                        if (enqueue) {
                            rCurrentTask.cData[0] = CAST_TO(cuda::std::byte, bookkeeping.gGateCombine());
                            rCurrentTask.bData[0] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::hiddenStatesPtr);
                            rCurrentTask.bData[1] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::gateWeightsPtr);
                            rCurrentTask.dData[0] = CAST_TO(cuda::std::byte, flashmoe::moe::gradInputBasePtr);
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradCombine ENQUEUE: block=%u syncIdx=%u combineSyncIdx=%u remote=%u CALLING notifyGradient...\n",
                                       blockIdx.x, rCurrentTask.syncIdx, rCurrentTask.syncIdx + bookkeeping.gtQCl, rCurrentTask.isPeerRemote);
                            }
#endif
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
#if FLASHMOE_DEBUG
                                if (!threadIdx.x && blockIdx.x < 2) {
                                    printf("DEBUG gradCombine ENQUEUE: block=%u syncIdx=%u CALLING notifyGateGradient...\n",
                                           blockIdx.x, rCurrentTask.syncIdx);
                                }
#endif
                                notifyGateGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
#if FLASHMOE_DEBUG
                                if (!threadIdx.x && blockIdx.x < 2) {
                                    printf("DEBUG gradCombine ENQUEUE: block=%u syncIdx=%u CALLING notifyGateGradient...\n",
                                           blockIdx.x, rCurrentTask.syncIdx);
                                }
#endif
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
                        // Use global gradOutputBasePtr instead of cData[0] (which is packet, not full buffer)
                        const auto* __restrict__ const gradOut = flashmoe::moe::gradOutputBasePtr;
                        const auto mGradOut = make_tensor(cute::make_gmem_ptr(gradOut),
                            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                                cute::Stride<cute::Int<H>, cute::_1>>{});
                        // Use global gateBuffer - write directly to correct token positions using tokenIdx from TPS
                        auto* __restrict__ const gateBuffer = bookkeeping.gGateCombine();
                        auto* __restrict__ const tokenIds = CONST_CAST_TO(TPS, rCurrentTask.aData);
                        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
                        constexpr auto convertProb = cutlass::NumericConverter<Element, mp_t>{};
                        constexpr auto toCompute = cutlass::NumericConverter<ComputeElement, Element>{};
                        constexpr auto toElement = cutlass::NumericConverter<Element, ComputeElement>{};
                        using NativeElement = typename ToCDx<Element>::T;
                        constexpr auto convertNative = cutlass::NumericConverter<NativeElement, Element>{};
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            printf("DEBUG gradGateCombine ENTER: rank=%d block=%u tile=%u tileSize=%u expert=%u peer=%u remote=%u syncIdx=%u\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tileSize, rCurrentTask.expertIdx,
                                   rCurrentTask.peerIdx, static_cast<unsigned>(rCurrentTask.isPeerRemote), rCurrentTask.syncIdx);
                        }
#endif
                        // No per-tile zeroing needed - gateBuffer is pre-zeroed in bootstrap
                        // Write directly to global positions using tokenIdx from TPS array
                        for (uint idx = threadIdx.x; idx < tileSize; idx += threads) {
                            const auto tokenEntry = tokenIds[idx];
                            const auto tokenIdx = tokenEntry.tokenIdx;
                            if (tokenIdx >= S) {
// #if FLASHMOE_DEBUG
//                                 if (!threadIdx.x) {
//                                     printf("DEBUG gradGateCombine: block=%u SKIP tokenIdx=%u >= S=%u\n",
//                                            blockIdx.x, tokenIdx, S);
//                                 }
// #endif
                                continue;
                            }
                            ComputeElement gradSum = ComputeElement(0);
                            #pragma unroll
                            for (uint n = 0; n < H; ++n) {
                                gradSum = fmaf(toCompute(mGradOut(tokenIdx, n)), ComputeElement(1), gradSum);
                            }
                            const auto probVal = convertProb(tokenEntry.probability);
                            const auto routingVal = probVal * toElement(gradSum);
                            // Write to global position: gateBuffer[tokenIdx, expertIdx]
                            auto* const slot = gateBuffer + tokenIdx * E + rCurrentTask.expertIdx;
                            atomicAdd(CAST_TO(NativeElement, slot), convertNative(routingVal));
                        }
                        __syncthreads();
                        if (!threadIdx.x) {
                            // Pass global gateBuffer base to gradGateGEMM (not tile-offset)
                            rCurrentTask.cData[gradIndex] = CAST_TO(cuda::std::byte, gateBuffer);
                            rCurrentTask.bData[0] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::hiddenStatesPtr);
                            rCurrentTask.bData[1] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::gateWeightsPtr);
                            // Set dData[0] for gradGateGEMM to write input gradients
                            rCurrentTask.dData[0] = CAST_TO(cuda::std::byte, flashmoe::moe::gradInputBasePtr);
                            __threadfence();
                            // FIX: Use offset syncIdx for tQS access to avoid collision with gradPostGEMM
                            // gradPostGEMM uses tQS[0, gtQCl), combine tasks use tQS[gtQCl, 2*gtQCl)
                            const auto combineSyncIdx = rCurrentTask.syncIdx + bookkeeping.gtQCl;
                            // tNx gradCombine + 1 gradGateCombine = tNx + 1 tasks per syncIdx
                            constexpr auto threshold = tNx + 1;
                            const auto countBefore = atomicAdd(pA.tQS + combineSyncIdx, 1U);
                            enqueue = countBefore + 1 == threshold;
#if FLASHMOE_DEBUG
                            if (blockIdx.x < 3) {
                                printf("DEBUG gradGateCombine SYNC: rank=%d block=%u syncIdx=%u combineSyncIdx=%u threshold=%u countBefore=%u countAfter=%u enqueue=%u\n",
                                       nvshmem_my_pe(), blockIdx.x, rCurrentTask.syncIdx, combineSyncIdx, threshold, countBefore, countBefore + 1, enqueue ? 1U : 0U);
                            }
#endif
                        }
                        __syncthreads();
                        if (enqueue) {
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradGateCombine ENQUEUE: block=%u syncIdx=%u combineSyncIdx=%u remote=%u CALLING notifyGradient...\n",
                                       blockIdx.x, rCurrentTask.syncIdx, rCurrentTask.syncIdx + bookkeeping.gtQCl, rCurrentTask.isPeerRemote);
                            }
#endif
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
#if FLASHMOE_DEBUG
                                if (!threadIdx.x && blockIdx.x < 2) {
                                    printf("DEBUG gradGateCombine ENQUEUE: block=%u syncIdx=%u CALLING notifyGateGradient...\n",
                                           blockIdx.x, rCurrentTask.syncIdx);
                                }
#endif
                                notifyGateGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
#if FLASHMOE_DEBUG
                                if (!threadIdx.x && blockIdx.x < 2) {
                                    printf("DEBUG gradGateCombine ENQUEUE: block=%u syncIdx=%u CALLING notifyGateGradient...\n",
                                           blockIdx.x, rCurrentTask.syncIdx);
                                }
#endif
                                notifyGateGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 2) {
                                printf("DEBUG gradGateCombine ENQUEUE: block=%u syncIdx=%u DONE both notifications\n",
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
                            printf("DEBUG gradGateGEMM ENTER: rank=%d block=%u tile=%u tileSize=%u expert=%u peer=%u remote=%u syncIdx=%u\n",
                                   nvshmem_my_pe(), blockIdx.x, rCurrentTask.tileIdx, tileSize, rCurrentTask.expertIdx,
                                   rCurrentTask.peerIdx, static_cast<unsigned>(rCurrentTask.isPeerRemote), rCurrentTask.syncIdx);
                            printf("DEBUG gradGateGEMM PTRS: tokenIds=%p hiddenStates=%p gateWeights=%p gradRouting=%p gradInput=%p routingScores=%p gateWeightGrad=%p\n",
                                   tokenIds, hiddenStates, gateWeights, gradRouting, gradInput, routingScores, gateWeightGrad);
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
                    }
                    break;
                    case TaskType::gradPostGEMM: {
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            printf("DEBUG gradPostGEMM CASE ENTERED: block=%u tile=%u aData=%p\n",
                                   blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.aData);
                        }
#endif
                        constexpr unsigned int w2Index = 1;
                        constexpr unsigned int w1Index = 0;
                        const Element* z2Activation = nullptr;
                        long long xMOffset2 = 0;
                        long long rowIdx2 = 0;
                        {
                            auto* xMBase = CAST_TO(Element, bookkeeping.xM());
                            auto* xMPtr = CAST_TO(Element, rCurrentTask.cData[w1Index]);
                            // Compute row index in xM (which has P columns)
                            xMOffset2 = xMPtr - xMBase;
                            rowIdx2 = xMOffset2 / P;
                            // z2 has H columns, so offset is rowIdx * H
                            z2Activation = bookkeeping.z2() + rowIdx2 * H;
                        }
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            auto* z2Base = bookkeeping.z2();
                            printf("DEBUG gradPostGEMM ENTER block=%u tile=%u M=%u: xMOffset=%lld rowIdx=%lld z2Base=%p z2Act=%p cData[0]=%p\n",
                                   blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.M,
                                   xMOffset2, rowIdx2, z2Base, z2Activation, rCurrentTask.cData[w1Index]);
                        }
#endif
                        fGET<GradPostGEMM, ACC::H::value, ACC::P::value>(
                            CAST_TO(typename GradPostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename GradPostGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename GradPostGEMM::MatrixBType, rCurrentTask.bData[w2Index]),
                            CAST_TO(typename GradPostGEMM::MatrixDType, rCurrentTask.cData[w2Index]),
                            CONST_CAST_TO(typename GradPostGEMM::MatrixDType, z2Activation),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
                            enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) + 1 == tN;
#if FLASHMOE_DEBUG
                            if (blockIdx.x < 3) {
                                printf("DEBUG gradPostGEMM block=%u syncIdx=%u tN=%u enqueue=%u\n",
                                       blockIdx.x, rCurrentTask.syncIdx, tN, enqueue ? 1U : 0U);
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
                        constexpr unsigned int w1Index = 0;
                        // Compute z1 offset for saved activation (same layout as forward preGEMM)
                        // z1 was stored at same offset as cData[0] relative to xM base
                        const Element* z1Activation = nullptr;
                        long long xMOffset = 0;
                        {
                            auto* xMBase = CAST_TO(Element, bookkeeping.xM());
                            auto* cDataPtr = CAST_TO(Element, rCurrentTask.cData[w1Index]);
                            xMOffset = cDataPtr - xMBase;
                            z1Activation = bookkeeping.z1() + xMOffset;
                        }
#if FLASHMOE_DEBUG
                        if (!threadIdx.x && blockIdx.x < 3) {
                            auto* z1Base = bookkeeping.z1();
                            printf("DEBUG gradPreGEMM block=%u tile=%u M=%u: xMOffset=%lld z1Base=%p z1Act=%p aData=%p cData=%p\n",
                                   blockIdx.x, rCurrentTask.tileIdx, rCurrentTask.M,
                                   xMOffset, z1Base, z1Activation,
                                   rCurrentTask.aData, rCurrentTask.cData[w1Index]);
                        }
#endif
                        fGET<GradPreGEMM, ACC::P::value, ACC::H::value>(
                            CAST_TO(typename GradPreGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename GradPreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename GradPreGEMM::MatrixBType, rCurrentTask.bData[w1Index]),
                            CAST_TO(typename GradPreGEMM::MatrixDType, rCurrentTask.cData[w1Index]),
                            CONST_CAST_TO(typename GradPreGEMM::MatrixDType, z1Activation),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
                        {
                            const auto peer = rCurrentTask.peerIdx;
                            const auto localExpertIdx = rCurrentTask.expertIdx;

                            // DEBUG: Verify task fields
                            if (!threadIdx.x && blockIdx.x < 3) {
                                printf("DEBUG gradWeights TASK: block=%u peer=%u expert=%u batchIdx=%u tileSize=%u sHeap=%p gW=%p\n",
                                       blockIdx.x, peer, localExpertIdx, rCurrentTask.batchIdx, rCurrentTask.tileSize,
                                       bookkeeping.sHeap, bookkeeping.gW());
                            }

                            // Get original input from heap (forward pass stored it at stage 0, cell 1)
                            const auto tokenOffset = rCurrentTask.batchIdx * BLOCK_M;
                            const auto* heapPtr = heap::advance<0, 1>(bookkeeping.sHeap, peer, localExpertIdx, tokenOffset);
                            const auto* originalInput = CONST_CAST_TO(typename PreGEMM::MatrixAType, heapPtr);

                            if (!threadIdx.x && blockIdx.x < 3) {
                                printf("DEBUG gradWeights PTRS: block=%u tokenOffset=%u heapPtr=%p origInput=%p aData=%p\n",
                                       blockIdx.x, tokenOffset, heapPtr, originalInput, rCurrentTask.aData);
                                // const auto* heapCell0 = heap::advance<0, 0>(bookkeeping.sHeap, peer, localExpertIdx, tokenOffset);
                                // const auto* heapCell1 = heapPtr; // already cell 1
                                // const auto* val0 = CONST_CAST_TO(Element, heapCell0);
                                // const auto* val1 = CONST_CAST_TO(Element, heapCell1);
                                // printf("DEBUG BWD_HEAP_COMPARE: block=%u peer=%u expert=%u token=%u cell0=%p val0[0]=%.6f cell1=%p val1[0]=%.6f\n",
                                //        blockIdx.x, peer, localExpertIdx, tokenOffset, heapCell0, static_cast<float>(val0[0]), heapCell1, static_cast<float>(val1[0]));
                            }

                            constexpr auto expertStride = 2 * P * H + P + H;
                            auto* const weightGradBuffer = CAST_TO(Element, bookkeeping.gW()) + localExpertIdx * expertStride;

                            if (!threadIdx.x && blockIdx.x < 3) {
                                printf("DEBUG gradWeights BUF: block=%u expertStride=%u buffer=%p\n",
                                       blockIdx.x, expertStride, weightGradBuffer);
                            }
#if FLASHMOE_DEBUG
                            if (!threadIdx.x && blockIdx.x < 3) {
                                printf("DEBUG gradPreGEMM->gradWeights DIRECT: block=%u peer=%u localExpert=%u tileSize=%u tileIdx=%u tokenOffset=%u gW=%p buffer=%p aData=%p origInput=%p\n",
                                       blockIdx.x, peer, localExpertIdx, rCurrentTask.tileSize, rCurrentTask.tileIdx, tokenOffset,
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
                                printf("DEBUG gradPreGEMM->gradWeights COMPLETE: block=%u localExpert=%u buffer[0]=%.6f\n",
                                       blockIdx.x, localExpertIdx, static_cast<float>(weightGradBuffer[0]));
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
                        constexpr auto threads = ACC::PeakHardware::OS::threads::value;

                        const auto tileSize = rCurrentTask.tileSize;
                        // aData already points to this tile's TPS
                        const auto* __restrict__ tokenIds = CONST_CAST_TO(TPS, rCurrentTask.aData);

                        // Source: grad_input from xM (P2P: cData[0]) or heap (remote: cData[1])
                        const auto* __restrict__ gradInputSrc = CONST_CAST_TO(Element,
                            rCurrentTask.isPeerRemote ? rCurrentTask.cData[1] : rCurrentTask.cData[0]);

                        // Destination: global gradInputBasePtr
                        auto* __restrict__ gradInputDst = flashmoe::moe::gradInputBasePtr;

                        using NativeElement = typename ToCDx<Element>::T;
                        constexpr auto convertNative = cutlass::NumericConverter<NativeElement, Element>{};

                        // Load TPS to shared memory for coalesced access
                        auto* __restrict__ sTPS = CAST_TO(TPS, workspace);
                        if (threadIdx.x < tileSize) {
                            sTPS[threadIdx.x] = tokenIds[threadIdx.x];
                        }
                        __syncthreads();

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

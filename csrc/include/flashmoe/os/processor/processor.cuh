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
        sTPS[threadIdx.x] = tokenIndices[threadIdx.x];
        __syncthreads();
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
        sTPS[threadIdx.x] = tokenIndices[threadIdx.x];
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
        const unsigned int& tileIdx) {
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
        // prefetch bias from global memory
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
        unsigned int tasks = ACC::TNx::value>
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
        constexpr unsigned int gradIndex = 0;

        const auto offset = ACC::TNx::value * rCurrentTask.batchIdx;
        auto* __restrict__ tQ = CAST_TO(uint, pA.ptQ + (rCurrentTask.syncIdx * ACC::TNx::value));
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
            atomicAdd(pA.tQH + rCurrentTask.syncIdx, tasks);
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

    __device__ __forceinline__
    void enqueueTask(const Task& task, const ProcessorArgs& pA) {
        static_assert(sizeof(Task) == 128);
        constexpr auto eS = sizeof(Task) / sizeof(uint);
        static_assert(eS == WARP_SIZE);
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        constexpr auto sharedSize = ACC::sharedSize::value;
        static_assert(sharedSize % sizeof(Task) == 0);
        static_assert(sharedSize / sizeof(Task) >= threads);

        __shared__ uint sTask[eS];
        if (threadIdx.x < eS) {
            const auto* __restrict__ uT = CONST_CAST_TO(uint, &task);
            sTask[threadIdx.x] = uT[threadIdx.x];
        }
        __syncthreads();

        if (!threadIdx.x) {
            const auto taskIdx = atomicAdd(pA.tQH + task.syncIdx, 1U);
            auto* __restrict__ tQ = CAST_TO(uint, pA.tQ + taskIdx);
            #pragma unroll
            for (uint i = 0; i < eS; ++i) {
                tQ[i] = sTask[i];
            }
            __threadfence();
        }
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

        #pragma unroll
        for (uint m = 0; m < bM; ++m) {
            if (m < tileSize) {
                #pragma unroll
                for (uint k = 0; k < K; ++k) {
                    const auto actVal = mAct(m, k);
                    #pragma unroll
                    for (uint n = 0; n < N; ++n) {
                        const auto gradVal = mGrad(m, n);
                        using ProdT = decltype(actVal * gradVal);
                        using NativeElement = typename ToCDx<Element>::T;
                        constexpr auto convert = cutlass::NumericConverter<NativeElement, ProdT>{};
                        auto* nativePtr = CAST_TO(NativeElement, &mWGrad(k, n));
                        atomicAdd(nativePtr, convert(actVal * gradVal));
                    }
                }
            }
        }
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
                switch (rCurrentTask.taskType) {
                    case TaskType::preGEMM: {
                        constexpr unsigned int preIndex = 0;
                        fGET<PreGEMM, ACC::P::value, ACC::H::value>(
                            CAST_TO(typename PreGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename PreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PreGEMM::MatrixBType, rCurrentTask.bData[preIndex]),
                            CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.cData[preIndex]),
                            CONST_CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.dData[preIndex]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
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
                        fGET<PostGEMM, ACC::H::value, ACC::P::value>(
                            CAST_TO(typename PostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixBType, rCurrentTask.bData[postIndex]),
                            CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.cData[postIndex]),
                            CONST_CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.dData[postIndex]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);  // was prev currentTask.tileIdx. look into this
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
                                if (atomicIncrement(pA.tQS + rCurrentTask.syncIdx) + 1 == tN + tNx) {
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
// #if FLASHMOE_DEBUG
//                         if (!threadIdx.x && !blockIdx.x) {
//                             printf("DEBUG combine complete rank=%d sb=%u tile=%u tileSize=%u expert=%u peer=%u remote=%u\n",
//                                    nvshmem_my_pe(),
//                                    rSeqBit,
//                                    rCurrentTask.tileIdx,
//                                    rCurrentTask.tileSize,
//                                    rCurrentTask.expertIdx,
//                                    rCurrentTask.peerIdx,
//                                    static_cast<unsigned>(rCurrentTask.isPeerRemote));
//                         }
// #endif
                    }
                    break;
                    case TaskType::gradCombine: {
                        constexpr unsigned int gradIndex = 0;
                        splitGradients<ACC::CM::value>(
                            workspace,
                            CONST_CAST_TO(TPS, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.bData[gradIndex]),
                            CAST_TO(Element, rCurrentTask.cData[gradIndex]),
                            sW,
                            rCurrentTask.tileIdx,
                            rCurrentTask.tileSize,
                            rCurrentTask.expertIdx);
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
                            enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) + 1 == tNx;
                        }
                        __syncthreads();
                        if (enqueue) {
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                        }
                    }
                    break;
                    case TaskType::gradGateCombine: {
                        constexpr unsigned int gradIndex = 0;
                        using ComputeElement = ACC::ElementC;
                        constexpr auto H = ACC::H::value;
                        constexpr auto E = ACC::E::value;
                        constexpr auto S = ACC::S::value;
                        constexpr auto blockTokens = BLOCK_M;
                        const auto tileSize = rCurrentTask.tileSize;
                        const auto baseToken = static_cast<unsigned int>(rCurrentTask.tileIdx) * blockTokens;
                        const auto safeBaseToken = baseToken < S ? baseToken :
                            (S > blockTokens ? S - blockTokens : 0U);
                        const auto* __restrict__ const gradOut = CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.cData[gradIndex]);
                        const auto mGradOut = make_tensor(cute::make_gmem_ptr(gradOut),
                            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                                cute::Stride<cute::Int<H>, cute::_1>>{});
                        auto* __restrict__ const gateBuffer = bookkeeping.gGateCombine();
                        auto* __restrict__ const tokenIds = CONST_CAST_TO(TPS, rCurrentTask.aData);
                        auto* __restrict__ const tileGateBuffer = gateBuffer + safeBaseToken * E;
                        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
                        constexpr auto convertProb = cutlass::NumericConverter<Element, mp_t>{};
                        constexpr auto toCompute = cutlass::NumericConverter<ComputeElement, Element>{};
                        constexpr auto toElement = cutlass::NumericConverter<Element, ComputeElement>{};
                        using NativeElement = typename ToCDx<Element>::T;
                        constexpr auto convertNative = cutlass::NumericConverter<NativeElement, Element>{};
                        if (!threadIdx.x) {
                            const auto tileEntries = static_cast<unsigned int>(tileSize) * E;
                            #pragma unroll 4
                            for (uint i = 0; i < tileEntries; ++i) {
                                tileGateBuffer[i] = Element(0);
                            }
                        }
                        __syncthreads();

                        for (uint idx = threadIdx.x; idx < tileSize; idx += threads) {
                            const auto tokenEntry = tokenIds[idx];
                            const auto tokenIdx = tokenEntry.tokenIdx;
                            if (tokenIdx >= S) {
                                continue;
                            }
                            ComputeElement gradSum = ComputeElement(0);
                            #pragma unroll
                            for (uint n = 0; n < H; ++n) {
                                gradSum = fmaf(toCompute(mGradOut(tokenIdx, n)), ComputeElement(1), gradSum);
                            }
                            const auto probVal = convertProb(tokenEntry.probability);
                            const auto routingVal = probVal * toElement(gradSum);
                            auto* const slot = tileGateBuffer + idx * E + rCurrentTask.expertIdx;
                            atomicAdd(CAST_TO(NativeElement, slot), convertNative(routingVal));
                        }
                        __syncthreads();
                        if (!threadIdx.x) {
                            rCurrentTask.cData[gradIndex] = CAST_TO(cuda::std::byte, tileGateBuffer);
                            rCurrentTask.bData[0] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::hiddenStatesPtr);
                            rCurrentTask.bData[1] = CONST_CAST_TO(cuda::std::byte, flashmoe::moe::gateWeightsPtr);
                            __threadfence();
                            enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) + 1 == tNx;
                        }
                        __syncthreads();
                        if (enqueue) {
                            if (!rCurrentTask.isPeerRemote) {
                                notifyGateGradient<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyGateGradient<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                        }
                    }
                    break;
                    case TaskType::gradGateGEMM: {
                        constexpr auto H = ACC::H::value;
                        constexpr auto E = ACC::E::value;
                        constexpr auto S = ACC::S::value;
                        constexpr auto blockTokens = BLOCK_M;
                        const auto tileSize = rCurrentTask.tileSize;
                        const auto baseToken = static_cast<unsigned int>(rCurrentTask.tileIdx) * blockTokens;
                        const auto safeBaseToken = baseToken < S ? baseToken :
                            (S > blockTokens ? S - blockTokens : 0U);
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

                        for (uint t = threadIdx.x; t < tileSize; t += threads) {
                            const auto tokenIdx = safeBaseToken + t;
                            if (tokenIdx >= S) {
                                continue;
                            }
                            const auto* __restrict__ const gradRoutingRow = gradRouting + t * E;
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
                            const auto* __restrict__ const hiddenRow = hiddenStates + t * H;
                            auto* __restrict__ const inputRow = gradInput + t * H;
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
                        constexpr unsigned int w2Index = 1;
                        fGET<GradPostGEMM, ACC::H::value, ACC::P::value>(
                            CAST_TO(typename GradPostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename GradPostGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename GradPostGEMM::MatrixBType, rCurrentTask.bData[w2Index]),
                            CAST_TO(typename GradPostGEMM::MatrixDType, rCurrentTask.cData[w2Index]),
                            CONST_CAST_TO(typename GradPostGEMM::MatrixDType, rCurrentTask.dData[w2Index]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            const auto flagSignal = SignalPayload<PacketStage::last>{
                                rCurrentTask.batchIdx,
                                rCurrentTask.tileSize,
                                rSeqBit,
                            };
                            if (rCurrentTask.isPeerRemote) {
                                __threadfence();
                                if (atomicIncrement(pA.tQS + rCurrentTask.syncIdx) + 1 == tN + tNx) {
                                    nvshmem_putmem_signal_nbi(rCurrentTask.rcData,
                                        rCurrentTask.cData[w2Index],
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
                    case TaskType::gradPreGEMM: {
                        constexpr unsigned int w1Index = 0;
                        fGET<GradPreGEMM, ACC::P::value, ACC::H::value>(
                            CAST_TO(typename GradPreGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename GradPreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename GradPreGEMM::MatrixBType, rCurrentTask.bData[w1Index]),
                            CAST_TO(typename GradPreGEMM::MatrixDType, rCurrentTask.cData[w1Index]),
                            CONST_CAST_TO(typename GradPreGEMM::MatrixDType, rCurrentTask.dData[w1Index]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            constexpr auto expertStride = 2 * P * H + P + H;
                            auto* const weightGradBuffer = CAST_TO(Element, bookkeeping.gW()) + rCurrentTask.expertIdx * expertStride;
                            const auto gradWeightTask = Task{
                                TaskType::gradWeights,
                                rCurrentTask.aData,
                                rCurrentTask.bData,
                                rCurrentTask.cData,
                                rCurrentTask.dData,
                                CAST_TO(cuda::std::byte, weightGradBuffer),
                                rCurrentTask.flags,
                                rCurrentTask.syncIdx,
                                rCurrentTask.tileIdx,
                                rCurrentTask.M,
                                rCurrentTask.tileSize,
                                rCurrentTask.peerIdx,
                                rCurrentTask.batchIdx,
                                rCurrentTask.isPeerRemote,
                            };
                            enqueueTask(gradWeightTask, pA);
                        }
                    }
                    break;
                    case TaskType::gradWeights: {
                        computeWeightGradients<ACC::P::value, ACC::H::value>(
                            workspace,
                            CONST_CAST_TO(typename PreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.cData[0]),
                            CAST_TO(Element, rCurrentTask.rcData),
                            rCurrentTask.tileSize,
                            rCurrentTask.expertIdx);
                    }
                    break;
                }
            }
        }
    }
}
#endif //FLASHMOE_COMPUTE_CUH

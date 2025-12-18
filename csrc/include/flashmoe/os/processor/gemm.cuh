/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */
//
// Created by oja7 on 11/25/24.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_mma.hpp>

#include "../../types.cuh"
#include "mmaConfig.cuh"
#include "../../arch.cuh"

namespace flashmoe {
    /// Fused, Add, Activate
    template <typename Element, typename ActivationFunction>
    requires(flashmoe::TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
    struct FAA {
        __forceinline__ __device__
        Element operator()(const Element& accumulator, const Element& term) const {
            constexpr ActivationFunction op{};
            return op(accumulator + term);
        }
    };

    // specialization for half-precision and relu
    template<>
    struct FAA<cute::half_t, cutlass::epilogue::thread::ReLU<cute::half_t>> {
        __forceinline__ __device__
        cute::half_t operator()(const cute::half_t& accumulator, const cute::half_t& term) const {
            return cute::half_t(__hfma_relu(__float2half(1.0f), accumulator.to_half(), term.to_half()));
        }
    };

    // specialization for bfloat16 and relu
    template<>
    struct FAA<cute::bfloat16_t, cutlass::epilogue::thread::ReLU<cute::bfloat16_t>> {
        __forceinline__ __device__
        cute::bfloat16_t operator()(const cute::bfloat16_t& accumulator, const cute::bfloat16_t& term) const {
            // TODO(byungsoo): Further validate this code.
            // See https://github.com/osayamenja/FlashMoE/issues/6
            // Note: This issue also occurs on NVIDIA A100 GPU.
            float acc_f = __bfloat162float(accumulator.to_nv_bfloat16());
            float term_f = __bfloat162float(term.to_nv_bfloat16());
            float result = acc_f * 1.0f + term_f;
            result = result > 0.0f ? result : 0.0f;  // ReLU
            return cute::bfloat16_t(__float2bfloat16(result));
        }
    };

    template<typename F>
    struct isFAA : cuda::std::false_type {};

    template<typename Element, typename ActivationFunction>
    struct isFAA<FAA<Element, ActivationFunction>> : cuda::std::true_type {};

    template <typename Element, typename ActivationFunction>
    requires(flashmoe::TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
    struct ActivationDerivative;

    template<typename F>
    struct isActivationDerivative : cuda::std::false_type {};

    template<typename Element, typename ActivationFunction>
    struct isActivationDerivative<ActivationDerivative<Element, ActivationFunction>> : cuda::std::true_type {};

    template<typename F>
    struct extract_activation_function {};

    template<typename Element, typename ActivationFunction>
    struct extract_activation_function<ActivationDerivative<Element, ActivationFunction>> {
        using type = ActivationFunction;
    };

    template<bool isDerivative, typename ActivationOp, typename ElementC>
    struct activation_epilogue_helper;

    template<typename ActivationOp, typename ElementC>
    struct activation_epilogue_helper<true, ActivationOp, ElementC> {
        using type = ActivationOp;
    };

    template<typename ActivationOp, typename ElementC>
    struct activation_epilogue_helper<false, ActivationOp, ElementC> {
        using type = FAA<ElementC, ActivationOp>;
    };

    /// activation deriv: (matmul_result) ⊙ activation'(saved_forward)
    template <typename Element, typename ActivationFunction>
    requires(flashmoe::TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
    struct ActivationDerivative {
        __forceinline__ __device__
        Element operator()(const Element& matmul_result, const Element& saved_forward) const {
            constexpr ActivationFunction op{};
            return matmul_result * derivative(op, saved_forward);
        }

    private:
        template<typename Op>
        __forceinline__ __device__
        static Element derivative(const Op&, const Element& x) {
            if constexpr (cuda::std::is_same_v<Op, cutlass::epilogue::thread::ReLU<Element>>) {
                return x > Element(0) ? Element(1) : Element(0);
            } else if constexpr (cuda::std::is_same_v<Op, cutlass::epilogue::thread::GELU<Element>>) {
                constexpr float sqrt_2_over_pi = 0.7978845608f;
                constexpr float coeff = 0.044715f;
                float x_f = static_cast<float>(x);
                float x_cubed = x_f * x_f * x_f;
                float inner = sqrt_2_over_pi * (x_f + coeff * x_cubed);
                float tanh_val = tanhf(inner);
                float result = 0.5f * (1.0f + tanh_val) + 0.5f * x_f * (1.0f - tanh_val * tanh_val) * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_f * x_f);
                return static_cast<Element>(result);
            } else if constexpr (cuda::std::is_same_v<Op, cute::identity>) {
                return Element(1);
            }
        }
    };

    template <typename Element>
    struct ActivationDerivative<Element, cute::identity> {
        __forceinline__ __device__
        Element operator()(const Element& matmul_result, const Element&) const {
            return matmul_result;
        }
    };

    template<>
    struct ActivationDerivative<cute::half_t, cutlass::epilogue::thread::ReLU<cute::half_t>> {
        __forceinline__ __device__
        cute::half_t operator()(const cute::half_t& matmul_result, const cute::half_t& saved_forward) const {
            float grad_f = __half2float(matmul_result.to_half());
            float forward_f = __half2float(saved_forward.to_half());
            float deriv = forward_f > 0.0f ? 1.0f : 0.0f;
            return cute::half_t(__float2half(grad_f * deriv));
        }
    };

    template<>
    struct ActivationDerivative<cute::bfloat16_t, cutlass::epilogue::thread::ReLU<cute::bfloat16_t>> {
        __forceinline__ __device__
        cute::bfloat16_t operator()(const cute::bfloat16_t& matmul_result, const cute::bfloat16_t& saved_forward) const {
            float grad_f = __bfloat162float(matmul_result.to_nv_bfloat16());
            float forward_f = __bfloat162float(saved_forward.to_nv_bfloat16());
            float deriv = forward_f > 0.0f ? 1.0f : 0.0f;
            return cute::bfloat16_t(__float2bfloat16(grad_f * deriv));
        }
    };

    template<>
    struct ActivationDerivative<cute::half_t, cutlass::epilogue::thread::GELU<cute::half_t>> {
        __forceinline__ __device__
        cute::half_t operator()(const cute::half_t& matmul_result, const cute::half_t& saved_forward) const {
            constexpr float sqrt_2_over_pi = 0.7978845608f;
            constexpr float coeff = 0.044715f;
            float grad_f = __half2float(matmul_result.to_half());
            float x_f = __half2float(saved_forward.to_half());
            float x_cubed = x_f * x_f * x_f;
            float inner = sqrt_2_over_pi * (x_f + coeff * x_cubed);
            float tanh_val = tanhf(inner);
            float deriv = 0.5f * (1.0f + tanh_val) + 0.5f * x_f * (1.0f - tanh_val * tanh_val) * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_f * x_f);
            return cute::half_t(__float2half(grad_f * deriv));
        }
    };

    template<>
    struct ActivationDerivative<cute::bfloat16_t, cutlass::epilogue::thread::GELU<cute::bfloat16_t>> {
        __forceinline__ __device__
        cute::bfloat16_t operator()(const cute::bfloat16_t& matmul_result, const cute::bfloat16_t& saved_forward) const {
            constexpr float sqrt_2_over_pi = 0.7978845608f;
            constexpr float coeff = 0.044715f;
            float grad_f = __bfloat162float(matmul_result.to_nv_bfloat16());
            float x_f = __bfloat162float(saved_forward.to_nv_bfloat16());
            float x_cubed = x_f * x_f * x_f;
            float inner = sqrt_2_over_pi * (x_f + coeff * x_cubed);
            float tanh_val = tanhf(inner);
            float deriv = 0.5f * (1.0f + tanh_val) + 0.5f * x_f * (1.0f - tanh_val * tanh_val) * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_f * x_f);
            return cute::bfloat16_t(__float2bfloat16(grad_f * deriv));
        }
    };

    template<
        typename ActivationOp,
        typename ElementA,
        typename ElementB = ElementA,
        typename ElementC = ACC::ElementC,
        unsigned int sizeK = ACC::PeakHardware::bKBase::value,
        unsigned int Arch = cute::min(ACC::PeakHardware::arch::value,800), // clamp at 800 for now
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        unsigned int pipeStages = ACC::PeakHardware::pipeStages::value
    >
    requires(cuda::std::is_same_v<ElementC, ACC::ElementC> ||
        (cuda::std::is_same_v<ElementC, cute::half_t> &&
            cuda::std::is_same_v<ElementA, cute::half_t> &&
            cuda::std::is_same_v<ElementB, cute::half_t>))
    struct BlockMM {
        // will clamp at Ampere for now, until we implement Hopper specific GEMM
        static_assert(BLOCK_M == THREADS && BLOCK_M == threads);
        static_assert(BLOCK_M == 128);
        static_assert(BLOCK_N == 64, "64 is a very good value for N, change it back!");
        using Threads = cute::C<threads>;
        using MatrixAType = ElementA;
        using MatrixBType = ElementB;
        using MatrixCType = ElementC;
        using MatrixDType = ElementA;
        using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<sizeK>>;
        using TilerOut = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
        using Parameters = CollectiveMMAConfig<BLOCK_M, BLOCK_N, sizeK, Arch, ElementA, ElementB, ElementC,
            LayoutOptimization::UseSwizzle>;
        using MMA = typename Parameters::mma_t;
        using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
            cuda::std::conditional_t<Arch < 800,
                    cutlass::gemm::MainloopSm70TwoStageUnpredicated,
                        cutlass::gemm::MainloopSm80CpAsyncUnpredicated<pipeStages>>,
            BlockTiler,
            ElementA,
            cute::Underscore,
            ElementB,
            cute::Underscore,
            typename Parameters::mma_t,
            typename Parameters::gCopyA,
            typename Parameters::sLayA,
            typename Parameters::sCopyA,
            cute::identity,
            typename Parameters::gCopyB,
            typename Parameters::sLayB,
            typename Parameters::sCopyB,
            cute::identity
        >;
        using FusedEpilogue = typename activation_epilogue_helper<
            isActivationDerivative<ActivationOp>::value,
            ActivationOp,
            ElementC
        >::type;
    };
}
#endif //GEMM_CUH

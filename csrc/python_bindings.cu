/**
 * Python bindings for FlashMoE CUDA kernels
 * This wraps FlashMoE CUDA code to be callable from Python
 */
#include <cuda/std/array>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include "include/flashmoe/bootstrap.cuh"
#include "include/flashmoe/moe/moe.cuh"

namespace py = pybind11;

using Element = flashmoe::ACC::Element;

torch::Tensor moe_forward(
    torch::Tensor input,               // [batch, seq_len, hidden_size] - Activations
    torch::Tensor gate_weights,        // [hidden_size, num_experts] - Gate weights
    torch::Tensor expert_weights       // [local_experts, 2, intermediate_size, hidden_size] - Expert weights
) {
    TORCH_CHECK(flashmoe::isInitialized, "Must call initialize() before moe_forward");
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(gate_weights.is_cuda(), "Gate weights must be CUDA tensor");
    TORCH_CHECK(expert_weights.is_cuda(), "Expert weights must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gate_weights.is_contiguous(), "Gate weights must be contiguous");
    TORCH_CHECK(expert_weights.is_contiguous(), "Expert weights must be contiguous");
    
    const auto rank = flashmoe::getRank();
    
    // Get dimensions from compile-time config
    constexpr auto S = flashmoe::ACC::S::value;
    constexpr auto H = flashmoe::ACC::H::value;
    constexpr auto E = flashmoe::ACC::E::value;
    constexpr auto P = flashmoe::ACC::P::value;
    constexpr auto PX = flashmoe::ACC::PX::value;
    const auto nLx = flashmoe::hostBookkeeping.nLx;
    
    // Validate input dimensions match compile-time config
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch, seq, H]");
    TORCH_CHECK(input.size(0) * input.size(1) == S, 
                "Input batch*seq must equal compiled S=" + std::to_string(S) + 
                ". Got batch=" + std::to_string(input.size(0)) + 
                ", seq=" + std::to_string(input.size(1)) + 
                " (product=" + std::to_string(input.size(0) * input.size(1)) + ")");
    TORCH_CHECK(input.size(2) == H, 
                "Input hidden_size must equal compiled H=" + std::to_string(H) + 
                ". Got " + std::to_string(input.size(2)));
    TORCH_CHECK(gate_weights.size(0) == H && gate_weights.size(1) == E,
                "Gate weights must be [H=" + std::to_string(H) + 
                ", E=" + std::to_string(E) + "]. Got [" + 
                std::to_string(gate_weights.size(0)) + ", " + 
                std::to_string(gate_weights.size(1)) + "]");
    TORCH_CHECK(expert_weights.size(0) == nLx, 
                "Expert count mismatch. Expected " + std::to_string(nLx) + 
                " local experts, got " + std::to_string(expert_weights.size(0)));
    TORCH_CHECK(expert_weights.size(1) == 2, 
                "Expert weights must have up and down projections [nLx, 2, P, H]");
    TORCH_CHECK(expert_weights.size(2) == P && expert_weights.size(3) == H,
                "Expert weights must be [*, 2, P=" + std::to_string(P) + 
                ", H=" + std::to_string(H) + "]. Got [*, 2, " + 
                std::to_string(expert_weights.size(2)) + ", " + 
                std::to_string(expert_weights.size(3)) + "]");
    
    // Calculate memory layout
    constexpr unsigned long aZ = S * H;              // Activations
    constexpr auto gwZ = aZ + PX * H;                // + Gate weights
    const auto bZ = gwZ + nLx * P * H;               // + Expert up weights
    const auto b2Z = bZ + nLx * P * H;               // + Expert down weights
    const auto dZ = b2Z + nLx * (P + H);             // + Bias
    const auto gZ = dZ + S * PX;                     // + Gate output
    const auto cZ = gZ + S * H;                      // + MoE output
    
    // Allocate device memory for all data
    cuda::std::byte* p;
    FLASHMOE_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(Element), flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(Element), flashmoe::flashmoeStream));
    
    auto* __restrict__ dP = reinterpret_cast<Element*>(p);
    
    // Copy activations
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        dP,
        input.data_ptr(),
        aZ * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));
    
    // Copy gate weights
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        dP + aZ,
        gate_weights.data_ptr(),
        E * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));
    
    // Copy expert weights
    for (uint i = 0; i < nLx; ++i) {
        // Copy up projection
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            dP + gwZ + i * (P * H),
            expert_weights[i][0].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));
        
        // Copy down projection
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            dP + bZ + i * (P * H),
            expert_weights[i][1].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));
    }
    
    // Call kernel
    float timed = 0;
    flashmoe::moe::forwardHostBench<0, 1>(p, p + dZ * sizeof(Element), timed);
    
    printf("Process %d: FlashMoE forward pass took %.2f ms\n", rank, timed);
    
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    
    // Extract output
    auto output = torch::empty({input.size(0), input.size(1), H}, 
                               torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        output.data_ptr(),
        dP + gZ,
        S * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));
    
    // Synchronize
    FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoe::flashmoeStream));
    
    // Free memory
    FLASHMOE_CHECK_CUDA(cudaFreeAsync(p, flashmoe::flashmoeStream));
    
    return output;
}

py::tuple moe_backward(
    torch::Tensor grad_output,         // [batch, seq_len, hidden_size] - Gradient from loss
    torch::Tensor input,               // Saved hidden states / original activations
    torch::Tensor gate_weights,        // Same gate weights that were used in forward
    torch::Tensor expert_weights       // [local_experts, 2, intermediate_size, hidden_size]
) {
    TORCH_CHECK(flashmoe::isInitialized, "Must call initialize() before moe_backward");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gate_weights.is_cuda(), "Gate weights must be a CUDA tensor");
    TORCH_CHECK(expert_weights.is_cuda(), "Expert weights must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gate_weights.is_contiguous(), "Gate weights must be contiguous");
    TORCH_CHECK(expert_weights.is_contiguous(), "Expert weights must be contiguous");
    TORCH_CHECK(grad_output.dtype() == input.dtype(),
        "grad_output and input must share the same dtype");
    TORCH_CHECK(gate_weights.dtype() == input.dtype(),
        "gate_weights must match input dtype");
    TORCH_CHECK(expert_weights.dtype() == input.dtype(),
        "expert_weights must match input dtype");

    constexpr auto S = flashmoe::ACC::S::value;
    constexpr auto H = flashmoe::ACC::H::value;
    constexpr auto E = flashmoe::ACC::E::value;
    constexpr auto P = flashmoe::ACC::P::value;
    constexpr auto PX = flashmoe::ACC::PX::value;
    const auto nLx = flashmoe::hostBookkeeping.nLx;

    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch, seq, H]");
    TORCH_CHECK(input.size(0) * input.size(1) == S,
        "Input batch*seq must equal compiled S=" + std::to_string(S));
    TORCH_CHECK(input.size(2) == H,
        "Input hidden_size must equal compiled H=" + std::to_string(H));
    TORCH_CHECK(gate_weights.dim() == 2, "Gate weights must be 2D [H, E]");
    TORCH_CHECK(gate_weights.size(0) == H && gate_weights.size(1) == E,
        "Gate weights must be [H=" + std::to_string(H) + ", E=" + std::to_string(E) + "]");
    TORCH_CHECK(expert_weights.dim() == 4, "Expert weights must be 4D [nLx, 2, P, H]");
    TORCH_CHECK(expert_weights.size(0) == nLx, "Expert weights must match local expert count");
    TORCH_CHECK(expert_weights.size(1) == 2, "Expert weights must have two projections");
    TORCH_CHECK(expert_weights.size(2) == P && expert_weights.size(3) == H,
        "Expert weights must be [nLx, 2, P=" + std::to_string(P) +
        ", H=" + std::to_string(H) + "]");

    constexpr unsigned long aZ = S * H;
    constexpr unsigned long gwZ = aZ + PX * H;
    const auto bZ = gwZ + nLx * P * H;
    const auto b2Z = bZ + nLx * P * H;
    const auto cZ = b2Z + nLx * (P + H) + S * PX + S * H;

    cuda::std::byte* p = nullptr;
    FLASHMOE_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(Element), flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(Element), flashmoe::flashmoeStream));

    auto* __restrict__ dP = reinterpret_cast<Element*>(p);

    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        dP,
        input.data_ptr(),
        aZ * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));

    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        dP + aZ,
        gate_weights.data_ptr(),
        E * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));

    for (uint i = 0; i < nLx; ++i) {
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            dP + gwZ + i * (P * H),
            expert_weights[i][0].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));

        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            dP + bZ + i * (P * H),
            expert_weights[i][1].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));
    }

    // z1: preGEMM pre-activation values, z2: postGEMM pre-activation values
    TORCH_CHECK(flashmoe::hostBookkeeping.savedZ1 != nullptr,
        "Saved activation z1 buffer is not initialized");
    TORCH_CHECK(flashmoe::hostBookkeeping.savedZ2 != nullptr,
        "Saved activation z2 buffer is not initialized");

    const auto savedActivations = cuda::std::array<const cuda::std::byte*, GEMMs>{
        CONST_CAST_TO(cuda::std::byte, flashmoe::hostBookkeeping.savedZ1),
        CONST_CAST_TO(cuda::std::byte, flashmoe::hostBookkeeping.savedZ2)
    };

    auto grad_input = torch::empty_like(input);
    float timed = 0;
    flashmoe::moe::backwardHost(
        grad_output.data_ptr(),
        &savedActivations,
        grad_input.data_ptr(),
        p,
        timed
    );

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());

    TORCH_CHECK(flashmoe::hostBookkeeping.gradGateWeights != nullptr,
        "Gate gradient buffers are not initialized");
    TORCH_CHECK(flashmoe::hostBookkeeping.gradWeights != nullptr,
        "Expert gradient buffers are not initialized");

    auto grad_gate_weights = torch::empty({H, E}, options);
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        grad_gate_weights.data_ptr(),
        flashmoe::hostBookkeeping.gradGateWeights,
        H * E * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));

    auto grad_expert_up = torch::empty({static_cast<int64_t>(nLx), P, H}, options);
    auto grad_expert_down = torch::empty({static_cast<int64_t>(nLx), P, H}, options);
    auto grad_bias_up = torch::empty({static_cast<int64_t>(nLx), P}, options);
    auto grad_bias_down = torch::empty({static_cast<int64_t>(nLx), H}, options);
    {
        const auto totalGradWeightsSize = (2 * P * H + P + H) * nLx;
        float debugVals[10] = {0};
        const auto numToCheck = std::min(10UL, static_cast<unsigned long>(totalGradWeightsSize));
        FLASHMOE_CHECK_CUDA(cudaMemcpy(debugVals, flashmoe::hostBookkeeping.gradWeights,
            numToCheck * sizeof(float), cudaMemcpyDeviceToHost));
        float debugSum = 0;
        for (size_t i = 0; i < numToCheck; ++i) {
            debugSum += debugVals[i];
        }
        printf("DEBUG moe_backward: gradWeights buffer check - first 10 vals sum=%.6f, vals=[%.4f,%.4f,%.4f,%.4f,...]\n",
               debugSum, debugVals[0], debugVals[1], debugVals[2], debugVals[3]);
        printf("DEBUG moe_backward: gradWeights ptr=%p, nLx=%u, expertStride=%lu, totalSize=%lu\n",
               flashmoe::hostBookkeeping.gradWeights, nLx, 2UL * P * H + P + H, totalGradWeightsSize);
    }

    const auto expertStride = 2 * P * H + P + H;
    for (uint i = 0; i < nLx; ++i) {
        const auto* __restrict__ base = flashmoe::hostBookkeeping.gradWeights + i * expertStride;
        auto* __restrict__ upDst = reinterpret_cast<Element*>(grad_expert_up.data_ptr()) + i * P * H;
        auto* __restrict__ downDst = reinterpret_cast<Element*>(grad_expert_down.data_ptr()) + i * P * H;
        auto* __restrict__ biasUpDst = reinterpret_cast<Element*>(grad_bias_up.data_ptr()) + i * P;
        auto* __restrict__ biasDownDst = reinterpret_cast<Element*>(grad_bias_down.data_ptr()) + i * H;

        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            upDst,
            base,
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));

        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            downDst,
            base + P * H,
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));

        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            biasUpDst,
            base + 2 * P * H,
            P * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));

        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            biasDownDst,
            base + 2 * P * H + P,
            H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));
    }

    FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaFreeAsync(p, flashmoe::flashmoeStream));

    return py::make_tuple(
        grad_input,
        grad_gate_weights,
        grad_expert_up,
        grad_expert_down,
        grad_bias_up,
        grad_bias_down,
        timed
    );
}


/**
 * Initialize NVSHMEM/Flashmoe
 */
void initialize() {
    flashmoe::initialize();
}


/**
 * Finalize NVSHMEM/Flashmoe
 */
void finalize() {
    flashmoe::finalize();
}

// Helper function
py::dict get_compiled_config() {
    py::dict result;
    result["S"] = flashmoe::ACC::S::value;
    result["H"] = flashmoe::ACC::H::value;
    result["E"] = flashmoe::ACC::E::value;
    result["P"] = flashmoe::ACC::P::value;
    result["PX"] = flashmoe::ACC::PX::value;
    result["Element_size"] = sizeof(Element);
    return result;
}

py::dict get_bookkeeping() {
    py::dict result;
    result["nLx"] = flashmoe::hostBookkeeping.nLx;
    return result;
}

uint16_t get_num_local_experts() {
    return flashmoe::hostBookkeeping.nLx;
}

/**
 * PyBind11 module definition
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMoE: Fast Distributed MoE in a Single Kernel";
    
    m.def("moe_forward", &moe_forward, 
          "MoE forward pass. Tensors must match compiled config dimensions.",
          py::arg("input"),
          py::arg("gate_weights"),
          py::arg("expert_weights"));

    m.def("moe_backward", &moe_backward,
          "MoE backward pass (requires saved activations + weights).",
          py::arg("grad_output"),
          py::arg("input"),
          py::arg("gate_weights"),
          py::arg("expert_weights"));
    
    m.def("initialize", &initialize,
          "Initialize NVSHMEM/Flashmoe");
    
    m.def("finalize", &finalize,
          "Finalize NVSHMEM/Flashmoe");

    m.def("get_compiled_config", &get_compiled_config,
      "Get compile-time configuration values");

    m.def("get_bookkeeping", &get_bookkeeping,
      "Get internal bookkeeping values");

    m.def("get_num_local_experts", &get_num_local_experts,
      "Get the number of local experts");
}

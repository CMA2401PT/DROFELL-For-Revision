// Symmetric Marlin INT4 (kU4B8) reconstruction and dense GEMM helpers.
// Marlin stores each 16 x 64 weight tile as 128 packed int32 words. Each
// reconstruction thread handles one packed word and unpacks its eight nibbles.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kRowsPerTile = 16;
constexpr int kColumnsPerTile = 64;
constexpr int kValuesPerTile = kRowsPerTile * kColumnsPerTile;
constexpr int kValuesPerWord = 8;
constexpr int kWordsPerTile = kValuesPerTile / kValuesPerWord;

__global__ void marlin_dequant_tile_kernel(
    const int32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const int16_t* __restrict__ weight_permutation,
    const int32_t* __restrict__ scale_column_map,
    const int32_t* __restrict__ sort_indices,
    half* __restrict__ output,
    int in_features,
    int out_features,
    int group_size) {
  const int packed_row_tile = blockIdx.y;
  const int column_tile = blockIdx.x;
  const int packed_word_in_tile = threadIdx.x;
  const int packed_words_per_row_tile = out_features * 2;
  const int packed_word_offset = column_tile * kWordsPerTile;
  if (packed_word_offset + packed_word_in_tile >=
      packed_words_per_row_tile) {
    return;
  }

  const uint32_t packed = static_cast<uint32_t>(
      qweight[static_cast<int64_t>(packed_row_tile) *
                  packed_words_per_row_tile +
              packed_word_offset + packed_word_in_tile]);
  const int permutation_offset = packed_word_in_tile * kValuesPerWord;

#pragma unroll
  for (int nibble = 0; nibble < kValuesPerWord; ++nibble) {
    // weight_permutation maps a packed scalar position back to its logical
    // position within a 16 x 64 Marlin tile.
    const int logical = weight_permutation[permutation_offset + nibble];
    const int logical_column_tile = logical >> 8;
    const int logical_in_16x16 = logical & 255;
    const int packed_row =
        packed_row_tile * kRowsPerTile + (logical_in_16x16 >> 4);
    const int output_column =
        column_tile * kColumnsPerTile +
        logical_column_tile * 16 + (logical_in_16x16 & 15);
    if (packed_row >= in_features || output_column >= out_features) {
      continue;
    }

    const int output_row = sort_indices == nullptr
        ? packed_row
        : sort_indices[packed_row];
    const int group = packed_row / group_size;
    const half scale = scales[
        static_cast<int64_t>(group) * out_features +
        scale_column_map[output_column]];
    const int quantized = (packed >> (4 * nibble)) & 0xF;
    output[static_cast<int64_t>(output_row) * out_features + output_column] =
        __hmul(__int2half_rn(quantized - 8), scale);
  }
}

void check_dense_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& output) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "dense Marlin tensors must be CUDA tensors");
  TORCH_CHECK(x.device() == weight.device() && x.device() == output.device(),
              "dense Marlin tensors must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kFloat16 &&
                  output.scalar_type() == torch::kFloat16,
              "dense Marlin GEMM currently supports float16 only");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous(),
              "dense Marlin GEMM tensors must be contiguous");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && output.dim() == 2,
              "dense Marlin GEMM expects rank-2 tensors");
}

}  // namespace

void marlin_dequant_into(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor weight_permutation,
    torch::Tensor scale_column_map,
    torch::Tensor sort_indices,
    torch::Tensor output,
    int64_t in_features,
    int64_t out_features,
    int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && output.is_cuda(),
              "Marlin reconstruction tensors must be CUDA tensors");
  TORCH_CHECK(weight_permutation.is_cuda() && scale_column_map.is_cuda() &&
                  sort_indices.is_cuda(),
              "Marlin reconstruction metadata must be CUDA tensors");
  TORCH_CHECK(qweight.device() == scales.device() &&
                  qweight.device() == weight_permutation.device() &&
                  qweight.device() == scale_column_map.device() &&
                  qweight.device() == sort_indices.device() &&
                  qweight.device() == output.device(),
              "Marlin reconstruction tensors must be on the same device");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "Marlin qweight must be int32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16 &&
                  output.scalar_type() == torch::kFloat16,
              "Marlin reconstruction currently supports float16 only");
  TORCH_CHECK(weight_permutation.scalar_type() == torch::kInt16 &&
                  weight_permutation.numel() == kValuesPerTile,
              "Marlin weight permutation must contain 1024 int16 values");
  TORCH_CHECK(scale_column_map.scalar_type() == torch::kInt32 &&
                  scale_column_map.numel() == out_features,
              "Marlin scale-column map has the wrong shape");
  TORCH_CHECK(output.is_contiguous() &&
                  output.numel() == in_features * out_features,
              "Marlin reconstruction output has the wrong shape");
  TORCH_CHECK(qweight.is_contiguous() && scales.is_contiguous() &&
                  weight_permutation.is_contiguous() &&
                  scale_column_map.is_contiguous() &&
                  sort_indices.is_contiguous(),
              "Marlin reconstruction tensors must be contiguous");
  TORCH_CHECK(group_size > 0 && in_features % group_size == 0,
              "Marlin group_size must divide in_features");
  TORCH_CHECK(in_features % kRowsPerTile == 0,
              "Marlin in_features must be divisible by 16");
  TORCH_CHECK(out_features % kColumnsPerTile == 0,
              "Marlin out_features must be divisible by 64");
  TORCH_CHECK(qweight.numel() == in_features * out_features / 8,
              "Marlin qweight has the wrong number of packed values");
  TORCH_CHECK(scales.numel() ==
                  (in_features / group_size) * out_features,
              "Marlin scales has the wrong number of values");
  TORCH_CHECK(sort_indices.numel() == 0 ||
                  sort_indices.numel() == in_features,
              "Marlin sort_indices must be empty or contain in_features values");
  c10::cuda::CUDAGuard device_guard(qweight.device());

  const int32_t* sort_indices_ptr = sort_indices.numel() == 0
      ? nullptr
      : sort_indices.data_ptr<int32_t>();
  const dim3 block(kWordsPerTile);
  const dim3 grid(
      (out_features + kColumnsPerTile - 1) / kColumnsPerTile,
      (in_features + kRowsPerTile - 1) / kRowsPerTile);
  const auto stream = at::cuda::getCurrentCUDAStream();
  marlin_dequant_tile_kernel<<<grid, block, 0, stream>>>(
      qweight.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      weight_permutation.data_ptr<int16_t>(),
      scale_column_map.data_ptr<int32_t>(),
      sort_indices_ptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      in_features,
      out_features,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void marlin_dense_matmul_out(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor output,
    bool transpose_weight) {
  check_dense_inputs(x, weight, output);
  c10::cuda::CUDAGuard device_guard(x.device());
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  cublasStatus_t status;
  if (!transpose_weight) {
    // Row-major X[M,K] @ W[K,N] is column-major W[N,K] @ X[K,M].
    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    TORCH_CHECK(weight.size(0) == k,
                "forward dense Marlin GEMM has incompatible K");
    const int64_t n = weight.size(1);
    TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                "forward dense Marlin GEMM output has the wrong shape");
    status = cublasHgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), n,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), k,
        &beta,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), n);
  } else {
    // Row-major X[M,N] @ W.T[N,K].
    const int64_t m = x.size(0);
    const int64_t n = x.size(1);
    TORCH_CHECK(weight.size(1) == n,
                "backward dense Marlin GEMM has incompatible N");
    const int64_t k = weight.size(0);
    TORCH_CHECK(output.size(0) == m && output.size(1) == k,
                "backward dense Marlin GEMM output has the wrong shape");
    status = cublasHgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        k, m, n,
        &alpha,
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), n,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), n,
        &beta,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), k);
  }
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "dense Marlin cublasHgemm failed with status ", status);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "marlin_dequant_into",
      &marlin_dequant_into,
      "Reconstruct symmetric Marlin INT4 weights into a dense tensor");
  module.def(
      "marlin_dense_matmul_out",
      &marlin_dense_matmul_out,
      "Multiply using a reconstructed Marlin weight and preallocated output");
}

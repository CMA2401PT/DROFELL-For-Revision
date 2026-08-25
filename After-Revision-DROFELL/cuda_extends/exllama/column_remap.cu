// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#include "column_remap.cuh"
#include "util.cuh"

// const int SHUF_BLOCKSIZE_X = 256;
// const int SHUF_BLOCKSIZE_Y = 16;

__global__ void column_remap_kernel
(
    const half* __restrict__ x,
    half* __restrict__ x_new,
    const int x_width,
    const int x_height,
    const uint32_t* x_map,
    const int collumn_shuffle_block_x,
    const int collumn_shuffle_block_y
)
{
    int x_column = collumn_shuffle_block_x * blockIdx.x + threadIdx.x;
    int x_row = collumn_shuffle_block_y * blockIdx.y;

    int x_stride = x_width;
    int x_idx = x_row * x_stride + x_column;

    int x_row_end = min(x_row + collumn_shuffle_block_y, x_height);
    int x_idx_end = x_row_end * x_stride + x_column;

    int s_column = x_map[x_column];
    int s_idx = x_row * x_stride + s_column;

    while (x_idx < x_idx_end)
    {
        x_new[x_idx] = x[s_idx];
        x_idx += x_stride;
        s_idx += x_stride;
    }
}

// Remap columns in x to correspond to sequential group index before matmul
//
// perform x -> seq_x such that seq_x @ seq_w == x @ w

void column_remap_cuda
(
    const half* x,
    half* x_new,
    const int x_height,
    const int x_width,
    const uint32_t* x_map,
    const int collumn_shuffle_block_x,
    const int collumn_shuffle_block_y,
    cudaStream_t alt_stream
)
{
    dim3 threads(collumn_shuffle_block_x, 1, 1);

    dim3 blocks
    (
        (x_width + collumn_shuffle_block_x - 1) / collumn_shuffle_block_x,
        (x_height + collumn_shuffle_block_y - 1) / collumn_shuffle_block_y,
        1
    );

    column_remap_kernel<<<blocks, threads,0, alt_stream>>> (x, x_new, x_width, x_height, x_map,collumn_shuffle_block_x,collumn_shuffle_block_y);
}

__global__ void column_restore_kernel
(
    const half* __restrict__ x,
    half* __restrict__ x_new,
    const int x_width,
    const int x_height,
    const uint32_t* x_map,
    const int collumn_shuffle_block_x,
    const int collumn_shuffle_block_y
)
{
    int x_column = collumn_shuffle_block_x * blockIdx.x + threadIdx.x;
    int x_row = collumn_shuffle_block_y * blockIdx.y;

    int x_stride = x_width;
    int x_idx = x_row * x_stride + x_column;

    int x_row_end = min(x_row + collumn_shuffle_block_y, x_height);
    int x_idx_end = x_row_end * x_stride + x_column;

    int s_column = x_map[x_column];
    int s_idx = x_row * x_stride + s_column;

    while (x_idx < x_idx_end)
    {
        x_new[s_idx] = x[x_idx];
        x_idx += x_stride;
        s_idx += x_stride;
    }
}

void column_restore_cuda
(
    const half* x,
    half* x_new,
    const int x_height,
    const int x_width,
    const uint32_t* x_map,
    const int collumn_shuffle_block_x,
    const int collumn_shuffle_block_y,
    cudaStream_t alt_stream
)
{
    dim3 threads(collumn_shuffle_block_x, 1, 1);

    dim3 blocks
    (
        (x_width + collumn_shuffle_block_x - 1) / collumn_shuffle_block_x,
        (x_height + collumn_shuffle_block_y - 1) / collumn_shuffle_block_y,
        1
    );

    column_restore_kernel<<<blocks, threads, 0, alt_stream>>>(x, x_new, x_width, x_height, x_map,collumn_shuffle_block_x,collumn_shuffle_block_y);
}

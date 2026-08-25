// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _q4_matrix_cuh
#define _q4_matrix_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

class Q4Matrix
{
public:

    int device;

    int height;
    int width;
    int groups;
    int groupsize;
    bool v2_mode;

    uint32_t* cuda_qweight = NULL;
    uint32_t* cuda_qzeros = NULL;
    half* cuda_scales = NULL;
    uint32_t* cuda_x_map = NULL;

    Q4Matrix
    (
        const int _height,
        const int _width,
        const int _groups,

        uint32_t* _qweight,
        uint32_t* _qzeros,
        half* _scales,
        uint32_t* _g_idx,
        uint32_t* _cuda_x_map,

        const int _device,
        bool _v2_mode,
        cudaStream_t alt_stream=NULL
    );

    ~Q4Matrix();

    void reconstruct(
        half* out,
        const int reconstruct_threads_x,
        const int reconstruct_threads_y,
        cudaStream_t alt_stream=NULL
    );

    void reconstruct_exllama_v2(
        half* out,
        const int block_kn_size,
        cudaStream_t alt_stream=NULL
    );

    void reconstruct_nv_bfloat16(
        nv_bfloat16* out,
        const int reconstruct_threads_x,
        const int reconstruct_threads_y,
        cudaStream_t alt_stream=NULL
    );

    void reconstruct_nv_bfloat16_exllama_v2(
        nv_bfloat16* out,
        const int block_kn_size,
        cudaStream_t alt_stream=NULL
    );

private:

    void make_sequential(const uint32_t* cpu_g_idx,cudaStream_t alt_stream);
};

#endif
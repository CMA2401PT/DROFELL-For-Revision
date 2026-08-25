// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _util_cuh
#define _util_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#define cudaUnspecified cudaErrorApiFailureBase

// React to failure on return code != cudaSuccess

#define _cuda_check(fn) \
do { \
    {_cuda_err = fn;} \
    if (_cuda_err != cudaSuccess) goto _cuda_fail; \
} while(false)

// React to failure on return code == 0

#define _alloc_check(fn) \
do { \
    if (!(fn)) { _cuda_err = cudaUnspecified; goto _cuda_fail; } \
    else _cuda_err = cudaSuccess; \
} while(false)

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#endif

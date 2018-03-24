#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   int abort = 1;
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif

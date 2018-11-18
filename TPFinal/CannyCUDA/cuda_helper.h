#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime_api.h> 

#define TILE_WIDTH 16u
#define TILE_HEIGHT 16u

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

inline unsigned int calc_x(unsigned int cols)
{
   return (cols + TILE_WIDTH - 1u) / TILE_WIDTH;
}

inline unsigned int calc_y(unsigned int rows)
{
   return (rows + TILE_HEIGHT - 1u) / TILE_HEIGHT;
}

inline void get_dimgrid(dim3* grid, const unsigned int cols, const unsigned int rows)
{
   grid->x = calc_x(cols);
   grid->y = calc_y(rows); 
}

inline void get_dimblock(dim3* block)
{
   block->x = TILE_WIDTH;
   block->y = TILE_HEIGHT; 
}

#endif

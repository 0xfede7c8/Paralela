#ifndef CANNY_EDGE_H
#define CANNY_EDGE_H

#define VERBOSE 1
#define BOOSTBLURFACTOR 90.0

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

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

void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname);

void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int *smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols,
        short int **delta_x, short int **delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
        short int **magnitude);
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
        float tlow, float thigh, unsigned char *edge);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);

__global__
void cuda_gaussian_smoothX( const unsigned char* image, 
                            float*               tempim, 
                            const int            rows, 
                            const int            cols, 
                            const float*         kernel, 
                            short int*           smoothedim, 
                            const int            windowsize);

__global__
void cuda_gaussian_smoothY( const unsigned char* image, 
                            float*               tempim, 
                            const int            rows, 
                            const int            cols, 
                            const float*         kernel, 
                            short int*           smoothedim, 
                            const int            windowsize);

void printTime(const struct timeval* start,const struct timeval* stop, const char* msg);

#endif
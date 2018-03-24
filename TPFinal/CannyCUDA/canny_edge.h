#ifndef CANNY_EDGE_H
#define CANNY_EDGE_H

#define VERBOSE 1
#define BOOSTBLURFACTOR 90.0

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_helper.h"

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
                            const int            windowsize,
                            const int            center);

__global__
void cuda_gaussian_smoothY( const unsigned char* image,
                            float*               tempim,
                            const int            rows,
                            const int            cols,
                            const float*         kernel,
                            short int*           smoothedim,
                            const int            windowsize,
                            const int            center);

__global__
void cuda_derrivative_x_y(  short int* smoothedimDevice,
                            int rows,
                            int cols,
                            short int* delta_xDevice,
                            short int* delta_yDevice);

__global__
void cuda_magnitude_x_y(short int *delta_x,
                        short int *delta_y,
                        int rows, int cols,
                        short int *magnitude);


void printTime(const struct timeval* start,const struct timeval* stop, const char* msg);

#endif

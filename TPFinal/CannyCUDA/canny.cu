#include "canny_edge.h"
#include "hysteresis.h"
#include "pgm_io.h"

#include "cuda_helper.h"

void canny(unsigned char *image, const int rows, const int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname)
{  
   unsigned char *nms_device,
                 *nms;        /* Points that are local maximal magnitude. */
   short int *smoothedimDevice, /* The device image after gaussian smoothing.      */
             *delta_x_device,        /* The first devivative image, x-direction. */
             *delta_y_device,        /* The first derivative image, y-direction. */
             *magnitude_device,      /* The magnitude of the gradient image.      */
             *magnitude;

   unsigned char* image_device, edge_device;

   size_t smoothedimSz = (rows*cols)*sizeof(short int);

   /****************************************************************************
   * Allocs memory for cuda image operations and copies the read info into it.
   ****************************************************************************/
   const size_t imagesz = rows * cols;
   //image to device
   gpuErrchk(cudaMalloc((void**) &image_device, imagesz));
   gpuErrchk(cudaMemcpy(image_device, image, imagesz, cudaMemcpyHostToDevice));
   //allocate memory for result
   gpuErrchk(cudaMalloc((void**) &edge_device, imagesz));
   gpuErrchk(cudaMalloc((void**) &smoothedimDevice, smoothedimSz));

   /****************************************************************************
   * Perform gaussian smoothing on the image using the input standard
   * deviation.
   ****************************************************************************/
   if(VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");
   gaussian_smooth(image_device, rows, cols, sigma, smoothedimDevice);

   /****************************************************************************
   * Compute the first derivative in the x and y directions.
   ****************************************************************************/
   if(VERBOSE) printf("Computing the X and Y first derivatives.\n");
   derrivative_x_y(smoothedimDevice, rows, cols, &delta_x_device, &delta_y_device);

   /****************************************************************************
   * Compute the magnitude of the gradient.
   ****************************************************************************/
   if(VERBOSE) printf("Computing the magnitude of the gradient.\n");

   magnitude_x_y(delta_x_device, delta_y_device, rows, cols, &magnitude_device);

   /****************************************************************************
   * Perform non-maximal suppression.
   ****************************************************************************/

   if(VERBOSE) printf("   Computing the Non Max suppresion.\n");

   non_max_supp(magnitude_device, delta_x_device, delta_y_device, rows, cols, &nms_device);

   // /*Testing host reallock to check workingshit*/
   const size_t deltaSz = rows*cols*sizeof(short int);
   magnitude = (short int*) malloc(deltaSz);

   const size_t nms_size = rows*cols * sizeof(unsigned char);
   nms = (unsigned char*) malloc(nms_size);

   gpuErrchk(cudaMemcpy(magnitude, magnitude_device, deltaSz, cudaMemcpyDeviceToHost));
   gpuErrchk(cudaMemcpy(nms, nms_device, nms_size, cudaMemcpyDeviceToHost));

   /****************************************************************************
   * Use hysteresis to mark the edge pixels.
   ****************************************************************************/
   if(VERBOSE) printf("Doing hysteresis thresholding.\n");
   if((*edge=(unsigned char *)calloc(rows*cols, sizeof(unsigned char))) ==NULL){
      fprintf(stderr, "Error allocating the edge image.\n");
      exit(1);
   }

   apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);

   /****************************************************************************
   * Free all of the memory that we allocated except for the edge image that
   * is still being used to store out result.
   ****************************************************************************/
   cudaFree(delta_x_device);
   cudaFree(delta_y_device);
   cudaFree(magnitude);
   cudaFree(smoothedimDevice);
   cudaFree(image_device);
   free(magnitude);
   free(nms);
}

/*******************************************************************************
* Procedure: radian_direction
* Purpose: To compute a direction of the gradient image from component dx and
* dy images. Because not all derriviatives are computed in the same way, this
* code allows for dx or dy to have been calculated in different ways.
*
* FOR X:  xdirtag = -1  for  [-1 0  1]
*         xdirtag =  1  for  [ 1 0 -1]
*
* FOR Y:  ydirtag = -1  for  [-1 0  1]'
*         ydirtag =  1  for  [ 1 0 -1]'
*
* The resulting angle is in radians measured counterclockwise from the
* xdirection. The angle points "up the gradient".
*******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag)
{
   int r, c, pos;
   float *dirim=NULL;
   double dx, dy;

   /****************************************************************************
   * Allocate an image to store the direction of the gradient.
   ****************************************************************************/
   if((dirim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
      fprintf(stderr, "Error allocating the gradient direction image.\n");
      exit(1);
   }
   *dir_radians = dirim;

   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
         dx = (double)delta_x[pos];
         dy = (double)delta_y[pos];

         if(xdirtag == 1) dx = -dx;
         if(ydirtag == -1) dy = -dy;

         dirim[pos] = (float)angle_radians(dx, dy);
      }
   }
}

/*******************************************************************************
* FUNCTION: angle_radians
* PURPOSE: This procedure computes the angle of a vector with components x and
* y. It returns this angle in radians with the answer being in the range
* 0 <= angle <2*PI.
*******************************************************************************/
double angle_radians(double x, double y)
{
   double xu, yu, ang;

   xu = fabs(x);
   yu = fabs(y);

   if((xu == 0) && (yu == 0)) return(0);

   ang = atan(yu/xu);

   if(x >= 0){
      if(y >= 0) return(ang);
      else return(2*M_PI - ang);
   }
   else{
      if(y >= 0) return(M_PI - ang);
      else return(M_PI + ang);
   }
}

/*******************************************************************************
* PROCEDURE: magnitude_x_y
* PURPOSE: Compute the magnitude of the gradient. This is the square root of
* the sum of the squared derivative values.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void magnitude_x_y( short int *delta_x,
                    short int *delta_y,
                    int rows, int cols,
                    short int **magnitude)
{
    gpuErrchk(cudaMalloc((void**) magnitude, rows*cols*sizeof(short int)));

    dim3 dgrid;
    dim3 dblock;
   
    get_dimgrid(&dgrid, cols, rows);
    get_dimblock(&dblock);

    cuda_magnitude_x_y<<<dgrid, dblock>>>(delta_x, delta_y, rows, cols, *magnitude);

}

__global__
void cuda_magnitude_x_y(short int *delta_x,
                        short int *delta_y,
                        int rows,
                        int cols,
                        short int *magnitude)
{
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols)
    {
        const int pos = r*cols + c;
        const int sq1 = (int)delta_x[pos] * (int)delta_x[pos];
        const int sq2 = (int)delta_y[pos] * (int)delta_y[pos];

        magnitude[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
    }
}

/*******************************************************************************
* PROCEDURE: derrivative_x_y
* PURPOSE: Compute the first derivative of the image in both the x any y
* directions. The differential filters that are used are:
*
*                                          -1
*         dx =  -1 0 +1     and       dy =  0
*                                          +1
*
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void derrivative_x_y(   short int* smoothedimDevice,
                        int rows, int cols,
                        short int** delta_xDevice,
                        short int** delta_yDevice)
{

   /****************************************************************************
   * Allocate images to store the derivatives.
   ****************************************************************************/

   gpuErrchk(cudaMalloc((void**) delta_xDevice, rows*cols*sizeof(short int)));
   gpuErrchk(cudaMalloc((void**) delta_yDevice, rows*cols*sizeof(short int)));

   /****************************************************************************
   * Compute the x-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/

   dim3 dgrid;
   dim3 dblock;
   
   get_dimgrid(&dgrid, cols, rows);
   get_dimblock(&dblock);

   if(VERBOSE) printf("   Computing the X-Y direction derivative.\n");
   cuda_derrivative_x_y<<<dgrid, dblock>>>(smoothedimDevice, rows, cols, *delta_xDevice, *delta_yDevice);
}



__global__
void cuda_derrivative_x_y(  short int* smoothedimDevice,
                            int rows,
                            int cols,
                            short int* delta_xDevice,
                            short int* delta_yDevice)
{

    const unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols)
    {
        unsigned int pos;

        pos = r * cols + c;

        if (c == 0u)
        {
            delta_xDevice[pos] = smoothedimDevice[pos+1] - smoothedimDevice[pos];
        }
        else if (c == (cols - 1))
        {
            delta_xDevice[pos] = smoothedimDevice[pos] - smoothedimDevice[pos-1];
        }
        else
        {
            delta_xDevice[pos] = smoothedimDevice[pos+1] - smoothedimDevice[pos-1];
        }

        /****************************************************************************
        * Compute the y-derivative. Adjust the derivative at the borders to avoid
        * losing pixels.
        ****************************************************************************/

        if (r == 0)
        {
            delta_yDevice[pos] = smoothedimDevice[pos+cols] - smoothedimDevice[pos];
        }
        else if (r == (rows - 1))
        {
            delta_yDevice[pos] = smoothedimDevice[pos] - smoothedimDevice[pos-cols];
        }
        else
        {
            delta_yDevice[pos] = smoothedimDevice[pos+cols] - smoothedimDevice[pos-cols];
        }
    }
}


/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int *smoothedim)
{

   int windowsize;
   float *tempim,        /* Buffer for separable filter gaussian smoothing. */
         *kernel,        /* A one dimensional gaussian kernel. */
         *kernelDevice;

   /****************************************************************************
   * Create a 1-dimensional gaussian smoothing kernel.
   ****************************************************************************/
   if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
   make_gaussian_kernel(sigma, &kernel, &windowsize);

   size_t kernelSz = windowsize * sizeof(float);

   /*
   ***************************************************************************
   * Allocate a temporary buffer image
   ***************************************************************************
   */
   gpuErrchk(cudaMalloc((void**)&kernelDevice, kernelSz));
   gpuErrchk(cudaMalloc((void**)&tempim, (rows*cols)*sizeof(float)));

   /*Copy kernel to gpu*/
   gpuErrchk(cudaMemcpy(kernelDevice, kernel, kernelSz, cudaMemcpyHostToDevice));

   const int center = windowsize / 2;

   if(VERBOSE) printf("   Bluring the image.\n");

   dim3 dgrid;
   dim3 dblock;
   
   get_dimgrid(&dgrid, cols, rows);
   get_dimblock(&dblock);
   
   cuda_gaussian_smoothX<<<dgrid, dblock>>>(image, tempim, rows, cols, kernelDevice, center);
   cuda_gaussian_smoothY<<<dgrid, dblock>>>(image, tempim, rows, cols, kernelDevice, smoothedim, center);

   cudaFree(kernelDevice);
   cudaFree(tempim);
   free(kernel);
}

__global__
void cuda_gaussian_smoothX( const unsigned char* image,
                            float*               tempim,
                            const int            rows,
                            const int            cols,
                            const float*         kernel,
                            const int            center)
{
   const int r = blockIdx.y * blockDim.y + threadIdx.y;
   const int c = blockIdx.x * blockDim.x + threadIdx.x;
   if (r < rows && c < cols)
   {
        int cc;
        float dot = 0.0;
        float sum = 0.0;
        for(cc=(-center);cc<=center;cc++)
        {
            if(((c+cc) >= 0) && ((c+cc) < cols))
            {
                dot += (float)image[r*cols+(c+cc)] * kernel[center+cc];
                sum += kernel[center+cc];
            }
        }
        tempim[r*cols+c] = dot/sum;
   }
}

__global__
void cuda_gaussian_smoothY( const unsigned char* image,
                            float*               tempim,
                            const int            rows,
                            const int            cols,
                            const float*         kernel,
                            short int*           smoothedim,
                            const int            center)
{
   const int r = blockIdx.y * blockDim.y + threadIdx.y;
   const int c = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (r < rows && c < cols)
   {
       float dot = 0.0;
       float sum = 0.0;
       int rr;
       for(rr=(-center); rr<=center; rr++)
       {
            if(((r+rr) >= 0) && ((r+rr) < rows))
            {
                dot += tempim[(r+rr)*cols+c] * kernel[center+rr];
                sum += kernel[center+rr];
            }
       }
       smoothedim[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
    }
}


/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
* PURPOSE: Create a one dimensional gaussian kernel.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum=0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   center = (*windowsize) / 2;

   if(VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
   if((*kernel = (float *) calloc((*windowsize), sizeof(float))) == NULL){
      fprintf(stderr, "Error callocing the gaussian kernel array.\n");
      exit(1);
   }

   for(i=0;i<(*windowsize);i++){
      x = (float)(i - center);
      fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
      (*kernel)[i] = fx;
      sum += fx;
   }

   for(i=0;i<(*windowsize);i++) (*kernel)[i] /= sum;

   if(VERBOSE){
      printf("The filter coefficients are:\n");
      for(i=0;i<(*windowsize);i++)
         printf("kernel[%d] = %f\n", i, (*kernel)[i]);
   }
}

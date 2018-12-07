/*
"Canny" edge detector code:
---------------------------

This text file contains the source code for a "Canny" edge detector. It
was written by Mike Heath (heath@csee.usf.edu) using some pieces of a
Canny edge detector originally written by someone at Michigan State
University.

There are three 'C' source code files in this text file. They are named
"canny_edge.c", "hysteresis.c" and "pgm_io.c". They were written and compiled
under SunOS 4.1.3. Since then they have also been compiled under Solaris.
To make an executable program: (1) Separate this file into three files with
the previously specified names, and then (2) compile the code using

  gcc -o canny_edge canny_edge.c hysteresis.c pgm_io.c -lm
  (Note: You can also use optimization such as -O3)

The resulting program, canny_edge, will process images in the PGM format.
Parameter selection is left up to the user. A broad range of parameters to
use as a starting point are: sigma 0.60-2.40, tlow 0.20-0.50 and,
thigh 0.60-0.90.

If you are using a Unix system, PGM file format conversion tools can be found
at ftp://wuarchive.wustl.edu/graphics/graphics/packages/pbmplus/.
Otherwise, it would be easy for anyone to rewrite the image I/O procedures
because they are listed in the separate file pgm_io.c.

If you want to check your compiled code, you can download grey-scale and edge
images from http://marathon.csee.usf.edu/edge/edge_detection.html. You can use
the parameters given in the edge filenames and check whether the edges that
are output from your program match the edge images posted at that address.

Mike Heath
(10/29/96)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "hysteresis.h"

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
    int cols, char *comment, int maxval);

void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int **smoothedim);
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

void printTime(const struct timeval* start,const struct timeval* stop, const char* msg);

int main(int argc, char *argv[])
{
   char *infilename = NULL;  /* Name of the input image */
   char *dirfilename = NULL; /* Name of the output gradient direction image */
   char outfilename[128];    /* Name of the output "edge" image */
   char composedfname[128];  /* Name of the output "direction" image */
   unsigned char *image;     /* The input image */
   unsigned char *edge;      /* The output edge image */
   int rows, cols;           /* The dimensions of the image. */
   float sigma,              /* Standard deviation of the gaussian kernel. */
	 tlow,               /* Fraction of the high threshold in hysteresis. */
	 thigh;              /* High hysteresis threshold control. The actual
			        threshold is the (100 * thigh) percentage point
			        in the histogram of the magnitude of the
			        gradient image that passes non-maximal
			        suppression. */

   struct timeval stop, start;

   /****************************************************************************
   * Get the command line arguments.
   ****************************************************************************/
   if(argc < 5){
   fprintf(stderr,"\n<USAGE> %s image sigma tlow thigh [writedirim]\n",argv[0]);
      fprintf(stderr,"\n      image:      An image to process. Must be in ");
      fprintf(stderr,"PGM format.\n");
      fprintf(stderr,"      sigma:      Standard deviation of the gaussian");
      fprintf(stderr," blur kernel.\n");
      fprintf(stderr,"      tlow:       Fraction (0.0-1.0) of the high ");
      fprintf(stderr,"edge strength threshold.\n");
      fprintf(stderr,"      thigh:      Fraction (0.0-1.0) of the distribution");
      fprintf(stderr," of non-zero edge\n                  strengths for ");
      fprintf(stderr,"hysteresis. The fraction is used to compute\n");
      fprintf(stderr,"                  the high edge strength threshold.\n");
      fprintf(stderr,"      writedirim: Optional argument to output ");
      fprintf(stderr,"a floating point");
      fprintf(stderr," direction image.\n\n");
      exit(1);
   }

   infilename = argv[1];
   sigma = atof(argv[2]);
   tlow = atof(argv[3]);
   thigh = atof(argv[4]);

   if(argc == 6) dirfilename = infilename;
   else dirfilename = NULL;

   /****************************************************************************
   * Read in the image. This read function allocates memory for the image.
   ****************************************************************************/
   if(VERBOSE) printf("Reading the image %s.\n", infilename);
   if(read_pgm_image(infilename, &image, &rows, &cols) == 0){
      fprintf(stderr, "Error reading the input image, %s.\n", infilename);
      exit(1);
   }

   /****************************************************************************
   * Perform the edge detection. All of the work takes place here.
   ****************************************************************************/
   if(VERBOSE) printf("Starting Canny edge detection.\n");
   if(dirfilename != NULL){
      sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
      sigma, tlow, thigh);
      dirfilename = composedfname;
   }
   gettimeofday(&start, NULL);
   canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);
   gettimeofday(&stop, NULL);
   printTime(&start, &stop, "Canny took: : ");

   /****************************************************************************
   * Write out the edge image to a file.
   ****************************************************************************/
   sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename,
      sigma, tlow, thigh);
   if(VERBOSE) printf("Writing the edge iname in the file %s.\n", outfilename);
   if(write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0){
      fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
      exit(1);
   }
   free(image);
   return 0;
}

/*******************************************************************************
* PROCEDURE: canny
* PURPOSE: To perform canny edge detection.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname)
{
    FILE *fpdir=NULL;          /* File to write the gradient image to.     */
    unsigned char *nms;        /* Points that are local maximal magnitude. */
    short int *smoothedim,     /* The image after gaussian smoothing.      */
             *delta_x,        /* The first devivative image, x-direction. */
             *delta_y,        /* The first derivative image, y-direction. */
             *magnitude;      /* The magnitude of the gadient image.      */
    int r, c, pos;
    float *dir_radians=NULL;   /* Gradient direction image.                */

    struct timeval stop, start;

    /////////////////////////////////////////////////////////////////////////////
    // Perform gaussian smoothing on the image using the input standard
    // deviation.
    /////////////////////////////////////////////////////////////////////////////
    if(VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");
    gettimeofday(&start, NULL);
    gaussian_smooth(image, rows, cols, sigma, &smoothedim);
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "gaussian smooth took: : ");

    //---------------------------------------------------------------------------
    // Compute the first derivative in the x and y directions.
    //---------------------------------------------------------------------------
    if(VERBOSE) printf("Computing the X and Y first derivatives.\n");
    gettimeofday(&start, NULL);
    derrivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "derrivative_x_y took: : ");

    //---------------------------------------------------------------------------
    // This option to write out the direction of the edge gradient was added
    // to make the information available for computing an edge quality figure
    // of merit.
    //---------------------------------------------------------------------------
    if(fname != NULL){
      //-------------------------------------------------------------------------
      // Compute the direction up the gradient, in radians that are
      // specified counteclockwise from the positive x-axis.
      //-------------------------------------------------------------------------
      radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

      //-------------------------------------------------------------------------
      // Write the gradient direction image out to a file.
      //-------------------------------------------------------------------------
      if((fpdir = fopen(fname, "wb")) == NULL){
         fprintf(stderr, "Error opening the file %s for writing.\n", fname);
         exit(1);
      }
      fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
      fclose(fpdir);
      free(dir_radians);
    }

    //---------------------------------------------------------------------------
    // Compute the magnitude of the gradient.
    //---------------------------------------------------------------------------
    if(VERBOSE) printf("Computing the magnitude of the gradient.\n");
    gettimeofday(&start, NULL);
    magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "magnitude_x_y took: : ");

    //---------------------------------------------------------------------------
    // Perform non-maximal suppression.
    //---------------------------------------------------------------------------
    if(VERBOSE) printf("Doing the non-maximal suppression.\n");
    if((nms = (unsigned char *) calloc(rows*cols,sizeof(unsigned char)))==NULL){
      fprintf(stderr, "Error allocating the nms image.\n");
      exit(1);
    }

    gettimeofday(&start, NULL);
    non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "no_max_supp took: : ");

    //---------------------------------------------------------------------------
    // Use hysteresis to mark the edge pixels.
    //---------------------------------------------------------------------------
    if(VERBOSE) printf("Doing hysteresis thresholding.\n");
    if((*edge=(unsigned char *)calloc(rows*cols,sizeof(unsigned char))) ==NULL){
      fprintf(stderr, "Error allocating the edge image.\n");
      exit(1);
    }

    gettimeofday(&start, NULL);
    apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "apply_hysteresis took: : ");

    //---------------------------------------------------------------------------
    // Free all of the memory that we allocated except for the edge image that
    // is still being used to store out result.
    //---------------------------------------------------------------------------
    free(smoothedim);
    free(delta_x);
    free(delta_y);
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
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
        short int **magnitude)
{
   int r, c, pos, sq1, sq2;

   /****************************************************************************
   * Allocate an image to store the magnitude of the gradient.
   ****************************************************************************/
   if((*magnitude = (short *) calloc(rows*cols, sizeof(short))) == NULL){
      fprintf(stderr, "Error allocating the magnitude image.\n");
      exit(1);
   }

   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
         sq1 = (int)delta_x[pos] * (int)delta_x[pos];
         sq2 = (int)delta_y[pos] * (int)delta_y[pos];
         (*magnitude)[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
      }
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
void derrivative_x_y(short int *smoothedim, int rows, int cols,
        short int **delta_x, short int **delta_y)
{
    int r, c, pos;
    struct timeval stop, start;

    /****************************************************************************
    * Allocate images to store the derivatives.
    ****************************************************************************/
    if(((*delta_x) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
      fprintf(stderr, "Error allocating the delta_x image.\n");
      exit(1);
    }
    if(((*delta_y) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
      fprintf(stderr, "Error allocating the delta_x image.\n");
      exit(1);
    }

    /****************************************************************************
    * Compute the x-derivative. Adjust the derivative at the borders to avoid
    * losing pixels.
    ****************************************************************************/
    if(VERBOSE) printf("   Computing the X-direction derivative.\n");
    gettimeofday(&start, NULL);
    for(r=0;r<rows;r++){
      pos = r * cols;
      (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos];
      pos++;
      for(c=1;c<(cols-1);c++,pos++){
         (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos-1];
      }
      (*delta_x)[pos] = smoothedim[pos] - smoothedim[pos-1];
    }
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "Derivative in X took: ");

    /****************************************************************************
    * Compute the y-derivative. Adjust the derivative at the borders to avoid
    * losing pixels.
    ****************************************************************************/
    if(VERBOSE) printf("   Computing the Y-direction derivative.\n");
    gettimeofday(&start, NULL);
    for(r=0;r<rows;r++){
        pos = r * cols;
        if(r==0){
            for(c=0;c<cols;c++, pos++){
                (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos];
            }
        }
        else if(r==rows-1){
            for(c=0;c<cols;c++, pos++){
                (*delta_y)[pos] = smoothedim[pos] - smoothedim[pos-cols];
            }
        }
        else{
            for(c=0;c<cols;c++, pos++){
                (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos-cols];
            }
            
        }

    }
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "Derivative in Y took: ");
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int **smoothedim)
{
    int r, c, rr, cc,     /* Counter variables. */
      windowsize,        /* Dimension of the gaussian kernel. */
      center;            /* Half of the windowsize. */
    float *tempim,        /* Buffer for separable filter gaussian smoothing. */
         *kernel,        /* A one dimensional gaussian kernel. */
         dot,            /* Dot product summing variable. */
         sum;            /* Sum of the kernel weights variable. */
    struct timeval stop, start;

    /****************************************************************************
    * Create a 1-dimensional gaussian smoothing kernel.
    ****************************************************************************/
    if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
    make_gaussian_kernel(sigma, &kernel, &windowsize);
    center = windowsize / 2;

    /****************************************************************************
    * Allocate a temporary buffer image and the smoothed image.
    ****************************************************************************/
    if((tempim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
      fprintf(stderr, "Error allocating the buffer image.\n");
      exit(1);
    }
    if(((*smoothedim) = (short int *) calloc(rows*cols,
         sizeof(short int))) == NULL){
      fprintf(stderr, "Error allocating the smoothed image.\n");
      exit(1);
    }

    /****************************************************************************
    * Blur in the x - direction.
    ****************************************************************************/
    if(VERBOSE) printf("   Bluring the image in the X-direction.\n");
    gettimeofday(&start, NULL);
    for(r=0;r<rows;r++){
      for(c=0;c<cols;c++){
         dot = 0.0;
         sum = 0.0;
         for(cc=(-center);cc<=center;cc++){
            if(((c+cc) >= 0) && ((c+cc) < cols)){
               dot += (float)image[r*cols+(c+cc)] * kernel[center+cc];
               sum += kernel[center+cc];
            }
         }
         tempim[r*cols+c] = dot/sum;
      }
    }
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "Smoothed in X took: ");

    /****************************************************************************
    * Blur in the y - direction.
    ****************************************************************************/
    if(VERBOSE) printf("   Bluring the image in the Y-direction.\n");
    gettimeofday(&start, NULL);
    for(r=0;r<rows;r++){
      for(c=0;c<cols;c++){
         sum = 0.0;
         dot = 0.0;
         for(rr=(-center);rr<=center;rr++){
            if(((r+rr) >= 0) && ((r+rr) < rows)){
               dot += tempim[(r+rr)*cols+c] * kernel[center+rr];
               sum += kernel[center+rr];
            }
         }
         (*smoothedim)[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
      }
    }
    gettimeofday(&stop, NULL);
    printTime(&start, &stop, "Smoothed in Y took: ");

    free(tempim);
    free(kernel);
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

   /*
   if(VERBOSE){
      printf("The filter coefficients are:\n");
      for(i=0;i<(*windowsize);i++)
         printf("kernel[%d] = %f\n", i, (*kernel)[i]);
   }
   */
}

void printTime(const struct timeval* start,const struct timeval* stop, const char* msg)
{
   printf("%s", msg);
   printf("%li microseconds.\n", ((stop->tv_sec - start->tv_sec)*1000000L+stop->tv_usec) - start->tv_usec);
}
//<------------------------- end canny_edge.c ------------------------->

//<------------------------- begin hysteresis.c ------------------------->
/*******************************************************************************
* FILE: hysteresis.c
* This code was re-written by Mike Heath from original code obtained indirectly
* from Michigan State University. heath@csee.usf.edu (Re-written in 1996).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "cuda_helper.h"

#include "hysteresis.h"

/*******************************************************************************
* PROCEDURE: follow_edges
* PURPOSE: This procedure edges is a recursive routine that traces edgs along
* all paths whose magnitude values remain above some specifyable lower
* threshhold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,
   int cols)
{
   short *tempmagptr;
   unsigned char *tempmapptr;
   int i;
   int x[8] = {1,1,0,-1,-1,-1,0,1},
       y[8] = {0,1,1,1,0,-1,-1,-1};

   for(i=0;i<8;i++){
      tempmapptr = edgemapptr - y[i]*cols + x[i];
      tempmagptr = edgemagptr - y[i]*cols + x[i];

      if((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval)){
         *tempmapptr = (unsigned char) EDGE;
         follow_edges(tempmapptr,tempmagptr, lowval, cols);
      }
   }
}

/*******************************************************************************
* PROCEDURE: apply_hysteresis
* PURPOSE: This routine finds edges that are above some high threshhold or
* are connected to a high pixel by a path of pixels greater than a low
* threshold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
    float tlow, float thigh, unsigned char *edge)
{
   int r, c, pos, numedges, highcount, lowthreshold, highthreshold,
       hist[32768];
   short int maximum_mag;

   /****************************************************************************
   * Initialize the edge map to possible edges everywhere the non-maximal
   * suppression suggested there could be an edge except for the border. At
   * the border we say there can not be an edge because it makes the
   * follow_edges algorithm more efficient to not worry about tracking an
   * edge off the side of the image.
   ****************************************************************************/
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
     if(nms[pos] == POSSIBLE_EDGE) edge[pos] = POSSIBLE_EDGE;
     else edge[pos] = NOEDGE;
      }
   }
   /****************************************************************************
   * Compute the histogram of the magnitude image. Then use the histogram to
   * compute hysteresis thresholds.
   ****************************************************************************/
   for(r=0;r<32768;r++) hist[r] = 0;
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
     if(edge[pos] == POSSIBLE_EDGE) hist[mag[pos]]++;
      }
   }

   /****************************************************************************
   * Compute the number of pixels that passed the nonmaximal suppression.
   ****************************************************************************/
   for(r=1,numedges=0;r<32768;r++){
      if(hist[r] != 0) maximum_mag = r;
      numedges += hist[r];
   }

   highcount = (int)(numedges * thigh + 0.5);

   /****************************************************************************
   * Compute the high threshold value as the (100 * thigh) percentage point
   * in the magnitude of the gradient histogram of all the pixels that passes
   * non-maximal suppression. Then calculate the low threshold as a fraction
   * of the computed high threshold value. John Canny said in his paper
   * "A Computational Approach to Edge Detection" that "The ratio of the
   * high to low threshold in the implementation is in the range two or three
   * to one." That means that in terms of this implementation, we should
   * choose tlow ~= 0.5 or 0.33333.
   ****************************************************************************/
   r = 1;
   numedges = hist[1];
   while((r<(maximum_mag-1)) && (numedges < highcount)){
      r++;
      numedges += hist[r];
   }
   highthreshold = r;
   lowthreshold = (int)(highthreshold * tlow + 0.5);

   if(VERBOSE){
      printf("The input low and high fractions of %f and %f computed to\n",
     tlow, thigh);
      printf("magnitude of the gradient threshold values of: %d %d\n",
     lowthreshold, highthreshold);
   }

   /****************************************************************************
   * This loop looks for pixels above the highthreshold to locate edges and
   * then calls follow_edges to continue the edge.
   ****************************************************************************/
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
     if((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold)){
            edge[pos] = EDGE;
            follow_edges((edge+pos), (mag+pos), lowthreshold, cols);
     }
      }
   }

   /****************************************************************************
   * Set all the remaining possible edges to non-edges.
   ****************************************************************************/
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++) if(edge[pos] != EDGE) edge[pos] = NOEDGE;
   }
}

__global__
void cuda_zero_edges( unsigned char* result,
                      int rows,
                      int cols)
{
    const int rowcount = blockIdx.y * blockDim.y + threadIdx.y;
    const int colcount = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowcount == 0 || rowcount == rows - 1 || colcount == 0 || colcount == cols - 1 )
    {
        *(result + (rowcount*rows + colcount)) = (unsigned char) 0;
    }
}

void non_max_supp(  short *mag,
                    short *gradx,
                    short *grady,
                    int rows,
                    int cols,
                    unsigned char **result)
{

    printf("%s", "Alloc\n");
    gpuErrchk(cudaMalloc((void**) result, rows*cols * sizeof(unsigned char)));

   /****************************************************************************
   * Zero the edges of the result image.
   ****************************************************************************/

    if (VERBOSE)
    {
        printf("%s", "Zeros\n");
    }

    dim3 dgrid;
    dim3 dblock;
   
    get_dimgrid(&dgrid, cols, rows);
    get_dimblock(&dblock);

    cuda_zero_edges<<<dgrid, dblock>>>(*result, rows, cols);

    if (VERBOSE)
    {
        printf("%s", "NMS\n");
    }
    cuda_non_max_supp<<<dgrid, dblock>>>(mag, gradx, grady, rows, cols, *result);
}

__global__
void cuda_non_max_supp( short *mag,
                        short *gradx,
                        short *grady,
                        int nrows,
                        int ncols,
                        unsigned char *result)
{
    short z1,z2;
    short m00,gx,gy;
    float mag1,mag2,xperp,yperp;

    const int rowcount = blockIdx.y * blockDim.y + threadIdx.y;
    const int colcount = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowcount > 0 && colcount > 0 && rowcount < nrows - 2 && colcount < ncols - 2)
    {
        /****************************************************************************
        * Suppress non-maximum points.
        ****************************************************************************/
        const short* magptr = mag + rowcount*ncols + colcount;
        const short* gxptr = gradx + rowcount*ncols + colcount;
        const short* gyptr = grady + rowcount*ncols + colcount;
        unsigned char* resultptr = result + rowcount*ncols + colcount;

        if (((rowcount < nrows - 2) && (rowcount > 0)) &&
          ((colcount < ncols - 2) && (colcount > 0)))
        {
          m00 = *magptr;
          if(m00 == 0)
          {
              *resultptr = (unsigned char) NOEDGE;
          }
          else
          {
              xperp = -(gx = *gxptr)/((float)m00);
              yperp = (gy = *gyptr)/((float)m00);
          }
          if(gx >= 0)
          {
              if(gy >= 0)
              {
                  if (gx >= gy)
                  {
                      /* 111 */
                      /* Left point */
                      z1 = *(magptr - 1);
                      z2 = *(magptr - ncols - 1);

                      mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;

                      /* Right point */
                      z1 = *(magptr + 1);
                      z2 = *(magptr + ncols + 1);

                      mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                  }
                  else
                  {
                      /* 110 */
                      /* Left point */
                      z1 = *(magptr - ncols);
                      z2 = *(magptr - ncols - 1);

                      mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                      /* Right point */
                      z1 = *(magptr + ncols);
                      z2 = *(magptr + ncols + 1);

                      mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                  }
              }
              else
              {
                  if (gx >= -gy)
                  {
                      /* 101 */
                      /* Left point */
                      z1 = *(magptr - 1);
                      z2 = *(magptr + ncols - 1);

                      mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;

                      /* Right point */
                      z1 = *(magptr + 1);
                      z2 = *(magptr - ncols + 1);

                      mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                  }
                  else
                  {
                      /* 100 */
                      /* Left point */
                      z1 = *(magptr + ncols);
                      z2 = *(magptr + ncols - 1);

                      mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                      /* Right point */
                      z1 = *(magptr - ncols);
                      z2 = *(magptr - ncols + 1);

                      mag2 = (z1 - z2)*xperp  + (m00 - z1)*yperp;
                  }
              }
          }
          else
          {
              if ((gy = *gyptr) >= 0)
              {
                  if (-gx >= gy)
                  {
                      /* 011 */
                      /* Left point */
                      z1 = *(magptr + 1);
                      z2 = *(magptr - ncols + 1);

                      mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                      /* Right point */
                      z1 = *(magptr - 1);
                      z2 = *(magptr + ncols - 1);

                      mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                  }
                  else
                  {
                      /* 010 */
                      /* Left point */
                      z1 = *(magptr - ncols);
                      z2 = *(magptr - ncols + 1);

                      mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                      /* Right point */
                      z1 = *(magptr + ncols);
                      z2 = *(magptr + ncols - 1);

                      mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                  }
              }
              else
              {
                  if (-gx > -gy)
                  {
                      /* 001 */
                      /* Left point */
                      z1 = *(magptr + 1);
                      z2 = *(magptr + ncols + 1);

                      mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                      /* Right point */
                      z1 = *(magptr - 1);
                      z2 = *(magptr - ncols - 1);

                      mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                  }
                  else
                  {
                      /* 000 */
                      /* Left point */
                      z1 = *(magptr + ncols);
                      z2 = *(magptr + ncols + 1);

                      mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                      /* Right point */
                      z1 = *(magptr - ncols);
                      z2 = *(magptr - ncols - 1);

                      mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                  }
              }
          }

          /* Now determine if the current point is a maximum point */

          if ((mag1 > 0.0) || (mag2 > 0.0))
          {
              *resultptr = (unsigned char) NOEDGE;
          }
          else
          {
              if (mag2 == 0.0)
              *resultptr = (unsigned char) NOEDGE;
              else
              *resultptr = (unsigned char) POSSIBLE_EDGE;
          }
        }
    }
}
//<------------------------- end hysteresis.c ------------------------->

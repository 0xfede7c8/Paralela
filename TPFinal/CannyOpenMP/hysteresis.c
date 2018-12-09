//<------------------------- begin hysteresis.c ------------------------->
/*******************************************************************************
* FILE: hysteresis.c
* This code was re-written by Mike Heath from original code obtained indirectly
* from Michigan State University. heath@csee.usf.edu (Re-written in 1996).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define VERBOSE 0

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

#ifndef NUM_THREADS
#define NUM_THREADS 6
#endif
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
   float thethresh;
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
    int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold,
        i, hist[32768], rr, cc;
    short int maximum_mag, sumpix;



    double tini, tfin;
    /****************************************************************************
    * Initialize the edge map to possible edges everywhere the non-maximal
    * suppression suggested there could be an edge except for the border. At
    * the border we say there can not be an edge because it makes the
    * follow_edges algorithm more efficient to not worry about tracking an
    * edge off the side of the image.
    ****************************************************************************/
    memcpy(edge, nms, rows*cols);
    tini = omp_get_wtime();
    for(r=0;r<32768;r++) hist[r] = 0;

    #pragma omp parallel for private(c, pos) reduction(+:hist)
    for(r=0;r<rows;r++){
        for(c=0;c<cols;c++){
            pos = r*cols+c;
            if(edge[pos] == POSSIBLE_EDGE) hist[mag[pos]]++;
        }
    }
    tfin = omp_get_wtime();
    printf("histograma ---------------------------------- %f\n", tfin-tini);

    /****************************************************************************
    * Compute the number of pixels that passed the nonmaximal suppression.
    ****************************************************************************/
    //insignificante para paralelizar (40us)
    tini = omp_get_wtime();
    for(r=1,numedges=0;r<32768;r++){
      if(hist[r] != 0) maximum_mag = r;
      numedges += hist[r];
    }

    highcount = (int)(numedges * thigh + 0.5);
    tfin = omp_get_wtime();
    printf("number of pixels ----------------------------- %f\n", tfin-tini);

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
    //insignificante para paralelizar (<1us)
    tini = omp_get_wtime();
    r = 1;
    numedges = hist[1];
    while((r<(maximum_mag-1)) && (numedges < highcount)){
      r++;
      numedges += hist[r];
    }
    highthreshold = r;
    lowthreshold = (int)(highthreshold * tlow + 0.5);
    tfin = omp_get_wtime();
    printf("compute thresholds ----------------------------- %f\n", tfin-tini);

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
    tini = omp_get_wtime();
    int limit;
    #pragma omp parallel for private(r, c, pos, limit)
    for(i=0; i<NUM_THREADS; i++){
        if(i == NUM_THREADS-1) limit = rows;
        else limit = (i+1)*(rows/NUM_THREADS);
        for(r=i*(rows/NUM_THREADS); r<limit; r++){
            for(c=0;c<cols;c++){
                pos = r*cols+c;
                if((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold)){
                    edge[pos] = EDGE;
                    follow_edges((edge+pos), (mag+pos), lowthreshold, cols);
                 }
            }
        }
    }

    tfin = omp_get_wtime();
    printf("follow_edges ----------------------------- %f\n", tfin-tini);

    /****************************************************************************
    * Set all the remaining possible edges to non-edges.
    ****************************************************************************/
    for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++) if(edge[pos] != EDGE) edge[pos] = NOEDGE;
    }
}

/*******************************************************************************
* PROCEDURE: non_max_supp
* PURPOSE: This routine applies non-maximal suppression to the magnitude of
* the gradient image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
    unsigned char *result) 
{
    int rowcount, colcount,count;
    short *magrowptr,*magptr;
    short *gxrowptr,*gxptr;
    short *gyrowptr,*gyptr,z1,z2;
    short m00,gx,gy;
    float mag1,mag2,xperp,yperp;
    unsigned char *resultrowptr, *resultptr;
    unsigned char *resultrowptr2, *resultptr2;


    /****************************************************************************
    * Zero the edges of the result image.
    ****************************************************************************/
    for(count=0,
        resultrowptr=result,
        resultptr=result+ncols*(nrows-1),
        resultptr2=result+ncols*(nrows-2);
        count<ncols; 
        resultptr++,resultrowptr++,resultptr2++,count++){
        *resultrowptr = *resultptr = *resultptr2 = (unsigned char) 255;
    }

    for(count=0,
        resultptr=result,
        resultrowptr=result+ncols-1,
        resultrowptr2=result+ncols-2;
        count<nrows;

        count++,
        
        resultptr+=ncols,
        resultrowptr+=ncols,
        resultrowptr2+=ncols){
        *resultptr = *resultrowptr = *resultrowptr2 = (unsigned char) 255;
    }

    //----------------------------------------------------------------------------
    // Suppress non-maximum points.
    //----------------------------------------------------------------------------
    int ncols_increment;
    #pragma omp parallel for private (magrowptr, gxrowptr, gyrowptr,\
                                      resultrowptr, colcount, magptr, gxptr,\
                                      gyptr, resultptr, m00, xperp, yperp,\
                                      gx, gy, z1, z2)
    for(rowcount=1; rowcount<nrows-2; rowcount++){   

      ncols_increment = rowcount*ncols;
      magrowptr = mag + ncols_increment + 1;
      gxrowptr = gradx + ncols_increment + 1;
      gyrowptr = grady + ncols_increment + 1;
      resultrowptr = result + ncols_increment + 1;


      for(colcount=1,magptr=magrowptr,gxptr=gxrowptr,gyptr=gyrowptr,
         resultptr=resultrowptr;colcount<ncols-2; 
         colcount++,magptr++,gxptr++,gyptr++,resultptr++){   
         m00 = *magptr;
         if(m00 == 0){
            *resultptr = (unsigned char) NOEDGE;
         }
         else{
            xperp = -(gx = *gxptr)/((float)m00);
            yperp = (gy = *gyptr)/((float)m00);
         }

         if(gx >= 0){
            if(gy >= 0){
                    if (gx >= gy)
                    {  
                        // 111 //
                        // Left point //
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);

                        mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                        
                        // Right point 
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                    }
                    else
                    {    
                        // 110 //
                        // Left point //
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                        // Right point //
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp; 
                    }
                }
                else
                {
                    if (gx >= -gy)
                    {
                        // 101 //
                        // Left point //
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;
            
                        // Right point //
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                    }
                    else
                    {    
                        // 100 //
                        // Left point //
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                        // Right point //
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
                        // 011 //
                        // Left point //
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                        // Right point //
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                    }
                    else
                    {
                        // 010 //
                        // Left point //
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                        // Right point //
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                    }
                }
                else
                {
                    if (-gx > -gy)
                    {
                        // 001 //
                        // Left point //
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                        // Right point //
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                    }
                    else
                    {
                        // 000 //
                        // Left point //
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                        // Right point //
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                    }
                }
            } 

            // Now determine if the current point is a maximum point //

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

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NPUNTOS 10000
 
int main( int argc, char *argv[] )
{
	int npuntos = NPUNTOS;
	int contador_circulo = 0;

    MPI_Init(&argc,&argv);

    int j;
    float x,y;
    for(j=0;j<NPUNTOS;j++){
    	x = (float)rand()/(float)(RAND_MAX);
    	y = (float)rand()/(float)(RAND_MAX);
    	if(sqrt(x*x+y*y)<=1.0){
    		contador_circulo++;
    	}

    float pi = 4.0*contador_circulo/NPUNTOS;
    printf("%s: %f","Pi",pi);

    }
 
    MPI_Finalize();
}
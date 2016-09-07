
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NPUNTOS 100000000
 
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

    }
 
    MPI_Finalize();

    float pi = 4.0*(float)contador_circulo/(float)NPUNTOS;
    printf("%s: %f\n","Pi",pi);
}
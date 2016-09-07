
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NPUNTOS 10000
 
int main( int argc, char *argv[] )
{
	int npuntos = NPUNTOS;
	int contador_circulo = 0;

    MPI_Init(&argc,&argv);

    int j;
    float x,y;
    for(j=0;j<NPUNTOS;j++){
    	x = rand()/RAND_MAX;
    	printf("%f\n",x);

    }
 
    MPI_Finalize();
}
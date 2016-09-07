#include <omp.h>
#include <stdio.h>
#include <string.h>
#define CHUNKSIZE 4096
#define N    1000000

main (int argc, char** argv)  
{

int i,j, chunk,tid;
float a[N];
float resultado = 0;
float parcial = 0;
double tini,tfin;

/* Some initializations */
for (i=0; i < N; i++){
  a[i] = 1.0;
  
}

if(argc>1){
  omp_set_num_threads(atoi(argv));
}

chunk = CHUNKSIZE;
tini=omp_get_wtime();
#pragma omp parallel for shared(a) reduction(+:resultado)
  for (i=0; i < N; i++){
   resultado += a[i];
  }

tfin=omp_get_wtime();

printf("Resultado: %f",resultado);
printf("\ndemoro %f\n",tfin-tini);

}

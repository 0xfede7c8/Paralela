#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define VECLEN 100000000
#define SUM 8

float a[VECLEN], b[VECLEN];
double sum, suma[SUM][64];
double sumalocal;

float dotprod ()
{
   int i,tid;

   tid = omp_get_thread_num();
#pragma omp for 
   for (i=0; i < VECLEN; i++)
   {
      tid = omp_get_thread_num();
      sum = sum + (a[i]*b[i]);
      if (i%10000==0) {
         printf("  tid= %d i=%d\n",tid,i);
      }
   }
}


int main (int argc, char *argv[]) {
    int i,nth,tid;
    double tini,tfin;

    if (argc>1){
       nth=atoi(argv[1]);
    } else {
       nth=1;
    }

    omp_set_num_threads(nth);
    for (i=0;i<SUM;i++)
        suma[i][0]=0.0;

    for (i=0; i < VECLEN; i++)
       a[i] = b[i] = 1.0 ;
    sum = 0.0;
    sumalocal = 0.0;
    tini=omp_get_wtime();

#pragma omp parallel default(shared) private(tid,i)
    { 
       tid = omp_get_thread_num();
       printf(" entro a dotprod tid= %d \n",tid);
#pragma omp for reduction(-:sum)
 for (i=0; i < VECLEN; i++)
   {
      sum = sum + (a[i]*b[i]);
   }

//printf("Sum = %f\n",sum);
/*
//       dotprod();
#pragma omp for 
       for (i=0; i < VECLEN; i++)
       {
//          tid = omp_get_thread_num();
//#pragma omp critical
//          {
            sumalocal= sumalocal + (a[i]*b[i]);
//          sum = sum + (a[i]*b[i]);
//          suma[tid][0] = suma[tid][0] + (a[i]*b[i]);
//          }
          if (i%1000000==0) {
             printf("  tid= %d i=%d\n",tid,i);
          }
       }
#pragma omp critical
{ 
           sum+=sumalocal;
}
*/
    }
    
    tfin=omp_get_wtime();
/*
    sum = 0.0;
    for (i=0;i<nth;i++)
        sum+=suma[i][0];
*/
printf("Sum = %f\n",sum);
printf("tiempo = %f\n",tfin-tini);

//    sum = 0.0;

//#pragma omp for reduction(+:sum)
/*  
 for (i=0; i < VECLEN; i++)
   {
      sum = sum + (a[i]*b[i]);
   }
*/
//printf("Sum = %f\n",sum);
}

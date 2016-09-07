#include <omp.h>
#include <stdio.h>
#define N     50000000

main ()
{

int i;
float a[N], b[N], c[N], d[N],e[N];
int tid;

/* Some initializations */
for (i=0; i < N; i++) {
  a[i] = i * 1.5;
  b[i] = i + 22.35;
  }

#pragma omp parallel shared(a,b,c,d) private(i,tid)
  {
  tid=omp_get_thread_num();
  #pragma omp sections nowait
    {

    #pragma omp section
    {
    printf("estoy en hilo %d\n",tid);
    for (i=0; i < N; i++)
      c[i] = a[i] + b[i];
    }

    #pragma omp section
    {
    printf("estoy en hilo %d\n",tid);
    for (i=0; i < N; i++)
      e[i] = a[i] + b[i];
    }

    #pragma omp section
    {
    printf("estoy en hilo %d\n",tid);
    for (i=0; i < N; i++)
      d[i] = a[i] * b[i];
    }

    }  /* end of sections */

  }  /* end of parallel section */

}

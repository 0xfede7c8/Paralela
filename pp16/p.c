//#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, tid;
nthreads=88;

omp_set_num_threads(12);

/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel default(shared) private(tid) 
  {

  /* Obtain thread number */
  tid = omp_get_thread_num();
  printf("Hello World from thread = %d\n", tid);

  /* Only master thread does this */
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("From %d, Number of threads = %d\n", tid,nthreads);
    }

  }  /* All threads join master thread and disband */

  printf("Number of threads = %d\n", nthreads);

}

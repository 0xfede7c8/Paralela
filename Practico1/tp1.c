#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define VECLEN 1000000

void prefixsum_inplace(float *x, int N) {
    float *suma;
    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        #pragma omp single
        {
            suma = new float[nthreads+1];
            suma[0] = 0;
        }
        float sum = 0;
        #pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            sum += x[i];
            x[i] = sum;
        }
        suma[ithread+1] = sum;
        #pragma omp barrier
        float offset = 0;
        for(int i=0; i<(ithread+1); i++) {
            offset += suma[i];
        }
        #pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            x[i] += offset;
        }
    }
    delete[] suma;
}

int main(int argc, char** argv) {

    int largo;
    double tini,tfin;
    if(argc>1){
        printf("Argumento. \n");
        largo = atoi(argv[1]);
    }else{largo = VECLEN;}

    const int n = largo;
    float x[n];
    for(int i=0; i<n; i++) x[i] = 1.0*i;

    tini=omp_get_wtime();
    prefixsum_inplace(x, n);
    tfin=omp_get_wtime();

   // for(int i=0; i<n; i++) printf("%f %f\n", x[i], 0.5*i*(i+1));

    printf("\ndemoro %f\n",tfin-tini);

}
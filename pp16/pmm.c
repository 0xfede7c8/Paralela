#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define VECLEN 1000

float a[VECLEN][VECLEN], b[VECLEN][VECLEN],c[VECLEN][VECLEN];

int i,j,k;
double tini,tfin;

int test(){
   int ret=0;
   for (i=0;i<VECLEN;i++){
       for (j=0;j<VECLEN;j++){
           if (c[i][j]!=1.0*VECLEN)
              ret=1;
       }
    }
    return ret;
}

int main (int argc, char *argv[]) {
    int i,nth,tid;

    if (argc>1){
       nth=atoi(argv[1]);
    } else {
       nth=1;
    }

    omp_set_num_threads(nth);

for (i=0;i<VECLEN;i++){
   for (j=0;j<VECLEN;j++){
       a[i][j]=1.0;
       b[i][j]=1.0;
       c[i][j]=0.0;
   }
}
tini=omp_get_wtime();
#pragma omp parallel default(shared) private(i,j,k)
{ 
       tid = omp_get_thread_num();
#pragma omp for 
for (i=0;i<VECLEN;i++){
//    for (j=0;j<VECLEN;j++){
     for (k=0;k<VECLEN;k++){
        for (j=0;j<VECLEN;j++){
          c[i][j]+=a[i][k]*b[k][j];
      }
   }
}

}
tfin=omp_get_wtime();

printf("tiempo = %f\n",tfin-tini);
printf("test = %d\n",test());
}

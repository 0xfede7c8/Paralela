#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define VECLEN 2000
#define BLOCKDIV 20
#define BLOCKSIZ (VECLEN/BLOCKDIV)

float a[VECLEN][VECLEN], b[VECLEN][VECLEN],c[VECLEN][VECLEN];

int i,j,k;
int ii,jj,kk;
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
for (i=0;i<VECLEN;i+=BLOCKSIZ){
   for (k=0;k<VECLEN;k+=BLOCKSIZ){
      for (j=0;j<VECLEN;j+=BLOCKSIZ){
         for (ii=0;ii<BLOCKSIZ;ii++){
            for (kk=0;kk<BLOCKSIZ;kk++){
#pragma omp simd linear(jj:1)
               for (jj=0;jj<BLOCKSIZ;jj++){
                  c[i+ii][j+jj]+=a[i+ii][k+kk]*b[k+kk][j+jj];
               }
            }
         }
      }
   }
}

}
tfin=omp_get_wtime();

printf("tiempo = %f\n",tfin-tini);
printf("test   = %d\n",test());
}

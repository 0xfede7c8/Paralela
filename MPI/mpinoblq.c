#include "mpi.h"
#include <stdio.h>
#define PROCS 4
#define DATALEN (PROCS * 4)


int main(int argc, char *argv[])  {
int numtasks, rank, next, prev, bufi[4], bufd[4], tag1=1, tag2=2;
int data[DATALEN];
int g,i,j,k;
int cual,flag,flag2;

MPI_Request reqs[PROCS];
MPI_Status stat;
MPI_Status stats[PROCS];
int flags[PROCS];

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

for (i=0;i<DATALEN;i++){
    data[i]=rank*DATALEN+i;
}

prev = rank-1;
next = rank+1;
if (rank == 0)  prev = numtasks - 1;
if (rank == (numtasks - 1))  next = 0;

MPI_Irecv(bufi, 4, MPI_INT, prev, tag1, MPI_COMM_WORLD, &reqs[0]);
MPI_Irecv(bufd, 4, MPI_INT, next, tag2, MPI_COMM_WORLD, &reqs[1]);
k = 0;
MPI_Isend(&data[k], 4, MPI_INT, prev, tag2, MPI_COMM_WORLD, &reqs[2]);
MPI_Isend(&data[k], 4, MPI_INT, next, tag1, MPI_COMM_WORLD, &reqs[3]);
  
//      {  do some work  }

//MPI_Waitall(4, reqs, stats);
flag=0;
flag2=0;
for (j=0;j<4;j++){
   flags[j]=0;
}
while (!flag2){
   MPI_Testall(4,reqs,&flag2,stats);
   for (j=0;j<4;j++){
      MPI_Test(&reqs[j],&flag,&stat);
      if (flag && !flags[j]){
         printf("Termino la reqs %d en rank %d\n",j,rank);
         flags[j]=1;
         if (j==0) { 
             for (g=0;g<4;g++){
                printf("en la iteracion %d del nodo %d llego en la pos %d:%d\n",k,rank,g,bufi[g]);
             }
         }
         if (j==1) { 
             for (g=0;g<4;g++){
                printf("en la iteracion %d del nodo %d llego en la pos %d:%d\n",k,rank,g,bufd[g]);
             }
         }
      }
   }
//   MPI_Testall(4,reqs,&flag,stats);
}
/*
for (j=0;j<4;j++){
   printf("en la iteracion %d del nodo %d llego en la pos %d:%d,%d\n",k,rank,j,bufi[j],bufd[j]);
}
*/
MPI_Finalize();

return 0;

}


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECLEN 1000000

void imprimirVector(long int vec[]);

int largo;

int main (int argc, char** argv) {

  clock_t tin,tfin; 

  

 if(argc>1){
    printf("Argumento. \n");
    largo = atoi(argv[1]);
  }else{
    largo = VECLEN;
    
  }

  long int vec[largo];

  int i;
  for(i=0;i<largo;i++){
    vec[i] = i;
  }

  tin = clock();

  long int aux = 0;
  for(i=0;i<largo;i++){
    aux+=vec[i];
    vec[i]=aux;
  }

  tfin = clock();

  double elapsed_time = (tfin -tin)/(double)CLOCKS_PER_SEC;

 // imprimirVector(vec);
  printf("tiempo = %.2f.\n",elapsed_time);

}

void imprimirVector(long int vec[]){
  int i;
  printf("%s\n","Vector: ");
  printf("%s","[");
  for(i=0; i<largo;i++){
    printf(" %li ",vec[i]);
  }
  printf("%s\n","]");

}
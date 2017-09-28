#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#define WIDTH 1200
#define HEIGHT 1200

typedef int_fast64_t fint;

fint* mult(fint* a, fint* b, fint* c);

int main(void)
{
	fint* a = (fint*)malloc(sizeof(fint)*HEIGHT*WIDTH);
	fint* b = (fint*)malloc(sizeof(fint)*HEIGHT*WIDTH);
	fint* c = (fint*)malloc(sizeof(fint)*HEIGHT*WIDTH);
	int i=0;
	int j=0;

	struct timeval stop, start;

	gettimeofday(&start, NULL);
	for(i=0; i<WIDTH; i++)
		for(j=0; j<HEIGHT; j++)
		{
			a[i*WIDTH + j] = 1+i+j;
			b[i*WIDTH + j] = 1+i+j;
			c[i*WIDTH + j] = 0;
		}
	gettimeofday(&stop, NULL);
	printf("Load took: %ld\n", stop.tv_usec - start.tv_usec);
	mult(a,b,c);
	free(a);
	free(b);
	free(c);
	return 0;
}

fint* mult(fint* a, fint* b, fint* c)
{
	struct timeval stop, start;
	int i=0;
	int j=0;
	int k=0;

	gettimeofday(&start, NULL);

	for(j=0; j<HEIGHT; j++)
	{
		for(i=0; i<WIDTH; i++)
		{
			for(k=0; k<HEIGHT; k++)
			{
				c[i*WIDTH + j] += a[k*WIDTH + j]*b[i*WIDTH + k];
			} 
		}
	}

	gettimeofday(&stop, NULL);

	printf("Time in microseconds: %ld microseconds\n",
            ((stop.tv_sec - start.tv_sec)*1000000L
           +stop.tv_usec) - start.tv_usec
            );
	return c;
}
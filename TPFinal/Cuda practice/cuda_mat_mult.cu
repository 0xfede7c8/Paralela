#include <stdint.h>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

#define WIDTH 2048
#define HEIGHT 2048

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

using namespace std;

void MatrixMultiplication(const uint32_t* const a, const uint32_t* const b, uint32_t* const c);

void printMat(uint32_t* P);

void printTime(const struct timeval& start,const struct timeval& stop, string msg);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__
void cudaMatrixMultiplication(uint32_t* Md, uint32_t* Nd, uint32_t* Pd, int width);

int main(int argc, char** argv)
{
	cout << "Usage: " << argv[0] << "<-p>\n\n";
	
	bool printset = false;
	if(argc > 1) 
	{
		if(strcmp(argv[1], "-p") == 0) 
		{
			cout << "Print setted ON.\n";
			printset = true;
		}
	}
	

	int matsize_b = WIDTH*HEIGHT*sizeof(uint32_t);

	uint32_t* const Mh = (uint32_t*)malloc(matsize_b);
	uint32_t* const Nh = (uint32_t*)malloc(matsize_b);
	uint32_t* const Ph = (uint32_t*)malloc(matsize_b);

	struct timeval stop, start;
	unsigned int i,j;	

	gettimeofday(&start, NULL);
	for(i=0; i<WIDTH; i++)
		for(j=0; j<HEIGHT; j++)
		{
			Mh[i*WIDTH + j] = 1+i+j;
			Nh[i*WIDTH + j] = 1+i+j;
			Ph[i*WIDTH + j] = 0;
		}
	gettimeofday(&stop, NULL);
	printTime(start, stop, "Load took: ");
	
	if(printset)
	{
		cout << "Mh:\n";
		printMat(Mh);
		cout << "Nh:\n";
		printMat(Nh);
		cout << "Ph:\n";
		printMat(Ph);
	}

	uint32_t* Md;
	uint32_t* Nd;
	uint32_t* Pd;

	//CUDA OPERATIONS
	gettimeofday(&start, NULL);
	
	gpuErrchk(cudaMalloc((void**) &Md, matsize_b));
	gpuErrchk(cudaMalloc((void**) &Nd, matsize_b));
	gpuErrchk(cudaMalloc((void**) &Pd, matsize_b));
	gpuErrchk(cudaMemcpy(Md, Mh, matsize_b, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(Nd, Nh, matsize_b, cudaMemcpyHostToDevice));
	dim3 dimGrid(WIDTH/TILE_WIDTH, HEIGHT/TILE_HEIGHT, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
	cudaMatrixMultiplication<<<dimGrid, dimBlock>>>(Md, Nd, Pd, WIDTH);
	gpuErrchk(cudaMemcpy(Ph, Pd, matsize_b, cudaMemcpyDeviceToHost));
	gettimeofday(&stop, NULL);
	printTime(start, stop, "Cuda took: ");
	
	if(printset)
	{
		cout << "Ph:\n";
		printMat(Ph);	
	} 

	memset(Ph, 0, matsize_b);

	//MATMULT OPERATIONS
	gettimeofday(&start, NULL);
	MatrixMultiplication(Mh, Nh, Ph);
	gettimeofday(&stop, NULL);
	printTime(start, stop, "Mult took: ");
	
	if(printset)
	{
		cout << "Ph:\n";
		printMat(Ph);	
	} 

	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);

	free(Mh);
	free(Nh);
	free(Ph);

}

void printTime(const struct timeval& start,const struct timeval& stop, string msg)
{
	cout << msg << ((stop.tv_sec - start.tv_sec)*1000000L+stop.tv_usec) - start.tv_usec << " microseconds.\n";
}

void printMat(uint32_t* P)
{
	for(int i=0; i<WIDTH; i++)
	{
		if(i!=0)
			printf("\n");
		for(int j=0; j<WIDTH; j++)
		{
			printf("%u ", P[i*WIDTH + j]);
		}
	}
	printf("\n");
}

__global__
void cudaMatrixMultiplication(uint32_t* Md, uint32_t* Nd, uint32_t* Pd, const int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;

	uint32_t Pvalue = 0;
	for(int k=0; k<width; k++)
	{
		Pvalue += Md[row*width + k] * Nd[k*width + col];
	}
	Pd[row*width + col] = Pvalue;
}


void MatrixMultiplication(const uint32_t* const a,const uint32_t* const b, uint32_t* const c)
{
	int i=0;
	int j=0;
	int k=0;

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
}

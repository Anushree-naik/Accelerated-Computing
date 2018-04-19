#include<stdio.h>
#include<stdlib.h>
#include<math.h>

__global__ void vecAdd(float* h_a, float* h_b, float* h_c, int n)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	//check if it is in bound
	if(id<n)
		h_c[id] = h_a[id]+ h_b[id];

}

int main(int argc, char* argv[])
{
	//size of vectors
	int n= 1000;
	
	float *h_a;//ip
	float *h_b;//ip
	float *h_c;//op
	
	float *d_a;//ip
	float *d_b;//ip
	float *d_c;//op

	int size = n * sizeof(float);		
	
	//allocating memory on host
	h_a = (float*)malloc(size);
	h_b = (float*)malloc(size);
	h_c = (float*)malloc(size);	
	
	//allocating memory for each vector on GPU
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);
	
	//initialize vectors on host
	int i;
	for(i = 0; i<n; i++)
	{
		h_a[i] = sin(i)*sin(i);
		h_b[i] = cos(i)*cos(i);
	}

	/*printf("h_a: \n");
	for(i=0; i<n; i++)
		printf("%.1f\n", h_a[i]);
	printf("\n");

	printf("h_b: \n");
	for(i=0; i<n; i++)
		printf("%.1f\n", h_b[i]);
	printf("\n");
	*/

	//copy host vectors to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	
	int threadPerBlocks, blockCount;
	
	//block size
	threadPerBlocks = 1024;
	
	//grid size
	blockCount = (int)ceil((float)n/threadPerBlocks);

	//executing kernel 
	vecAdd<<<threadPerBlocks, blockCount>>>(d_a, d_b, d_c, n);
	
	//copy array back to host
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	
	float sum = 0;
	for(i=0; i<n; i++)
		sum += h_c[i];
	printf("Final result is: %f\n", sum/n);
	
	//release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//releasing host memory
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
	
	
}

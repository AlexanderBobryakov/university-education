#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctime>

#define CSC(call) do { 			\
	cudaError_t res = call;		\
	if (res != cudaSuccess) {	\
		fprintf(stderr, "CUDA Error in %s:%d: %s", __FILE__, 	\
					__LINE__, cudaGetErrorString(res));			\
		exit(0);				\
	}							\
} while (0)

__global__ void kernel(double *dev_arr1, double *dev_arr2, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;		// Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;					// Общее кол-во потоков
    for (int i = idx; i < n; i += offset) {
        dev_arr1[i] = dev_arr1[i] + dev_arr2[i];
    }
}

int main() {
    int n = 10000000;
//    scanf("%d", &n);

    double *vector1 = (double *)malloc(sizeof(double) * n);
    double *vector2 = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++)	{
//        scanf("%lf", &vector1[i]);
        vector1[i] = i;
        vector2[i] = i;
    }
    for(int i = 0; i < n; i++)	{
//        scanf("%lf", &vector2[i]);
    }

    double *dev_arr1;
    cudaMalloc(&dev_arr1, sizeof(double) * n);		// Выделение памяти под массив на GPU
    cudaMemcpy(dev_arr1, vector1, sizeof(double) * n, cudaMemcpyHostToDevice);	// Копирование данных с CPU на GPU
    double *dev_arr2;
    cudaMalloc(&dev_arr2, sizeof(double) * n);		// Выделение памяти под массив на GPU
    cudaMemcpy(dev_arr2, vector2, sizeof(double) * n, cudaMemcpyHostToDevice);	// Копирование данных с CPU на GPU

    cudaEvent_t start, stop;
    float t;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start, 0));

    kernel<<<128, 512>>>(dev_arr1, dev_arr2, n);

    CSC(cudaGetLastError());
    CSC(cudaEventRecord(stop, 0));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaEventElapsedTime(&t, start, stop));
    printf("time = %f\n", t);
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    double *result = (double *)malloc(sizeof(double) * n);
    cudaMemcpy(result, dev_arr1, sizeof(double) * n, cudaMemcpyDeviceToHost);	// Копирование данных с GPU на CPU
    cudaFree(dev_arr1);
    cudaFree(dev_arr2);

    clock_t t1 = clock();
    for(int i = 0; i < n; i++)	{
        vector1[i] = vector1[i] + vector2[i];
    }
    clock_t t2 = clock();
    printf("%f", (t2 - t1 + .0) / CLOCKS_PER_SEC);

//    for(int i = 0; i < n; i++)	{
//        printf("%.10e ", result[i]);
//    }

    return 0;
}

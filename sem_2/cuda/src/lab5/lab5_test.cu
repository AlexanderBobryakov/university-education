#include <fstream>
#include <iostream>
#include <algorithm>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define GRID_SIZE 8
#define BLOCK_SIZE 256
#define LOG_NUM_BANKS 5
#define MAX_SIZE 135e6
#define LOG_SIZE 100

#define CONFLICT_FREE_OFFS(i) ((i) >> LOG_NUM_BANKS)
// #define CONFLICT_FREE_OFFS(x) (x + (x >> LOG_NUM_BANKS))

typedef int tdat;

void printAnswer(tdat *nums, int size) {
    freopen(NULL, "wb", stdout);
    fwrite(nums, sizeof(tdat), size, stdout);
    fclose(stdout);
}

__global__ void clearArray(int *array, int size) {
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = blockDim.x * gridDim.x;

    for (int i = idX; i < size; i += offsetX) {
        array[i] = 0;
    }

    __syncthreads();
}

__global__ void histogramSimpleKernel(tdat* numsCuda, int size, int *histArray, int histSize) {
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = blockDim.x * gridDim.x;

    for (int i = idX; i < size; i += offsetX) {
        atomicAdd(&(histArray[numsCuda[i]]), 1);
    }
    
    __syncthreads();
}

int nextPower(int i) {
    int power = 1;
    while (power <= i) {
        power *= 2;
    }
    return power;
}

__global__ void scanKernel(int *histArray, int *resCuda, int *sums, int size) {
    __shared__ int tmp[2 * BLOCK_SIZE + CONFLICT_FREE_OFFS(2 * BLOCK_SIZE)];

    int tId = threadIdx.x;
    int offset = 1;
    int ai = tId;
    int bi = tId + (size / 2);
    int offsetA = CONFLICT_FREE_OFFS(ai);
    int offsetB = CONFLICT_FREE_OFFS(bi);

    tmp[ai + offsetA] = histArray[ai + 2 * BLOCK_SIZE * blockIdx.x];
    tmp[bi + offsetB] = histArray[bi + 2 * BLOCK_SIZE * blockIdx.x];

    for (int d = size >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (tId < d) {
            int a_i = offset * (2 * tId + 1) - 1;
            int b_i = offset * (2 * tId + 2) - 1;

            a_i += CONFLICT_FREE_OFFS(a_i);
            b_i += CONFLICT_FREE_OFFS(b_i);

            tmp[b_i] += tmp[a_i];
        }

        offset <<= 1;
    }

    if (tId == 0) {
        int i = size - 1 + CONFLICT_FREE_OFFS(size - 1);

        sums[blockIdx.x] = tmp[i];
        tmp[i] = 0;
    }

    for (int d = 1; d < size; d <<= 1) {
        offset >>= 1;

        __syncthreads();

        if (tId < d) {
            int a_i = offset * (2 * tId + 1) - 1;
            int b_i = offset * (2 * tId + 2) - 1;
            int t;

            a_i += CONFLICT_FREE_OFFS(a_i);
            b_i += CONFLICT_FREE_OFFS(b_i);

            t = tmp[a_i];
            tmp[a_i] = tmp[b_i];
            tmp[b_i] += t;
        }
    }

    __syncthreads();

    resCuda[ai + 2 * BLOCK_SIZE * blockIdx.x] = tmp[ai + offsetA];
    resCuda[bi + 2 * BLOCK_SIZE * blockIdx.x] = tmp[bi + offsetB];
}

__global__ void scanDistributeKernel(int *resCuda, int *resSums) {
    resCuda[threadIdx.x + blockIdx.x * 2 * BLOCK_SIZE] += resSums[blockIdx.x];
}

void scan(int *histArray, int *countResCuda, int size) {
    int blockCnt = size / (2 * BLOCK_SIZE);
    int *blockSums = NULL;
    int *resSums = NULL;

    if (blockCnt < 1) {
        blockCnt = 1;
    }

    CSC(cudaMalloc((void **)&blockSums, blockCnt * sizeof(int)));
    CSC(cudaMalloc((void **)&resSums, blockCnt * sizeof(int)));

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks(blockCnt, 1, 1);

    scanKernel<<<blocks, threads>>>(histArray, countResCuda, blockSums, 2 * BLOCK_SIZE);
    CSC(cudaGetLastError());

    if (size >= 2 * BLOCK_SIZE) {
        scan(blockSums, resSums, blockCnt);
    } else {
        CSC(cudaMemcpy(resSums, blockSums, blockCnt * sizeof(int),
            cudaMemcpyDeviceToDevice));
    }

    threads = dim3(2 * BLOCK_SIZE, 1, 1);
    // blocks = dim3(blockCnt - 1, 1, 1);
    if (blockCnt == 1) {
        blocks = dim3(blockCnt, 1, 1);
    } else {
        blocks = dim3(blockCnt - 1, 1, 1);
    }

    scanDistributeKernel<<<blocks, threads>>>(countResCuda + 2 * BLOCK_SIZE, resSums + 1);
    CSC(cudaGetLastError());

    cudaFree(blockSums);
    cudaFree(resSums);
}

__global__ void calcRes(tdat *numsCuda, tdat *resCuda, int *sums, int size) {
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    
	for (int i = tId; i < size; i += offset) {
        resCuda[atomicAdd(&sums[numsCuda[i] + 1], -1) - 1] = numsCuda[i];
    }
}

int main(int argc, char **argv) {
    int size = 135000000;

    //std::freopen(NULL, "rb", stdin);
    //std::fread(&size, sizeof(int), 1, stdin);

    tdat *nums = (tdat *)malloc(sizeof(tdat) * size);
    //std::fread(nums, sizeof(tdat), size, stdin);
    //std::fclose(stdin);
    
    for(int i = 0; i < size; ++i)
    {
        nums[i] = static_cast<tdat>((size - i) % 16000000);
    }

    
    int histSize = (1 << 25) - 1;
    
    printf("size = %d, ", size);
    printf("histSize = %d\n", histSize);    
    
    tdat *numsCuda;
    CSC(cudaMalloc(&numsCuda, sizeof(tdat) * size));
    CSC(cudaMemcpy(numsCuda, nums, sizeof(tdat) * size,
        cudaMemcpyHostToDevice));
    
    int *histArray;
    CSC(cudaMalloc(&histArray, sizeof(int) * histSize));

    int *countResCuda;
    CSC(cudaMalloc(&countResCuda, sizeof(int) * histSize));

    tdat *resCuda;
    CSC(cudaMalloc(&resCuda, sizeof(tdat) * size));

    clearArray<<<GRID_SIZE, BLOCK_SIZE>>>(histArray, histSize);
    CSC(cudaGetLastError());
    
    float timeHist = 0;
    float timeScan = 0;
    float timeCalcRes = 0;

    {
        cudaEvent_t start, stop;
        
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start, 0));
        
        histogramSimpleKernel<<<GRID_SIZE, BLOCK_SIZE>>>(numsCuda, size, histArray, histSize);
        CSC(cudaGetLastError());
    
        CSC(cudaGetLastError());

        CSC(cudaEventRecord(stop, 0));

        CSC(cudaEventSynchronize(stop));

        CSC(cudaEventElapsedTime(&timeHist, start, stop));
        
        printf("timeHist = %f\n", timeHist);
        
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));
    }

    {
        cudaEvent_t start, stop;
        
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start, 0));
        
        scan(histArray, countResCuda, histSize);
        
        CSC(cudaEventRecord(stop, 0));

        CSC(cudaEventSynchronize(stop));

        CSC(cudaEventElapsedTime(&timeScan, start, stop));
        
        printf("timeScan = %f\n", timeScan);
        
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));
    }
    
    {
        cudaEvent_t start, stop;
        
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start, 0));
        
        calcRes<<<GRID_SIZE, BLOCK_SIZE>>>(numsCuda, resCuda, countResCuda, size);
        CSC(cudaGetLastError());
        
        CSC(cudaEventRecord(stop, 0));

        CSC(cudaEventSynchronize(stop));

        CSC(cudaEventElapsedTime(&timeCalcRes, start, stop));
        
        printf("timeCalcRes = %f\n", timeCalcRes);
        
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));
    }
    
    
    cudaMemcpy(nums, resCuda, sizeof(tdat) * size,
        cudaMemcpyDeviceToHost);

    cudaFree(numsCuda);
    cudaFree(histArray);
    cudaFree(resCuda);
    cudaFree(countResCuda);
    
    free(nums);

    return 0;
}

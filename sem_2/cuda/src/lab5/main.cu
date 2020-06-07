#include <fstream>
#include <iostream>
#include <algorithm>

#define CONFLICT_FREE_OFFS(i) ((i) >> 5)

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)

__global__ void histogram_kernel(int *nums, int size, int *histogram) {
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = blockDim.x * gridDim.x;
    for (int i = idX; i < size; i += offsetX) {
        atomicAdd(&(histogram[nums[i]]), 1);
    }
    __syncthreads();
}

__global__ void scan_kernel(int *histogram, int *result, int *sums, int size) {
    __shared__ int tmp[2 * 256 + CONFLICT_FREE_OFFS(2 * 256)];
    int threadId = threadIdx.x;
    int offset = 1;
    int ai = threadId;
    int bi = threadId + (size / 2);
    int offsetA = CONFLICT_FREE_OFFS(ai);
    int offsetB = CONFLICT_FREE_OFFS(bi);
    tmp[ai + offsetA] = histogram[ai + 2 * 256 * blockIdx.x];
    tmp[bi + offsetB] = histogram[bi + 2 * 256 * blockIdx.x];
    for (int j = size >> 1; j > 0; j >>= 1) {
        __syncthreads();

        if (threadId < j) {
            int a_i = offset * (2 * threadId + 1) - 1;
            int b_i = offset * (2 * threadId + 2) - 1;

            a_i = a_i + CONFLICT_FREE_OFFS(a_i);
            b_i = b_i + CONFLICT_FREE_OFFS(b_i);

            tmp[b_i] = tmp[b_i] + tmp[a_i];
        }
        offset <<= 1;
    }
    if (threadId == 0) {
        int i = size + CONFLICT_FREE_OFFS(size - 1) - 1;
        sums[blockIdx.x] = tmp[i];
        tmp[i] = 0;
    }
    for (int jj = 1; jj < size; jj <<= 1) {
        offset >>= 1;
        __syncthreads();

        if (threadId < jj) {
            int a_i = offset * (2 * threadId + 1) - 1;
            int b_i = offset * (2 * threadId + 2) - 1;
            int t;

            a_i = a_i + CONFLICT_FREE_OFFS(a_i);
            b_i = b_i + CONFLICT_FREE_OFFS(b_i);

            t = tmp[a_i];
            tmp[a_i] = tmp[b_i];
            tmp[b_i] = tmp[b_i] + t;
        }
    }
    __syncthreads();
    result[ai + 2 * 256 * blockIdx.x] = tmp[ai + offsetA];
    result[bi + 2 * 256 * blockIdx.x] = tmp[bi + offsetB];
}

__global__ void array_clear_kernel(int *array, int size) {
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = blockDim.x * gridDim.x;

    for (int i = idX; i < size; i += offsetX) {
        array[i] = 0;
    }
    __syncthreads();
}

__global__ void check_kernel(int *result, int *sums) {
    result[threadIdx.x + blockIdx.x * 2 * 256] += sums[blockIdx.x];
}

__global__ void get_result_kernel(int *nums, int *result, int *sums, int size) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = threadId; i < size; i += offset) {
        result[atomicAdd(&sums[nums[i] + 1], -1) - 1] = nums[i];
    }
}

int getNext(int i) {
    int power = 1;
    while (power <= i) {
        power *= 2;
    }
    return power;
}

void scan(int *histArray, int *countResCuda, int size) {
    int blockCnt = size / (2 * 256);
    int *blockSums = NULL;
    int *resSums = NULL;

    if (blockCnt < 1) {
        blockCnt = 1;
    }

    CSC(cudaMalloc((void **) &blockSums, blockCnt * sizeof(int)));
    CSC(cudaMalloc((void **) &resSums, blockCnt * sizeof(int)));

    dim3 threads(256, 1, 1);
    dim3 blocks(blockCnt, 1, 1);

    scan_kernel<<<blocks, threads>>>(histArray, countResCuda, blockSums, 2 * 256);
    CSC(cudaGetLastError());

    if (size >= 2 * 256) {
        scan(blockSums, resSums, blockCnt);
    } else {
        CSC(cudaMemcpy(resSums, blockSums, blockCnt * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    threads = dim3(2 * 256, 1, 1);
    if (blockCnt == 1) {
        blocks = dim3(blockCnt, 1, 1);
    } else {
        blocks = dim3(blockCnt - 1, 1, 1);
    }

    check_kernel<<<blocks, threads>>>(countResCuda + 2 * 256, resSums + 1);
    CSC(cudaGetLastError());

    cudaFree(blockSums);
    cudaFree(resSums);
}

int main(int argc, char **argv) { // G:\Projects\CUDA\lab5\original.bin    G:\Projects\CUDA\lab5\result.bin
    /*int size;
    std::freopen(NULL, "rb", stdin);
    std::fread(&size, sizeof(int), 1, stdin);

    int *nums = (int *) malloc(sizeof(int) * size);
    std::fread(nums, sizeof(int), size, stdin);
    std::fclose(stdin);*/

    int size = 135000000;
    int n = 135000000;
    int i_n = 0;
    int *nums = (int *)malloc(sizeof(int ) * n);
    for(int i = 0; i < size; ++i)
    {
        nums[i] = static_cast<int>((size - i) % 16000000);
    }

    if (n == 1) {
       /* freopen(NULL, "wb", stdout);
        fwrite(nums, sizeof(int), size, stdout);
        fclose(stdout);*/
        return 0;
    }

    int max = *std::max_element(nums, nums + size);
    int histSize = getNext(max + 2) - 1;
    int min = *std::min_element(nums, nums + size);
    histSize = (1 << 25) - 1;

    /*fprintf(stderr, "size = %d, ", size);
    fprintf(stderr, "min = %d, ", min);
    fprintf(stderr, "max = %d, ", max);
    fprintf(stderr, "histSize = %d\n", histSize);*/

    int *numsCuda;
    CSC(cudaMalloc(&numsCuda, sizeof(int) * size));
    CSC(cudaMemcpy(numsCuda, nums, sizeof(int) * size, cudaMemcpyHostToDevice));

    int *histogramArray;
    CSC(cudaMalloc(&histogramArray, sizeof(int) * histSize));
    int *resultCounting;
    CSC(cudaMalloc(&resultCounting, sizeof(int) * histSize));

    int *result;
    CSC(cudaMalloc(&result, sizeof(int) * size));


    array_clear_kernel<<<8, 256>>>(histogramArray, histSize);
    CSC(cudaGetLastError());
    histogram_kernel<<<8, 256>>>(numsCuda, size, histogramArray);
    CSC(cudaGetLastError());
    scan(histogramArray, resultCounting, histSize);
    get_result_kernel<<<8, 256>>>(numsCuda, result, resultCounting, size);
    CSC(cudaGetLastError());
    cudaMemcpy(nums, result, sizeof(int) * size, cudaMemcpyDeviceToHost);


    /*freopen(NULL, "wb", stdout);
    fwrite(nums, sizeof(int), size, stdout);
    fclose(stdout);*/

    cudaFree(numsCuda);
    cudaFree(histogramArray);
    cudaFree(result);
    cudaFree(resultCounting);
    free(nums);

    return 0;
}

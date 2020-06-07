#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define GRID 16
#define BLOCKS 256

//__device__ int max_element_index;
//
//__global__ void max_in_array(int* array, int n, int x) {
//    int max_index = x;
//    double max_value = fabs(array[x * n + x]);
//    double current_value;
//
//    for (int i = x + 1; i < n; i++) {
//        current_value = fabs(array[i * n + x]);
//        if (current_value > max_value) {
//            max_index = i;
//            max_value = current_value;
//        }
//    }
//
//    max_element_index = max_index;
//}

__global__ void histo(int* dev_data, int size, int *dev_hist, int histSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = blockDim.x * gridDim.x;

    for (int i = idx; i < histSize; i += offsetX) {
        dev_hist[i] = 0;
    }

    for (int i = idx; i < size; i += offsetX) {
        atomicAdd(&(dev_hist[dev_data[i]]), 1);
    }

    __syncthreads();
}

__global__ void scan_dist(int *res, int *sums) {
    res[threadIdx.x + blockIdx.x * 2 * BLOCKS] += sums[blockIdx.x];
}

__global__ void scan_cuda(int *dev_hist, int *dev_res, int *sums, int n) {
    __shared__ int temp[2 * BLOCKS + ((2 * BLOCKS) >> 5)];

    int idx = threadIdx.x;
    int offset = 1;
    int idx_2 = idx;
    int idx_3 = idx + (n >> 1);
    int offsetA = (idx_2 >> 5);
    int offsetB = (idx_3 >> 5);

    temp[idx_2 + offsetA] = dev_hist[idx_2 + 2 * BLOCKS * blockIdx.x];
    temp[idx_3 + offsetB] = dev_hist[idx_3 + 2 * BLOCKS * blockIdx.x];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (idx < d) {
            int idx_2_i = (offset * (2 * idx + 1) - 1);
            int idx_3_i = (offset * (2 * idx + 2) - 1);
            idx_2_i +=  (idx_2_i >> 5);
            idx_3_i += (idx_3_i >> 5);

            temp[idx_3_i] += temp[idx_2_i];
        }

        offset <<= 1;
    }

    if (idx == 0) {
        int i = n - 1 + ((n -1) >> 5);
        sums[blockIdx.x] = temp[i];
        temp[i] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;

        __syncthreads();

        if (idx < d) {
            int idx_2_i = offset * (2 * idx + 1) - 1;
            int idx_3_i = offset * (2 * idx + 2) - 1;
            int tmp;
            idx_2_i += (idx_2_i >> 5);
            idx_3_i += (idx_3_i >> 5);
            tmp = temp[idx_2_i];
            temp[idx_2_i] = temp[idx_3_i];
            temp[idx_3_i] += tmp;
        }
    }

    __syncthreads();

    dev_res[idx_2 + 2 * BLOCKS * blockIdx.x] = temp[idx_2 + offsetA];
    dev_res[idx_3 + 2 * BLOCKS * blockIdx.x] = temp[idx_3 + offsetB];
}

__global__
void result_kernel(int *dev_data, int *dev_res, int *sums, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = idx; i < size; i += offset) {
        dev_res[atomicAdd(&sums[dev_data[i] + 1], -1) - 1] = dev_data[i];
    }
}

void scan_recursive(int *dev_hist, int *count, int n) {
    int *sums = nullptr;
    int *b_sums = nullptr;
    int blocks_cnt = n / (2 * BLOCKS);

    if (blocks_cnt < 1) {
        blocks_cnt = 1;
    }

    CSC(cudaMalloc((void **)&sums, blocks_cnt * sizeof(int)));
    CSC(cudaMalloc((void **)&b_sums, blocks_cnt * sizeof(int)));

    dim3 BLOCKS_INNER(blocks_cnt, 1, 1);
    dim3 THREADS(BLOCKS, 1, 1);

    scan_cuda<<<BLOCKS_INNER, THREADS>>>(dev_hist, count, b_sums, 2 * BLOCKS);
    CSC(cudaGetLastError());

    if (n >= 2 * BLOCKS) {
        scan_recursive(b_sums, sums, blocks_cnt);
    } else {
        CSC(cudaMemcpy(sums, b_sums, blocks_cnt * sizeof(int),
                       cudaMemcpyDeviceToDevice));
    }

    if (blocks_cnt == 1) {
        BLOCKS_INNER = dim3(blocks_cnt, 1, 1);
    } else {
        BLOCKS_INNER = dim3(blocks_cnt - 1, 1, 1);
    }

    THREADS = dim3(2 * BLOCKS, 1, 1);
    scan_dist<<<BLOCKS_INNER, THREADS>>>(count + 2 * BLOCKS, sums + 1);
    CSC(cudaGetLastError());

    cudaFree(b_sums);
    cudaFree(sums);
}

int main() {
    int n;

    std::freopen(NULL, "rb", stdin);
    std::fread(&n, sizeof(int), 1, stdin);

    if (n == 0) {
        return 0;
    }

    int *data = (int *)malloc(sizeof(int ) * n);
    std::fread(data, sizeof(int), n, stdin);
    std::fclose(stdin);

//    int i_n = 0;
//    n = 135000000;
//    int *data = (int *)malloc(sizeof(int ) * n);
//    for (int i = 0; i < n; i++){
//        data[i] = i_n;
//        i_n++;
//        if (i_n == 16777215){
//            i_n = 0;
//        }
//    }

//    n = 16777215;
//    int *data = (int *)malloc(sizeof(int ) * n);
//    for (int i = 0; i < n; i++){
//        data[i] = i;
//    }

    if (n == 1) {
        freopen(NULL, "wb", stdout);
        fwrite(data, sizeof(int), n, stdout);
        fclose(stdout);
        return 0;
    }

    int *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(int) * n));
    CSC(cudaMemcpy(dev_data, data, sizeof(int) * n, cudaMemcpyHostToDevice));

    int max = *std::max_element(data, data + n);
    int temp = 1;
    while (temp <= (max + 2)) {
        temp *= 2;
    }
    int hist_n = temp - 1;

    hist_n = (1 << 25) - 1;

    int *count;
    CSC(cudaMalloc(&count, sizeof(int) * hist_n));

    int *dev_hist;
    CSC(cudaMalloc(&dev_hist, sizeof(int) * hist_n));
//    CSC(cudaMemcpy(dev_hist, hist, sizeof(int) * n, cudaMemcpyHostToDevice));
    int *dev_res;
    CSC(cudaMalloc(&dev_res, sizeof(int) * n));

//    cudaEvent_t start, stop;
//    float t;
//    CSC(cudaEventCreate(&start));
//    CSC(cudaEventCreate(&stop));
//
//    CSC(cudaEventRecord(start, 0));

    histo<<<GRID, BLOCKS>>>(dev_data, n, dev_hist, hist_n);
    CSC(cudaGetLastError());

    scan_recursive(dev_hist, count, hist_n);

    result_kernel<<<GRID, BLOCKS>>>(dev_data, dev_res, count, n);
//    CSC(cudaGetLastError());
//
//    CSC(cudaEventRecord(stop, 0));
//    CSC(cudaEventSynchronize(stop));
//    CSC(cudaEventElapsedTime(&t, start, stop));
//    printf("time = %f\n", t);


//    CSC(cudaEventDestroy(start));
//    CSC(cudaEventDestroy(stop));

    CSC(cudaMemcpy(data, dev_res, sizeof(int) * n, cudaMemcpyDeviceToHost));
//    int tp;
//    for (int i = 0; i < n-1; i++) {
//        if (data[i+1] > data[i]){
//            tp = 1;
//        }
//        else {
//            tp = 0;
//        }
//    }
//    printf("%i", tp);
//    for (int i = 0; i < n; i++) {
//        printf("%i", data[i]);
//    }

    freopen(NULL, "wb", stdout);
    fwrite(data, sizeof(int), n, stdout);
    fclose(stdout);

    cudaFree(dev_res);
    cudaFree(count);
    cudaFree(dev_hist);
    cudaFree(dev_data);
    free(data);

    return 0;
}

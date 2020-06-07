#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <stdlib.h>
#include <math.h>
#include <stack>
#include <vector>
#include <iomanip>

#define N 256
#define ZERO 1e-7

#define CSC(call) do {              \
    cudaError_t res = call;         \
    if (res != cudaSuccess) {       \
        fprintf(stderr, "CUDA error %s:%d message: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(res));   \
        exit(0);                            \
    }                                       \
} while(0)

struct AbsMax {
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

__global__ void swapRows(double* A, int n, int m, int k, int i, int j) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while(idx < m + k) {
        int cnt = idx * n;
        int first  = i + cnt;
        int second = j + cnt;
        double tmp = A[first];
        A[first] = A[second];
        A[second] = tmp;
        idx += offset;
    }
}

__global__ void forward(double* A, int n, int m, int k, int i, int j) {
    int col = blockIdx.x;
    int row = threadIdx.x;
    int colOffset = gridDim.x;
    int rowOffset = blockDim.x;

    double a1 = A[n*j + i];
    for(col = blockIdx.x + j + 1; col < m + k; col += colOffset) {
        double cur = A[col*n + i];
        for(row = threadIdx.x + i + 1; row < n; row += rowOffset) {
            double l = A[n*j + row] / a1;
            A[col*n + row] -= l * cur;
        }
    }
}

__global__ void back(double* A, int n, int m, int k, int i, int j) {
    int col = blockIdx.x;
    int row = threadIdx.x;
    int colOffset = gridDim.x;
    int rowOffset = blockDim.x;

    double a1 = A[j*n + i];
    int start = m*n;
    for(col = blockIdx.x; col < k; col += colOffset) {
        double cur = A[start + col*n + i];
        for(row = threadIdx.x; row < i; row += rowOffset) {
            double l = A[j*n + row] / a1;
            A[start + col*n + row] -= l * cur;
        }
    }
}

int main() {
    int n,m;
    scanf("%d%d", &n, &m);

    int k = 1;
    double* A = (double*)malloc(n * (m+k)  * sizeof(double));

    double* A1 = (double*)malloc(n * (m+k) * sizeof(double));
    double* B1 = (double*)malloc(n * k * sizeof(double));
    double* B_test = (double*)malloc(n * k * sizeof(double));
    double* A_test = (double*)malloc(n * m * sizeof(double));

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            scanf("%lf", &A[j*n + i]);
            A_test[j*n + i] = A[j*n + i];
        }
    }
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < k; ++j) {
            scanf("%lf", &A[n*m + j*n + i]);
            B_test[j*n+i] = A[n*m + j*n + i];
        }
    }

    double* dev_A;
    cudaMalloc(&dev_A, sizeof(double) * n*(m+k));
    cudaMemcpy(dev_A, A, sizeof(double) * n*(m+k), cudaMemcpyHostToDevice);

    AbsMax comp;
    int cnt = 0;

    std::vector< int > stairs;
    for(int i = 0; i < m; ++i) {
        cudaDeviceSynchronize();
        thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(dev_A);
        int offset = i * n;
        thrust::device_ptr<double> res = thrust::max_element(p_arr + i*n + cnt,
                                                             p_arr + (i+1) * n,comp);
        int j = (int)(res - (p_arr + offset));
        double max = *res;
        cudaDeviceSynchronize();

        double cur = *(p_arr + i*n + cnt);
        if(cnt == (n - 1) && fabs(cur) < ZERO) {
            continue;
        }

        if(fabs(max) > ZERO) {
            if(cnt != j) {
                swapRows<<<N,N>>>(dev_A, n, m, k, cnt, j);
                CSC(cudaGetLastError());
            }

            cudaDeviceSynchronize();
            forward<<<N, N>>>(dev_A, n, m, k, cnt, i);
            CSC(cudaGetLastError());

            cudaDeviceSynchronize();
            stairs.push_back(cnt);
            stairs.push_back(i);
            cnt++;
            if(cnt >= n || cnt >= m) {
                break;
            }
        }

    }
    cudaDeviceSynchronize();
    for(int i = (int)stairs.size() - 2; i >= 0; i -= 2) {
        back<<<N, N>>>(dev_A, n, m, k, stairs[i], stairs[i + 1]);
    }
    CSC(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(A1, dev_A, sizeof(double) * n*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(B1, dev_A + n*m, sizeof(double) * n*k, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    double* res = (double*)malloc(sizeof(double) * m*k);

    int cnt1 = 0;
    for(int z = 0; z < (int)stairs.size(); z += 2) {
        int i = stairs[z];
        int j = stairs[z + 1];
        while(cnt1 < j) {
            for(int x = 0; x < k; ++x) {
                printf("0.0000000000e+00 ");
            }
            printf("\n");
            cnt1++;
        }

        for(int x = 0; x < k; ++x) {
            printf("%.10e ", B1[x*n + i] / A1[j*n + i]);
        }
        printf("\n");
        cnt1++;
    }
    while(cnt1 < m) {
        for(int x = 0; x < k; ++x) {
            printf("0.0000000000e+00 ");
        }
        printf("\n");
        cnt1++;
    }
    cudaDeviceSynchronize();
    cudaFree(dev_A);
    free(A);
    free(A1);
    free(B1);
    free(A_test);
    free(B_test);
    free(res);
    return 0;
}
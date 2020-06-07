#include <cmath>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MY_DEBUG 0

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct float33{
    float data[9];
};

struct float_3{
    float data[3];
};


inline float3 operator+(const float3& f3, const uchar4& u4) {
    return make_float3(f3.x + u4.x, f3.y + u4.y, f3.z + u4.z);
}

inline float3 operator-(const uchar4& u4, const float3& f3) {
    return make_float3(u4.x - f3.x, u4.y - f3.y, u4.z - f3.z);
}

inline float3 operator/(const float3& f3, float c) {
    return make_float3(f3.x / c, f3.y / c, f3.z / c);
}

float33 operator+(const float33& f33_1, const float33& f33_2){
    float33 f33_out;
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            f33_out.data[i * 3 +j] = f33_1.data[i * 3 + j] + f33_2.data[i * 3 + j];
        }
    }
    return f33_out;
}

inline float33 operator*(const float3& f3, const float_3& f_3) {      //float3 - вектор столбец, float_3 - вектор строка
    float33 f33;
    for(int i = 0; i < 3; ++i){
        float f = (i == 0) ? f3.x : ( (i == 1) ? f3.y : f3.z );
        for(int j = 0; j < 3; ++j){
            f33.data[i * 3 +j] = f * f_3.data[j];
        }
    }
    return f33;
}

inline float33 operator/(const float33& f33, float c) {      //float3 - вектор столбец, float_3 - вектор строка
    float33 f33_out;
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            f33_out.data[i * 3 +j] = f33.data[i * 3 +j] / c;
        }
    }
    return f33_out;
}



inline float norm(const float3& t) {
    return sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
}

inline float norm(const float33& f) {
    float f_out = 0;
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            f_out +=  pow(f.data[i * 3 +j], 2);
        }
    }
    return sqrt(f_out);
}

double determinant(float33& f_s){
    double det  = static_cast<double>(f_s.data[0 * 3 + 0]) * static_cast<double>(f_s.data[1 * 3 + 1]) * static_cast<double>(f_s.data[2 * 3 + 2])
                  + static_cast<double>(f_s.data[0 * 3 + 1]) * static_cast<double>(f_s.data[1 * 3 + 2]) * static_cast<double>(f_s.data[2 * 3 + 0])
                  + static_cast<double>(f_s.data[0 * 3 + 2]) * static_cast<double>(f_s.data[1 * 3 + 0]) * static_cast<double>(f_s.data[2 * 3 + 1])

                  - static_cast<double>(f_s.data[0 * 3 + 2]) * static_cast<double>(f_s.data[1 * 3 + 1]) * static_cast<double>(f_s.data[2 * 3 + 0])
                  - static_cast<double>(f_s.data[0 * 3 + 0]) * static_cast<double>(f_s.data[1 * 3 + 2]) * static_cast<double>(f_s.data[2 * 3 + 1])
                  - static_cast<double>(f_s.data[0 * 3 + 1]) * static_cast<double>(f_s.data[1 * 3 + 0]) * static_cast<double>(f_s.data[2 * 3 + 2]);
    return det;
}

bool inversion(float33& f_s, float33& f_d){

    double det = determinant(f_s);

    if (det == 0){
        return false;
    }

    float33 f_buff;

    memcpy(&f_buff, &f_s, sizeof(float33));

    float fSwap;

    fSwap = f_buff.data[0 * 3 + 1];
    f_buff.data[0 * 3 + 1] = f_buff.data[1 * 3 + 0];
    f_buff.data[1 * 3 + 0] = fSwap;

    fSwap = f_buff.data[0 * 3 + 2];
    f_buff.data[0 * 3 + 2] = f_buff.data[2 * 3 + 0];
    f_buff.data[2 * 3 + 0] = fSwap;

    fSwap = f_buff.data[1 * 3 + 2];
    f_buff.data[1 * 3 + 2] = f_buff.data[2 * 3 + 1];
    f_buff.data[2 * 3 + 1] = fSwap;


    double d_d[9];

    d_d[0 * 3 + 0] = 0 + (f_buff.data[1 * 3 + 1] * f_buff.data[2 * 3 + 2] - f_buff.data[2 * 3 + 1] * f_buff.data[1 * 3 + 2]);
    d_d[0 * 3 + 1] = 0 - (f_buff.data[1 * 3 + 0] * f_buff.data[2 * 3 + 2] - f_buff.data[2 * 3 + 0] * f_buff.data[1 * 3 + 2]);
    d_d[0 * 3 + 2] = 0 + (f_buff.data[1 * 3 + 0] * f_buff.data[2 * 3 + 1] - f_buff.data[2 * 3 + 0] * f_buff.data[1 * 3 + 1]);


    d_d[1 * 3 + 0] = 0 - (f_buff.data[0 * 3 + 1] * f_buff.data[2 * 3 + 2] - f_buff.data[2 * 3 + 1] * f_buff.data[0 * 3 + 2]);
    d_d[1 * 3 + 1] = 0 + (f_buff.data[0 * 3 + 0] * f_buff.data[2 * 3 + 2] - f_buff.data[2 * 3 + 0] * f_buff.data[0 * 3 + 2]);
    d_d[1 * 3 + 2] = 0 - (f_buff.data[0 * 3 + 0] * f_buff.data[2 * 3 + 1] - f_buff.data[2 * 3 + 0] * f_buff.data[0 * 3 + 1]);

    d_d[2 * 3 + 0] = 0 + (f_buff.data[0 * 3 + 1] * f_buff.data[1 * 3 + 2] - f_buff.data[1 * 3 + 1] * f_buff.data[0 * 3 + 2]);
    d_d[2 * 3 + 1] = 0 - (f_buff.data[0 * 3 + 0] * f_buff.data[1 * 3 + 2] - f_buff.data[1 * 3 + 0] * f_buff.data[0 * 3 + 2]);
    d_d[2 * 3 + 2] = 0 + (f_buff.data[0 * 3 + 0] * f_buff.data[1 * 3 + 1] - f_buff.data[1 * 3 + 0] * f_buff.data[0 * 3 + 1]);

    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            f_d.data[i * 3 + j] = static_cast<float>(d_d[i * 3 + j] / det);
        }
    }
    return true;
}



__constant__ float3 g_avgs[32];
__constant__ float g_avg_norms[32];

__constant__ float g_covs[32][3][3];
__constant__ double g_covs_det[32];
__constant__ float g_covs_inv[32][3][3];
__constant__ float g_covs_norms[32];


__global__ void spectrum_method(int w, int h, int nc, uchar4* v) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * blockDim.x;
    int offsety = blockDim.y * blockDim.y;

    for (int i = idy; i < h; i += offsety) {
        for (int j = idx; j < w; j += offsetx) {
            int cls = 0x100;

            float max_val = 0;

            bool isFirst = true;

            uchar4& p = v[i * w + j];

            for (int k = 0; k < nc; ++k) {

                float3 f3_1 = make_float3(
                        static_cast<float>(p.x) - g_avgs[k].x,
                        static_cast<float>(p.y) - g_avgs[k].y,
                        static_cast<float>(p.z) - g_avgs[k].z
                );


                float3 f3_2 = make_float3(
                        f3_1.x * g_covs_inv[k][0][0] + f3_1.y * g_covs_inv[k][1][0] + f3_1.z * g_covs_inv[k][2][0],
                        f3_1.x * g_covs_inv[k][0][1] + f3_1.y * g_covs_inv[k][1][1] + f3_1.z * g_covs_inv[k][2][1],
                        f3_1.x * g_covs_inv[k][0][2] + f3_1.y * g_covs_inv[k][1][2] + f3_1.z * g_covs_inv[k][2][2]
                );


                float f_3   = f3_1.x * f3_2.x
                              + f3_1.y * f3_2.y
                              + f3_1.z * f3_2.z;


                float val = - f_3 - log(abs(g_covs_det[k]));

                if (isFirst){
                    max_val = val;
                    isFirst = false;
                }

                if (abs(max_val - val) < 1e-6) {
                    cls = min(cls, k);
                }
                else {
                    if (max_val < val) {
                        max_val = val;
                        cls = k;
                    }
                }
            }
            p.w = cls;
        }
    }
}

int main() {

    std::string fileName1;
    std::string fileName2;

    int w, h;

    std::cin >> fileName1;
    std::cin >> fileName2;


    FILE *fp;
    if ((fp = fopen(fileName1.c_str(), "rb")) == NULL) {
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",
                __FILE__, __LINE__, "File is not open");
        exit(0);
    }

    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);


    uchar4 *data_in = (uchar4 *)malloc(sizeof(uchar4) * w * h);

    fread(data_in, sizeof(uchar4), w * h, fp);
    fclose(fp);


    std::vector<float3> avgs;
    std::vector<float> norm_avgs;

    std::vector<float33> covs;
    std::vector<double> covs_det;
    std::vector<float33> covs_inv;
    std::vector<float> norm_covs;

    int nc;

    std::cin >> nc;

    int* np = new int[nc];

    int** x = new int*[nc];
    int** y = new int*[nc];

    for (int i = 0; i < nc; i++) {
        std::cin >> np[i];
        x[i] = new int[np[i]];
        y[i] = new int[np[i]];
        for (int j = 0; j < np[i]; j++) {
            std::cin >> x[i][j] >> y[i][j];
        }
    }

    for (int i = 0; i < nc; i++) {
        float3 avgj = make_float3(0, 0, 0);

        for (int j = 0; j < np[i]; j++) {
            avgj = avgj + data_in[y[i][j]* w + x[i][j]];
        }

        avgj = avgj / np[i];


        avgs.push_back(avgj);
        norm_avgs.push_back(norm(avgj));
    }

    for (int i = 0; i < nc; ++i) {
        float33 covj = {0};
        float33 covj_inv = {0};
        float33* f33 = new float33[np[i]];
        for (int j = 0; j < np[i]; ++j) {

            float3 f3 = (data_in[y[i][j]* w + x[i][j]] - avgs.at(i));
            float_3 f_3;

            f_3.data[0] = f3.x;
            f_3.data[1] = f3.y;
            f_3.data[2] = f3.z;

            f33[j] = (f3 * f_3);

            covj = covj + f33[j];
        }
        delete[] f33;
        covj = covj / ( static_cast<float>(np[i] - 1) );
        if (!inversion(covj, covj_inv)) {
            printf("Error!!!\n");
        }
        covs.push_back(covj);
        covs_det.push_back(determinant(covj));
        covs_inv.push_back(covj_inv);
        norm_covs.push_back(norm(covj));
    }

    CSC(cudaMemcpyToSymbol(g_avgs, avgs.data(), avgs.size() * sizeof(float3), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(g_avg_norms, norm_avgs.data(), norm_avgs.size() * sizeof(float), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(g_covs,          covs.data(),        covs.size()         * sizeof(float33), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(g_covs_det,      covs_det.data(),    covs.size()         * sizeof(double), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(g_covs_inv,      covs_inv.data(),    covs_inv.size()     * sizeof(float33), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(g_covs_norms,    norm_covs.data(),   norm_covs.size()    * sizeof(float), 0, cudaMemcpyHostToDevice));
    uchar4* dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(uchar4) * w * h));		// Выделение памяти под массив на GPU
    CSC(cudaMemcpy(dev_arr, data_in, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));	// Копирование данных с CPU на GPU


    int blocks_min = 16;
    int blocks_max = 16;
    int threads_min = 16;
    int threads_max = 16;
    int iterates = 1;
    for (int i = blocks_min; i <= blocks_max; i *= 2) {
        for (int j = threads_min; j <= threads_max; j *= 2) {
            float ellapsed_seconds = 0;
            for (int k = 0; k < iterates; k++) {

                cudaEvent_t start, stop;
                float t;
                CSC(cudaEventCreate(&start));
                CSC(cudaEventCreate(&stop));
                CSC(cudaEventRecord(start, 0));
                spectrum_method<<< dim3(i, i), dim3(j, j) >>>(w, h, nc, dev_arr);
                CSC(cudaGetLastError());
                CSC(cudaEventRecord(stop, 0));
                CSC(cudaEventSynchronize(stop));
                CSC(cudaEventElapsedTime(&t, start, stop));
                CSC(cudaEventDestroy(start));
                CSC(cudaEventDestroy(stop));
                ellapsed_seconds += t;
            }
        }
    }

    uchar4 *data_out = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    CSC(cudaMemcpy(data_out, dev_arr, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));	// Копирование данных с GPU на CPU
    CSC(cudaFree(dev_arr));							// Освобождаем память на GPU, так как дальше она нам не нужна




    if ((fp = fopen(fileName2.c_str(), "wb")) == NULL) {
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",
                __FILE__, __LINE__, "File is not open");
        exit(0);
    }
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data_out, sizeof(uchar4), w * h, fp);
    fclose(fp);
    free(data_in);
    free(data_out);
    for (int i = 0; i < nc; i++) {
        delete[] x[i];
        delete[] y[i];
    }
    delete[] np;
    delete[] x;
    delete[] y;

    return 0;
}

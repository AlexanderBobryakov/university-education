#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                                                    \
do {                                                                \
    cudaError_t res = call;                                            \
    if (res != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                                    \
    }                                                                \
} while(0)

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *out, int w, int h, int delta_w, int delta_h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    for (y = idy; y < (h/delta_h); y += offsety) {
        for (x = idx; x < (w/delta_w); x += offsetx) {
            int xx = 0;
            int yy = 0;
            int zz = 0;
            int ww = 0;
            for (int inner_x = x*delta_w; inner_x < x*delta_w + delta_w; inner_x++) {
                for (int inner_y = y*delta_h; inner_y < y*delta_h + delta_h; inner_y++) {
                    xx += (tex2D(tex, inner_x, inner_y)).x;
                    yy += (tex2D(tex, inner_x, inner_y)).y;
                    zz += (tex2D(tex, inner_x, inner_y)).z;
                    ww += (tex2D(tex, inner_x, inner_y)).w;
                }
            }
            out[y*(w/delta_w) + x] = make_uchar4(
                    xx/(delta_h*delta_w), yy/(delta_h*delta_w), , ww/(delta_h*delta_w)
            );
        }
    }
}

int main() {
    int w, h;
//    char in[9999];  // G:\Projects\CUDA\lab2\original.bin
//    scanf("%s", in);
//    char out[9999];  // G:\Projects\CUDA\lab2\result.bin
//    scanf("%s", out);
    int w_new = 335;
    int h_new = 93;
//    scanf("%d", &w_new);
//    scanf("%d", &h_new);
//    int delta_w = 67;  // 1, 3, 5, 15, 67,  201,  335, 1 005
//    int delta_h = 31;  // 1, 2, 3, 6, 9, 18, 31, 62, 93,  186,  279,  558
    FILE *fp = fopen("G:\\Projects\\CUDA\\lab2\\original.bin", "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    int delta_w = w / w_new;
    int delta_h = h / h_new;
    uchar4 *data = (uchar4 *) malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    // Подготовка данных для текстуры
    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * (w) * (h), cudaMemcpyHostToDevice));

    // Подготовка текстурной ссылки, настройка интерфейса работы с данными
    tex.addressMode[0] = cudaAddressModeClamp;    // Политика обработки выхода за границы по каждому измерению
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;        // Без интерполяции при обращении по дробным координатам
    tex.normalized = false;                        // Режим нормализации координат: без нормализации

    // Связываем интерфейс с данными
    CSC(cudaBindTextureToArray(tex, arr, ch));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * (w / delta_w) * (h / delta_h)));





    cudaEvent_t start, stop;
    float t;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start, 0));
    kernel<<<dim3(512, 512), dim3(16, 16)>>>(dev_out, w, h, delta_w, delta_h);
    CSC(cudaGetLastError());
    CSC(cudaEventRecord(stop, 0));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaEventElapsedTime(&t, start, stop));
    printf("time = %f\n", t);
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));




    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * (w / delta_w) * (h / delta_h), cudaMemcpyDeviceToHost));

    // Отвязываем данные от текстурной ссылки
    CSC(cudaUnbindTexture(tex));

    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    int a1 = (w / delta_w);
    int a2 = (h / delta_h);
    fp = fopen("G:\\\\Projects\\\\CUDA\\\\lab2\\\\result.bin", "wb");
    fwrite(&a1, sizeof(int), 1, fp);
    fwrite(&a2, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), (w / delta_w) * (h / delta_h), fp);
    fclose(fp);

    free(data);
    return 0;
}

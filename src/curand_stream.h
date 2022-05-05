#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)

class StreamRandom {
    public:
            StreamRandom(size_t, unsigned int);
            ~StreamRandom();
            double get();
    private:
            size_t n, offset;
            curandGenerator_t gen;
            double *devData, *hostData;
};
double StreamRandom::get(){
        if(offset >= n){
                CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(double), cudaMemcpyDeviceToHost));
                CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));
                offset = 0;
        }
        return hostData[offset++];
}
StreamRandom::StreamRandom(size_t n, unsigned seed){
        this->n = n;
        this->offset = 0;
        hostData = (double *)calloc(n, sizeof(double));
        CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double)));
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
        CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));
        CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(double), cudaMemcpyDeviceToHost));
        CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));
}

StreamRandom::~StreamRandom(){
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    free(hostData);
}

/*
int main(int argc, char *argv[])
{
    StreamRandom sr = StreamRandom(10, 1234);
    for(int i = 0; i < 200; i++) {
        printf("%1.4f ", sr.get());
    }
    printf("\n");
    return EXIT_SUCCESS;
}*/

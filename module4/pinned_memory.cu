#include <stdio.h>

//From https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char** argv)
{
  int shift = 20;
  int iterations = 10;
  if (argc > 1) {
    shift = atoi(argv[1]);
  }
  if (argc > 2) {
    iterations = atoi(argv[2]);
  }

  int N = 1<<shift;
  int bytes = N*sizeof(float);
  float *x, *y, *d_x, *d_y;

  cudaMallocHost((void**)&x, bytes);
  cudaMallocHost((void**)&y, bytes);

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  for (int i = 0; i < iterations; ++i) {
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  }

  //float maxError = 0.0f;
  //for (int i = 0; i < N; i++){
    //maxError = max(maxError, abs(y[i]-4.0f));
    //printf("y[%d]=%f\n",i,y[i]);
  //}
  //printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFreeHost(x);
  cudaFreeHost(y);
}

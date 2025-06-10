#include "kernel.cuh"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void square(int *inbuf, int *outbuf) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= blockDim.x * gridDim.x) return;
  outbuf[idx] = inbuf[idx] * inbuf[idx];
}


void run_square_kernel(int* in, int *out, int n) {
  int block_size = 32;
  int num_blocks = (n + block_size - 1) / block_size;
  square<<<num_blocks, block_size>>>(in, out);
  CHK_CU(cudaGetLastError());
  CHK_CU(cudaDeviceSynchronize());
}



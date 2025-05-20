#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define CHK_CU(error) \
  if (error != cudaSuccess) { \
    std::cerr << "CUDA API error: "<<  	cudaGetErrorString(error) << " " <<  __LINE__ << " at " << __FILE__ << std::endl; \
    throw std::runtime_error(cudaGetErrorString(error)); \
  }

#include "vulkan/vulkan.hpp"
#include "vulkan/buffers.hpp"

#include <cuda_runtime.h>
#include "cuda/cuda_utils.hpp"
#include "cuda/kernel.cuh"

#include <iostream>
#include <cstdint>

void run_square_kernel(int* in, int *out, int n);

class CudaInterop {
public:
  CudaInterop() : 
    instance_(),
    physical_device_(instance_),
    device_(instance_, physical_device_),
    input_buffer_(device_) {}
  ~CudaInterop() {}

  void run();

private:
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::StagingBuffer input_buffer_;
};

void CudaInterop::run() {
  // Allocate input buffers
  const uint32_t n_elements = 32;
  const uint32_t buffer_size = n_elements * sizeof(int32_t);
  // input_buffer_ = vk::StagingBuffer(device_, buffer_size);

  // // Set input data.
  // for (uint32_t i = 0; i < n_elements; ++i) {
  //   reinterpret_cast<int32_t*>(input_buffer_.mapped_)[i] = i;
  // }


  // CUDA
  int* input;
  int* output;
  int* d_input;
  int* d_output;
  CHK_CU(cudaMallocHost((void**)&input, buffer_size));
  CHK_CU(cudaMallocHost((void**)&output, buffer_size));
  CHK_CU(cudaMalloc((void**)&d_input, buffer_size));
  CHK_CU(cudaMalloc((void**)&d_output, buffer_size));

  // Set input data.
  for (uint32_t i = 0; i < n_elements; ++i) {
    reinterpret_cast<int32_t*>(input)[i] = i;
  }

  CHK_CU(cudaMemcpy(d_input, input, buffer_size, cudaMemcpyHostToDevice));

  run_square_kernel(d_input, d_output, n_elements);

  CHK_CU(cudaMemcpy(output, d_output, buffer_size, cudaMemcpyDeviceToHost));

  // Print the input data
  for (uint32_t i = 0; i < n_elements; ++i) {
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;

  for (uint32_t i = 0; i < n_elements; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;

  cudaFreeHost(input);
  cudaFreeHost(output);
  cudaFree(d_input);
  cudaFree(d_output);
}



int main(int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  CudaInterop app;
  app.run();
  return 0;
}

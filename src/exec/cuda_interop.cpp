#include "vulkan/vulkan.hpp"
#include "vulkan/buffers.hpp"
#include "vulkan/stream.hpp"

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
    vk_h_input_buffer_(device_, 0),
    vk_h_output_buffer_(device_, 0),
    vk_d_input_buffer_(device_, 0, true),
    vk_d_output_buffer_(device_, 0, true),
    compute_queue_(device_),
    command_pool_(device_),
    stream_(device_, compute_queue_, command_pool_) {}
  ~CudaInterop() {}

  void run();

private:
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::StagingBuffer vk_h_input_buffer_;
  vk::StagingBuffer vk_h_output_buffer_;
  vk::DeviceBuffer vk_d_input_buffer_;
  vk::DeviceBuffer vk_d_output_buffer_;
  vk::ComputeQueue compute_queue_;
  vk::CommandPool command_pool_;
  vk::Stream stream_;
};

void* cast_vk_to_cu(const vk::DeviceBuffer& vk_buffer, const size_t buffer_size) {
  cudaExternalMemoryHandleDesc external_memory_handle_desc;
  external_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  external_memory_handle_desc.handle.fd = vk_buffer.fd_;
  external_memory_handle_desc.size = buffer_size;
  external_memory_handle_desc.flags = 0;

  cudaExternalMemory_t external_memory;
  std::cout << "external_memory_handle_desc.handle.fd: " << external_memory_handle_desc.handle.fd << std::endl;
  CHK_CU(cudaImportExternalMemory(&external_memory, &external_memory_handle_desc));

  cudaExternalMemoryBufferDesc external_memory_buffer_desc = {};
  external_memory_buffer_desc.offset = 0;
  external_memory_buffer_desc.size = buffer_size;
  external_memory_buffer_desc.flags = 0;

  void* cuda_buffer_ptr;
  CHK_CU(cudaExternalMemoryGetMappedBuffer(&cuda_buffer_ptr, external_memory, &external_memory_buffer_desc));
  return cuda_buffer_ptr;
}

void CudaInterop::run() {
  // Allocate input buffers
  const uint32_t n_elements = 32;
  const uint32_t buffer_size = n_elements * sizeof(int32_t);
  // const VkDeviceSize memory_size = buffer_size;

  vk_h_input_buffer_ = vk::StagingBuffer(device_, buffer_size);
  vk_h_output_buffer_ = vk::StagingBuffer(device_, buffer_size);
  vk_d_input_buffer_ = vk::DeviceBuffer(device_, buffer_size, true);
  vk_d_output_buffer_ = vk::DeviceBuffer(device_, buffer_size, true);

  // Set input data.
  for (uint32_t i = 0; i < n_elements; ++i) {
    reinterpret_cast<int32_t*>(vk_h_input_buffer_.mapped_)[i] = n_elements - i;
  }
  stream_.begin();
  stream_.copy(vk_h_input_buffer_.buffer_, vk_d_input_buffer_.buffer_, buffer_size);
  stream_.submit();
  stream_.synchronize();

  // CUDA
  const bool in_cu = false;
  const bool out_cu = false;
  int* cu_h_input;
  int* cu_h_output;
  int* cu_d_input;
  int* cu_d_output;
  if (in_cu) {
    CHK_CU(cudaMallocHost((void**)&cu_h_input, buffer_size));
    CHK_CU(cudaMalloc((void**)&cu_d_input, buffer_size));
    // Set input data.
    for (uint32_t i = 0; i < n_elements; ++i) {
      reinterpret_cast<int32_t*>(cu_h_input)[i] = i;
    }
    CHK_CU(cudaMemcpy(cu_d_input, cu_h_input, buffer_size, cudaMemcpyHostToDevice));
  }
  if (out_cu) {
    CHK_CU(cudaMallocHost((void**)&cu_h_output, buffer_size));
    CHK_CU(cudaMalloc((void**)&cu_d_output, buffer_size));
  }

  void* vk_d_input_buffer_ptr = cast_vk_to_cu(vk_d_input_buffer_, buffer_size);
  void* vk_d_output_buffer_ptr = cast_vk_to_cu(vk_d_output_buffer_, buffer_size);

  // Now try the kernel
  run_square_kernel(
    (in_cu)  ? (int*)cu_d_input  : (int*)vk_d_input_buffer_ptr,
    (out_cu) ? (int*)cu_d_output : (int*)vk_d_output_buffer_ptr,
    n_elements);

  stream_.begin();
  stream_.copy(vk_d_output_buffer_.buffer_, vk_h_output_buffer_.buffer_, buffer_size);
  stream_.submit();
  stream_.synchronize();
  if (out_cu) {
    CHK_CU(cudaMemcpy(cu_h_output, cu_d_output, buffer_size, cudaMemcpyDeviceToHost));
  }

  // Print the input data
  std::cout << "Input data " << (in_cu ? "cu" : "vk") << ": " << std::endl;
  for (uint32_t i = 0; i < n_elements; ++i) {
    if (in_cu) {
      std::cout << cu_h_input[i] << " ";
    } else {
      std::cout << reinterpret_cast<int32_t*>(vk_h_input_buffer_.mapped_)[i] << " ";
    }
  }
  std::cout << std::endl;

  std::cout << "Output data " << (out_cu ? "cu" : "vk") << ": " << std::endl;
  for (uint32_t i = 0; i < n_elements; ++i) {
    if (out_cu) {
      std::cout << cu_h_output[i] << " ";
    } else {
      std::cout << reinterpret_cast<int32_t*>(vk_h_output_buffer_.mapped_)[i] << " ";
    }
  }
  std::cout << std::endl;

  cudaFreeHost(cu_h_input);
  cudaFreeHost(cu_h_output);
  cudaFree(cu_d_input);
  cudaFree(cu_d_output);
}



int main(int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  CudaInterop app;
  app.run();
  return 0;
}

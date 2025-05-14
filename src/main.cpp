#include "vulkan.hpp"
#include "buffers.hpp"
#include "stream.hpp"
#include "kernel.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <array>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstring>

bool gUseValidation = false;

class Application {
public:
  Application() :
  instance_(),
  physical_device_(instance_),
  device_(physical_device_),
  compute_queue_(device_, device_.compute_queue_family_index_),
  command_pool_(device_, device_.compute_queue_family_index_),
  descriptor_pool_(device_, 100),
  stream_(device_, compute_queue_, command_pool_),
  compute_shaders_{{
    vk::ComputeShader<PushConstants>(device_, descriptor_pool_,
      "build/shaders/shader.comp.spv"),
    vk::ComputeShader<PushConstants>(device_, descriptor_pool_,
      "build/shaders/shader.comp.spv")
  }},
  uniform_buffer_(device_, physical_device_.phys_memory_props_),
  input_buffer_(device_, physical_device_.phys_memory_props_),
  d_input_buffer_(device_, physical_device_.phys_memory_props_),
  output_buffer_(device_, physical_device_.phys_memory_props_),
  d_output_buffer_(device_, physical_device_.phys_memory_props_) {}
  ~Application() {}

  void run();

private:
 struct PushConstants {
    uint32_t x, y, z;
  };
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::ComputeQueue compute_queue_;
  vk::CommandPool command_pool_;
  vk::DescriptorPool descriptor_pool_;
  vk::Stream stream_;
  std::array<vk::ComputeShader<PushConstants>, 2> compute_shaders_;

  vk::UniformBuffer uniform_buffer_;
  vk::StagingBuffer input_buffer_;
  vk::DeviceBuffer d_input_buffer_;
  vk::StagingBuffer output_buffer_;
  vk::DeviceBuffer d_output_buffer_;

  bool synchronization2_supported_ = false;
};


void Application::run() {
  
  std::cout << "==== Allocate buffer & memory ====" << std::endl;
  // Allocate uniform buffer
  struct params{
    uint32_t x, y, z;
  };
  params grid = { 1, 2, 4 };
  const VkPhysicalDeviceMemoryProperties& phys_memory_props = physical_device_.phys_memory_props_;
  uniform_buffer_ = std::move(vk::UniformBuffer(device_, phys_memory_props, sizeof(params)));
  memcpy(uniform_buffer_.mapped_, &grid, sizeof(params));
  
  // Allocate input and output buffers
  const uint32_t n_elements = 32;
  const uint32_t buffer_size = n_elements * sizeof(int32_t);

  const VkDeviceSize memory_size = buffer_size;

  input_buffer_ = vk::StagingBuffer(device_, phys_memory_props, memory_size);
  output_buffer_ = vk::StagingBuffer(device_, phys_memory_props, memory_size);
  d_input_buffer_ = vk::DeviceBuffer(device_, phys_memory_props, memory_size);
  d_output_buffer_ = vk::DeviceBuffer(device_, phys_memory_props, memory_size);

  // Set input data.
  for (uint32_t i = 0; i < n_elements; ++i) {
    reinterpret_cast<int32_t*>(input_buffer_.mapped_)[i] = i;
  }

  stream_.begin();
  for (size_t i = 0; i < compute_shaders_.size(); ++i) {
    // Copy input buffer to device local buffer
    if (i == 0) {
      std::cout << "==== H2D ====" << std::endl;
      stream_.copy(input_buffer_.buffer_, d_input_buffer_.buffer_, memory_size);
    }

    std::cout << "==== Dispatch compute shader[" << i << "] ====" << std::endl;
    auto& cs = compute_shaders_[i];
    std::vector<std::tuple<VkDescriptorType, VkBuffer>> descriptor_types{
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniform_buffer_.buffer_},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_input_buffer_.buffer_},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_output_buffer_.buffer_},
    };
    PushConstants pc = {static_cast<uint32_t>(i), 1, 2};
    cs.bind(pc, descriptor_types);
    stream_.dispatch(cs, n_elements / 32, 1, 1);
    stream_.barrier();

    if (i == compute_shaders_.size() - 1) {
      std::cout << "==== D2H ====" << std::endl;
      stream_.copy(d_output_buffer_.buffer_, output_buffer_.buffer_, memory_size);
    }
    else {
      // Swap input and output buffers
      std::swap(d_input_buffer_, d_output_buffer_);
    }
  }

  stream_.submit();
  stream_.synchronize();

  for (uint32_t i = 0; i < n_elements; ++i) {
    std::cout << reinterpret_cast<int32_t*>(input_buffer_.mapped_)[i] << " ";
  }
  std::cout << std::endl;

  for (uint32_t i = 0; i < n_elements; ++i) {
    std::cout << reinterpret_cast<int32_t*>(output_buffer_.mapped_)[i] << " ";
  }
  std::cout << std::endl;
}


int main(int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  Application app;
  app.run();
  return 0;
}

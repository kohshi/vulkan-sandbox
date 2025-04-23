#include "buffers.hpp"
#include "semaphore.hpp"
#include "kernel.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstring>

// #define VOLK_IMPLEMENTATION
// #include "volk/volk.h"


bool gUseValidation = false;

class Application {
public:
  Application() :
  instance_(VK_NULL_HANDLE),
  physical_device_(VK_NULL_HANDLE),
  device_(VK_NULL_HANDLE),
  compute_queue_(VK_NULL_HANDLE),
  command_pool_(VK_NULL_HANDLE),
  descriptor_pool_(VK_NULL_HANDLE) {}
  ~Application() {
    for (auto& shader : compute_shaders_) {
      if (shader.get() != nullptr) { shader.reset(); }
    }
    if (stream_.get() != nullptr) { stream_.reset(); }
    if (uniform_buffer_.get() != nullptr) { uniform_buffer_.reset(); }
    if (input_buffer_.get() != nullptr)   { input_buffer_.reset(); }
    if (output_buffer_.get() != nullptr)  { output_buffer_.reset(); }
    if (d_input_buffer_.get() != nullptr) { d_input_buffer_.reset(); }
    if (d_output_buffer_.get() != nullptr) { d_output_buffer_.reset(); }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    }
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    if (device_ != VK_NULL_HANDLE) {
      CHK(vkDeviceWaitIdle(device_));
      std::cout << "==== Destroy device ====" << std::endl;
      vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
      std::cout << "==== Destroy vulkan instance ====" << std::endl;
      vkDestroyInstance(instance_, nullptr);
    }
  }

  void initialize();
  void run();

private:
  VkInstance instance_;
  VkPhysicalDevice physical_device_;
  VkPhysicalDeviceMemoryProperties phys_memory_props_;
  VkDevice device_;
  VkQueue compute_queue_;
  VkCommandPool command_pool_;
  VkDescriptorPool descriptor_pool_;
  std::unique_ptr<vk::Stream> stream_;

  std::unique_ptr<UniformBuffer> uniform_buffer_;
  std::unique_ptr<StagingBuffer> input_buffer_;
  std::unique_ptr<DeviceBuffer> d_input_buffer_;
  std::unique_ptr<StagingBuffer> output_buffer_;
  std::unique_ptr<DeviceBuffer> d_output_buffer_;
  std::vector<std::unique_ptr<vk::ComputeShader>> compute_shaders_;
  std::vector<VkDescriptorSet> descriptorSets_;

  bool synchronization2_supported_ = false;
  PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR_;
};


void Application::initialize() {

  const VkApplicationInfo appInfo = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType
    .pApplicationName = "VulkanCompute",      // Application Name
    .pEngineName = "VulkanCompute",            // Application Version
    .engineVersion = VK_MAKE_VERSION(1, 0, 0),    // Engine Version
    .apiVersion= VK_API_VERSION_1_3    // Vulkan API version
  };
  
  {// Create instance
    std::cout << "==== Create vulkan instance ====" << std::endl;
    VkInstanceCreateInfo instance_ci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo
    };

    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    
    layers.push_back("VK_LAYER_KHRONOS_synchronization2");
    if (gUseValidation)
    {
      layers.push_back("VK_LAYER_KHRONOS_validation");
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);  
    }
    instance_ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instance_ci.ppEnabledLayerNames = layers.data();
    instance_ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instance_ci.ppEnabledExtensionNames = extensions.data();

    CHK(vkCreateInstance(&instance_ci, nullptr, &instance_));
  }
  uint32_t version = 0;
  vkEnumerateInstanceVersion(&version);
  std::cout << "Vulkan Instance version: " <<
        VK_VERSION_MAJOR(version) << "." <<
        VK_VERSION_MINOR(version) << "." <<
        VK_VERSION_PATCH(version) << std::endl;

  {// Create physical device
    std::cout << "==== Create physical device ====" << std::endl;
    uint32_t n_phys_dev = 0;
    CHK(vkEnumeratePhysicalDevices(instance_, &n_phys_dev, nullptr));
    std::vector<VkPhysicalDevice> phys_devs(n_phys_dev);
    std::vector<VkPhysicalDeviceProperties> props(n_phys_dev);
    CHK(vkEnumeratePhysicalDevices(instance_, &n_phys_dev, phys_devs.data()));
    for (uint32_t i = 0; i < n_phys_dev; ++i) {
      vkGetPhysicalDeviceProperties(phys_devs[i], &props[i]);
      std::cout << "GPU " << i << ": " << props[i].deviceName << " "
                << VK_VERSION_MAJOR(props[i].apiVersion) << "."
                << VK_VERSION_MINOR(props[i].apiVersion) << "." 
                << VK_VERSION_PATCH(props[i].apiVersion) << std::endl;
    }

    // use gpu[0]
    const int used_index = 0;
    physical_device_ = phys_devs[used_index];
    const VkPhysicalDeviceLimits& limits = props[used_index].limits;
    std::cout << "Using GPU " << used_index << ": " << props[used_index].deviceName << std::endl;
    std::cout << "==== Physical device limits ====" << std::endl;
    std::cout << "maxUniformBufferRange: " << limits.maxUniformBufferRange << std::endl;
    std::cout << "maxStorageBufferRange: " << limits.maxStorageBufferRange << std::endl;
    std::cout << "maxPushConstantsSize: " << limits.maxPushConstantsSize << std::endl;
    std::cout << "maxMemoryAllocationCount: " << limits.maxMemoryAllocationCount << std::endl;
    std::cout << "maxSamplerAllocationCount: " << limits.maxSamplerAllocationCount << std::endl;

    // Get memory properties
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &phys_memory_props_);
  }

  // Get queue family properties
  uint32_t n_queue_family_props = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &n_queue_family_props, nullptr);
  std::vector<VkQueueFamilyProperties> queue_family_props(n_queue_family_props);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &n_queue_family_props, queue_family_props.data());
  uint32_t compute_queue_family_index = 0;
  uint32_t n_compute_queue = 0;
  for (uint32_t i = 0; i < n_queue_family_props; ++i) {
    if (queue_family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      compute_queue_family_index = i;
      n_compute_queue = queue_family_props[i].queueCount;
    }
  }
  std::cout << "computeQueueFamilyIndex: " << compute_queue_family_index << std::endl;
  std::cout << "computeQueueCount: " << n_compute_queue << std::endl;

  // Get supported extnsions
  uint32_t n_extensions = 0;
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &n_extensions, nullptr);
  std::vector<VkExtensionProperties> supported_exts(n_extensions);
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &n_extensions, supported_exts.data());

  synchronization2_supported_ = false;
  for (const auto& ext : supported_exts) {
      if (std::strcmp(ext.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) {
        synchronization2_supported_ = true;
          break;
      }
  }
  std::cout << "==== Supported extensions ====" << std::endl;
  std::cout << "VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME: " << synchronization2_supported_ << std::endl;

  {// Create device
    std::cout << "==== Create device ====" << std::endl;
    std::vector<const char*> extensions;
    if (synchronization2_supported_) {
      extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
    if (gUseValidation) {
      extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
    const float queue_priorities = 1.0f;
    VkPhysicalDeviceSynchronization2FeaturesKHR sync2_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .synchronization2 = VK_TRUE,
    };

    VkDeviceQueueCreateInfo device_queue_ci{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = compute_queue_family_index,
      .queueCount = 1,
      .pQueuePriorities = &queue_priorities,
    };
    VkDeviceCreateInfo device_ci{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = synchronization2_supported_ ? &sync2_features : nullptr,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &device_queue_ci,
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data(),
    };

    CHK(vkCreateDevice(physical_device_, &device_ci, nullptr, &device_));
    // Get process addresses
    vkCmdPipelineBarrier2KHR_ = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(
      vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2KHR"));
    
    vkGetDeviceQueue(device_, compute_queue_family_index, 0/*queueIndex*/, &compute_queue_);
  }

  {// Create command pool
    std::cout << "==== Create command pool ====" << std::endl;
    VkCommandPoolCreateInfo command_pool_ci{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = compute_queue_family_index,
    };
    CHK(vkCreateCommandPool(device_, &command_pool_ci, nullptr, &command_pool_));
  }

  {// Create descriptor pool
    std::cout << "==== Create descriptor pool ====" << std::endl;
    // A descriptor pool holds a sufficient 
    // number of descriptors to be used by the application, 
    // and when a descriptor set is allocated, it is cut out of the pool.
    const uint32_t n_descriptor = 100;
    std::vector<VkDescriptorPoolSize> pool_sizes {
      {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = n_descriptor,
      },
      {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = n_descriptor,
      },
    };
    VkDescriptorPoolCreateInfo descriptor_pool_ci {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = n_descriptor,
      .poolSizeCount = uint32_t(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
    };
    CHK(vkCreateDescriptorPool(device_, &descriptor_pool_ci, nullptr, &descriptor_pool_));
  }

  std::cout << "==== Create compute shader ====" << std::endl;
  for (size_t i = 0; i < 2; ++i) {
    compute_shaders_.push_back(std::make_unique<vk::ComputeShader>(
      device_,
      descriptor_pool_,
      "build/shaders/shader.comp.spv"));
  }

  stream_.reset(new vk::Stream(device_, compute_queue_, command_pool_));
}

void Application::run() {
  
  std::cout << "==== Allocate buffer & memory ====" << std::endl;
  // Allocate uniform buffer
  struct params{
    uint32_t x, y, z;
  };
  params grid = { 1, 2, 4 };
  uniform_buffer_.reset(new UniformBuffer(device_, phys_memory_props_));
  uniform_buffer_->allocate(sizeof(params));
  memcpy(uniform_buffer_->mapped_, &grid, sizeof(params));
  
  // Allocate input and output buffers
  const uint32_t n_elements = 32;
  const uint32_t buffer_size = n_elements * sizeof(int32_t);

  const VkDeviceSize memory_size = buffer_size;
  
  input_buffer_.reset(new StagingBuffer(device_, phys_memory_props_));
  output_buffer_.reset(new StagingBuffer(device_, phys_memory_props_));
  d_input_buffer_.reset(new DeviceBuffer(device_, phys_memory_props_));
  d_output_buffer_.reset(new DeviceBuffer(device_, phys_memory_props_));

  input_buffer_->allocate(memory_size);
  output_buffer_->allocate(memory_size);

  d_input_buffer_->allocate(memory_size);
  d_output_buffer_->allocate(memory_size);

  // Set input data.
  for (uint32_t i = 0; i < n_elements; ++i) {
    reinterpret_cast<int32_t*>(input_buffer_->mapped_)[i] = i;
  }

  stream_->begin();
  for (size_t i = 0; i < compute_shaders_.size(); ++i) {
    // Copy input buffer to device local buffer
    if (i == 0) {
      std::cout << "==== H2D ====" << std::endl;
      stream_->copy(input_buffer_->buffer_, d_input_buffer_->buffer_, memory_size);
    }

    std::cout << "==== Dispatch compute shader[" << i << "] ====" << std::endl;
    auto& cs = compute_shaders_[i];
    std::vector<std::tuple<VkDescriptorType, VkBuffer>> descriptor_types{
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniform_buffer_->buffer_},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_input_buffer_->buffer_},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_output_buffer_->buffer_},
    };
    cs->bind(descriptor_types);
    stream_->dispatch(*cs, n_elements / 32, 1, 1);
    stream_->barrier();

    if (i == compute_shaders_.size() - 1) {
      std::cout << "==== D2H ====" << std::endl;
      stream_->copy(d_output_buffer_->buffer_, output_buffer_->buffer_, memory_size);
    }
    else {
      // Swap input and output buffers
      std::swap(d_input_buffer_, d_output_buffer_);
    }
  }

  stream_->submit();
  stream_->synchronize();

  for (uint32_t i = 0; i < n_elements; ++i) {
    std::cout << reinterpret_cast<int32_t*>(input_buffer_->mapped_)[i] << " ";
  }
  std::cout << std::endl;

  for (uint32_t i = 0; i < n_elements; ++i) {
    std::cout << reinterpret_cast<int32_t*>(output_buffer_->mapped_)[i] << " ";
  }
  std::cout << std::endl;
}


int main(int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  Application app;
  app.initialize();
  app.run();
  return 0;
}

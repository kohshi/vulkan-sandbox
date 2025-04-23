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
  physiscalDevice_(VK_NULL_HANDLE),
  device_(VK_NULL_HANDLE),
  computeQueue_(VK_NULL_HANDLE),
  commandPool_(VK_NULL_HANDLE),
  descriptorPool_(VK_NULL_HANDLE) {}
  ~Application() {
    for (auto& shader : computeShaders_) {
      if (shader.get() != nullptr) { shader.reset(); }
    }
    if (stream_.get() != nullptr) { stream_.reset(); }
    if (uniformBuffer_.get() != nullptr) { uniformBuffer_.reset(); }
    if (inputBuffer_.get() != nullptr)   { inputBuffer_.reset(); }
    if (outputBuffer_.get() != nullptr)  { outputBuffer_.reset(); }
    if (d_inputBuffer_.get() != nullptr) { d_inputBuffer_.reset(); }
    if (d_outputBuffer_.get() != nullptr) { d_outputBuffer_.reset(); }
    for (auto& shader : shaderModules_) {
      vkDestroyShaderModule(device_, shader, nullptr);
    }
    if (descriptorPool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    }
    if (commandPool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, commandPool_, nullptr);
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
  VkPhysicalDevice physiscalDevice_;
  VkPhysicalDeviceMemoryProperties physMemoryProperties_;
  VkDevice device_;
  VkQueue computeQueue_;
  VkCommandPool commandPool_;
  VkDescriptorPool descriptorPool_;
  std::vector<VkShaderModule> shaderModules_;
  std::unique_ptr<vk::Stream> stream_;

  std::unique_ptr<UniformBuffer> uniformBuffer_;
  std::unique_ptr<StagingBuffer> inputBuffer_;
  std::unique_ptr<DeviceBuffer> d_inputBuffer_;
  std::unique_ptr<StagingBuffer> outputBuffer_;
  std::unique_ptr<DeviceBuffer> d_outputBuffer_;
  std::vector<std::unique_ptr<vk::ComputeShader>> computeShaders_;
  std::vector<VkDescriptorSet> descriptorSets_;

  bool sync2Supported_ = false;
  PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR_;
};


void Application::initialize() {
  std::cout << "==== Create vulkan instance ====" << std::endl;
  const VkApplicationInfo appInfo = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType
    .pApplicationName = "VulkanCompute",      // Application Name
    .pEngineName = "VulkanCompute",            // Application Version
    .engineVersion = VK_MAKE_VERSION(1, 0, 0),    // Engine Version
    .apiVersion= VK_API_VERSION_1_3    // Vulkan API version
  };
  
  VkInstanceCreateInfo ci = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &appInfo
  };

  std::vector<const char*> layers;
  std::vector<const char*> extensions;
  VkValidationFeaturesEXT validationFeatures = {
    .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
    .enabledValidationFeatureCount = 0,
    .pEnabledValidationFeatures = nullptr,
  };
  std::vector<VkValidationFeatureEnableEXT>  validationFeatEnables = {
    VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
  };
  layers.push_back("VK_LAYER_KHRONOS_synchronization2");
  if (gUseValidation)
  {
    layers.push_back("VK_LAYER_KHRONOS_validation");
    validationFeatures.enabledValidationFeatureCount = validationFeatEnables.size();
    validationFeatures.pEnabledValidationFeatures = validationFeatEnables.data();
    ci.pNext = &validationFeatures;
  }
  ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
  ci.ppEnabledLayerNames = layers.data();
  ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  ci.ppEnabledExtensionNames = extensions.data();

  CHK(vkCreateInstance(&ci, nullptr, &instance_));

  uint32_t version = 0;
  vkEnumerateInstanceVersion(&version);
  std::cout << "Vulkan Instance version: " <<
        VK_VERSION_MAJOR(version) << "." <<
        VK_VERSION_MINOR(version) << "." <<
        VK_VERSION_PATCH(version) << std::endl;

  std::cout << "==== Create physical device ====" << std::endl;
  uint32_t count = 0;
  CHK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
  std::vector<VkPhysicalDevice> physDevs(count);
  std::vector<VkPhysicalDeviceProperties> props(count);
  CHK(vkEnumeratePhysicalDevices(instance_, &count, physDevs.data()));
  for (uint32_t i = 0; i < count; ++i) {
    vkGetPhysicalDeviceProperties(physDevs[i], &props[i]);
    std::cout << "GPU " << i << ": " << props[i].deviceName << " "
              << VK_VERSION_MAJOR(props[i].apiVersion) << "."
              << VK_VERSION_MINOR(props[i].apiVersion) << "." 
              << VK_VERSION_PATCH(props[i].apiVersion) << std::endl;
  }

  // use gpu[0]
  const int usedIndex = 0;
  physiscalDevice_ = physDevs[usedIndex];
  const VkPhysicalDeviceLimits& limits = props[usedIndex].limits;
  std::cout << "Using GPU " << usedIndex << ": " << props[usedIndex].deviceName << std::endl;
  std::cout << "==== Physical device limits ====" << std::endl;
  std::cout << "maxUniformBufferRange: " << limits.maxUniformBufferRange << std::endl;
  std::cout << "maxStorageBufferRange: " << limits.maxStorageBufferRange << std::endl;
  std::cout << "maxPushConstantsSize: " << limits.maxPushConstantsSize << std::endl;
  std::cout << "maxMemoryAllocationCount: " << limits.maxMemoryAllocationCount << std::endl;
  std::cout << "maxSamplerAllocationCount: " << limits.maxSamplerAllocationCount << std::endl;

  // Get memory properties
  vkGetPhysicalDeviceMemoryProperties(physiscalDevice_, &physMemoryProperties_);

  // Get queue family properties
  vkGetPhysicalDeviceQueueFamilyProperties(physiscalDevice_, &count, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilyProps(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physiscalDevice_, &count, queueFamilyProps.data());
  uint32_t computeQueueFamilyIndex = 0;
  uint32_t computeQueueCount = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (queueFamilyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      computeQueueFamilyIndex = i;
      computeQueueCount = queueFamilyProps[i].queueCount;
    }
  }
  std::cout << "computeQueueFamilyIndex: " << computeQueueFamilyIndex << std::endl;
  std::cout << "computeQueueCount: " << computeQueueCount << std::endl;

  // Get supported extnsions
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(physiscalDevice_, nullptr, &extensionCount, nullptr);
  std::vector<VkExtensionProperties> supportedExts(extensionCount);
  vkEnumerateDeviceExtensionProperties(physiscalDevice_, nullptr, &extensionCount, supportedExts.data());

  sync2Supported_ = false;
  for (const auto& ext : supportedExts) {
      if (std::strcmp(ext.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) {
        sync2Supported_ = true;
          break;
      }
  }
  std::cout << "==== Supported extensions ====" << std::endl;
  std::cout << "VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME: " << sync2Supported_ << std::endl;
  if (sync2Supported_) {
    extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
  }

  std::cout << "==== Create device ====" << std::endl;
  const float queuePrioritory = 1.0f;
  VkDeviceQueueCreateInfo deviceQueueCI{
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = computeQueueFamilyIndex,
    .queueCount = 1,
    .pQueuePriorities = &queuePrioritory,
  };
  VkPhysicalDeviceSynchronization2FeaturesKHR sync2Features{
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
    .synchronization2 = VK_TRUE,
  };
  VkDeviceCreateInfo deviceCI{
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = sync2Supported_ ? &sync2Features : nullptr,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &deviceQueueCI,
    .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
    .ppEnabledExtensionNames = extensions.data(),
  };

  CHK(vkCreateDevice(physiscalDevice_, &deviceCI, nullptr, &device_));
  // Get process addresses
  vkCmdPipelineBarrier2KHR_ = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(
    vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2KHR"));

  vkGetDeviceQueue(device_, computeQueueFamilyIndex, 0/*queueIndex*/, &computeQueue_);

  std::cout << "==== Create command pool ====" << std::endl;
  VkCommandPoolCreateInfo commandPoolCI{
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = computeQueueFamilyIndex,
  };
  CHK(vkCreateCommandPool(device_, &commandPoolCI, nullptr, &commandPool_));

  std::cout << "==== Create descriptor pool ====" << std::endl;
  // A descriptor pool holds a sufficient 
  // number of descriptors to be used by the application, 
  // and when a descriptor set is allocated, it is cut out of the pool.
  const uint32_t descCount = 100;
  std::vector<VkDescriptorPoolSize> poolSizes {
    {
      .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = descCount,
    },
    {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = descCount,
    },
  };
  VkDescriptorPoolCreateInfo descriptorPoolCI {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = descCount,
    .poolSizeCount = uint32_t(poolSizes.size()),
    .pPoolSizes = poolSizes.data(),
  };
  CHK(vkCreateDescriptorPool(device_, &descriptorPoolCI, nullptr, &descriptorPool_));

  std::cout << "==== Create compute shader ====" << std::endl;
  for (size_t i = 0; i < 2; ++i) {
    computeShaders_.push_back(std::make_unique<vk::ComputeShader>(
      device_,
      descriptorPool_,
      "build/shaders/shader.comp.spv"));
  }

  stream_.reset(new vk::Stream(device_, computeQueue_, commandPool_));
}

void Application::run() {
  
  std::cout << "==== Allocate buffer & memory ====" << std::endl;
  // Allocate uniform buffer
  struct params{
    uint32_t x, y, z;
  };
  params grid = { 1, 2, 4 };
  uniformBuffer_.reset(new UniformBuffer(device_, physMemoryProperties_));
  uniformBuffer_->allocate(sizeof(params));
  memcpy(uniformBuffer_->mapped_, &grid, sizeof(params));
  
  // Allocate input and output buffers
  const uint32_t numElements = 32;
  const uint32_t bufferSize = numElements * sizeof(int32_t);

  const VkDeviceSize memorySize = bufferSize;
  
  inputBuffer_.reset(new StagingBuffer(device_, physMemoryProperties_));
  outputBuffer_.reset(new StagingBuffer(device_, physMemoryProperties_));
  d_inputBuffer_.reset(new DeviceBuffer(device_, physMemoryProperties_));
  d_outputBuffer_.reset(new DeviceBuffer(device_, physMemoryProperties_));

  inputBuffer_->allocate(memorySize);
  outputBuffer_->allocate(memorySize);

  d_inputBuffer_->allocate(memorySize);
  d_outputBuffer_->allocate(memorySize);

  // Set input data.
  for (uint32_t i = 0; i < numElements; ++i) {
    reinterpret_cast<int32_t*>(inputBuffer_->mapped_)[i] = i;
  }

  stream_->begin();
  for (size_t i = 0; i < computeShaders_.size(); ++i) {
    // Copy input buffer to device local buffer
    if (i == 0) {
      std::cout << "==== H2D ====" << std::endl;
      stream_->copy(inputBuffer_->buffer_, d_inputBuffer_->buffer_, memorySize);
    }

    std::cout << "==== Dispatch compute shader[" << i << "] ====" << std::endl;
    auto& cs = computeShaders_[i];
    std::vector<std::tuple<VkDescriptorType, VkBuffer>> descriptorTypes{
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBuffer_->buffer_},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_inputBuffer_->buffer_},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_outputBuffer_->buffer_},
    };
    cs->bind(descriptorTypes);
    stream_->dispatch(*cs, numElements / 32, 1, 1);
    stream_->barrier();

    if (i == computeShaders_.size() - 1) {
      std::cout << "==== D2H ====" << std::endl;
      stream_->copy(d_outputBuffer_->buffer_, outputBuffer_->buffer_, memorySize);
    }
    else {
      // Swap input and output buffers
      std::swap(d_inputBuffer_, d_outputBuffer_);
    }
  }

  stream_->submit();
  stream_->synchronize();

  for (uint32_t i = 0; i < numElements; ++i) {
    std::cout << reinterpret_cast<int32_t*>(inputBuffer_->mapped_)[i] << " ";
  }
  std::cout << std::endl;

  for (uint32_t i = 0; i < numElements; ++i) {
    std::cout << reinterpret_cast<int32_t*>(outputBuffer_->mapped_)[i] << " ";
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

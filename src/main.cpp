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
  descriptorPool_(VK_NULL_HANDLE),
  commandBuffer_(VK_NULL_HANDLE) {}
  ~Application() {
    if (computeShader_.get() != nullptr) { computeShader_.reset(); }
    if (stream_.get() != nullptr) { stream_.reset(); }
    if (uniformBuffer_.get() != nullptr) { uniformBuffer_.reset(); }
    if (inputBuffer_.get() != nullptr)   { inputBuffer_.reset(); }
    if (outputBuffer_.get() != nullptr)  { outputBuffer_.reset(); }
    if (d_inputBuffer_.get() != nullptr) { d_inputBuffer_.reset(); }
    if (d_outputBuffer_.get() != nullptr) { d_outputBuffer_.reset(); }
    if (commandBuffer_ != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer_);
    }
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
  VkCommandBuffer commandBuffer_;
  std::vector<VkShaderModule> shaderModules_;
  std::unique_ptr<vk::Stream> stream_;

  std::unique_ptr<UniformBuffer> uniformBuffer_;
  std::unique_ptr<StagingBuffer> inputBuffer_;
  std::unique_ptr<DeviceBuffer> d_inputBuffer_;
  std::unique_ptr<StagingBuffer> outputBuffer_;
  std::unique_ptr<DeviceBuffer> d_outputBuffer_;
  std::unique_ptr<vk::ComputeShader> computeShader_;
  std::vector<VkDescriptorSet> descriptorSets_;
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
  if (gUseValidation)
  {
    layers.push_back("VK_LAYER_KHRONOS_validation");
    // extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    validationFeatures.enabledValidationFeatureCount = validationFeatEnables.size();
    validationFeatures.pEnabledValidationFeatures = validationFeatEnables.data();

    ci.enabledExtensionCount = uint32_t(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = uint32_t(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    ci.pNext = &validationFeatures;
  }

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

  std::cout << "==== Create device ====" << std::endl;
  const float queuePrioritory = 1.0f;
  VkDeviceQueueCreateInfo deviceQueueCI{
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = computeQueueFamilyIndex,
    .queueCount = 1,
    .pQueuePriorities = &queuePrioritory,
  };
  VkDeviceCreateInfo deviceCI{
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &deviceQueueCI,
    .enabledExtensionCount = 0,
    .ppEnabledExtensionNames = 0,
  };

  CHK(vkCreateDevice(physiscalDevice_, &deviceCI, nullptr, &device_));

  vkGetDeviceQueue(device_, computeQueueFamilyIndex, 0/*queueIndex*/, &computeQueue_);

  stream_.reset(new vk::Stream(device_, computeQueue_));

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
  computeShader_.reset(new vk::ComputeShader(
    device_,
    descriptorPool_,
    "build/shaders/shader.comp.spv"));
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

  std::vector<std::tuple<VkDescriptorType, VkBuffer>> descriptorTypes{
    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBuffer_->buffer_},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_inputBuffer_->buffer_},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d_outputBuffer_->buffer_},
  };

  computeShader_->bind(descriptorTypes);

  std::cout << "==== Create command buffer ====" << std::endl;
  VkCommandBufferAllocateInfo commandAI{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = commandPool_,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };
  CHK(vkAllocateCommandBuffers(device_, &commandAI, &commandBuffer_));

  std::cout << "==== Begin command buffer ====" << std::endl;

  VkCommandBufferBeginInfo commandBufferBeginInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  CHK(vkBeginCommandBuffer(commandBuffer_, &commandBufferBeginInfo));

  // Copy input buffer to device local buffer
  VkBufferCopy copyRegion{
    .srcOffset = 0,
    .dstOffset = 0,
    .size = memorySize,
  };
  vkCmdCopyBuffer(commandBuffer_, inputBuffer_->buffer_, d_inputBuffer_->buffer_, 1, &copyRegion);

  vk::ComputeShader& cs = *computeShader_;
  vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, cs.pipeline_);
  vkCmdBindDescriptorSets(commandBuffer_,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    cs.pipelineLayout_,
    0 /*firstSet*/, 1/*descriptorSetCount*/,
    &(cs.descriptorSets_[0]),
    0/*DynamicOffsetCount*/,
    nullptr);
  // group count x,y,z
  vkCmdDispatch(commandBuffer_, numElements / 32, 1, 1);

  vkCmdCopyBuffer(commandBuffer_, d_outputBuffer_->buffer_, outputBuffer_->buffer_, 1, &copyRegion);
  CHK(vkEndCommandBuffer(commandBuffer_));

  stream_->submit(commandBuffer_);
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

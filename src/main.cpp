#include "buffers.hpp"
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

namespace {
VkShaderModule createShaderModule(VkDevice device, const void* code, size_t length)
{
  VkShaderModuleCreateInfo ci{
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = length,
    .pCode = reinterpret_cast<const uint32_t*>(code),
  };
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  CHK(vkCreateShaderModule(device, &ci, nullptr, &shaderModule));
  return std::move(shaderModule);
}

bool Load(std::filesystem::path filePath, std::vector<char>& data) {
  if (std::filesystem::exists(filePath))
  {
    std::ifstream infile(filePath, std::ios::binary);
    if (infile)
    {
      auto size = infile.seekg(0, std::ios::end).tellg();
      data.resize(size);
      infile.seekg(0, std::ios::beg).read(data.data(), size);
      return true;
    }
  }
  filePath = std::filesystem::path("../") / filePath;
  if (std::filesystem::exists(filePath))
  {
    std::ifstream infile(filePath, std::ios::binary);
    if (infile)
    {
      auto size = infile.seekg(0, std::ios::end).tellg();
      data.resize(size);
      infile.seekg(0, std::ios::beg).read(data.data(), size);
      return true;
    }
  }
  return false;
}
}// namespace {

class Application {
public:
  Application() :
  instance_(VK_NULL_HANDLE),
  physiscalDevice_(VK_NULL_HANDLE),
  device_(VK_NULL_HANDLE),
  computeQueue_(VK_NULL_HANDLE),
  commandPool_(VK_NULL_HANDLE),
  descriptorPool_(VK_NULL_HANDLE),
  commandBuffer_(VK_NULL_HANDLE),
  descriptorSetLayout_(VK_NULL_HANDLE),
  pipelineLayout_(VK_NULL_HANDLE),
  pipeline_(VK_NULL_HANDLE) {}
  ~Application() {
    if (uniformBuffer_.get() != nullptr) {
      uniformBuffer_.reset();
    }
    if (inputBuffer_.get() != nullptr) {
      inputBuffer_.reset();
    }
    if (outputBuffer_.get() != nullptr) {
      outputBuffer_.reset();
    }
    if (d_inputBuffer_.get() != nullptr) {
      d_inputBuffer_.reset();
    }
    if (d_outputBuffer_.get() != nullptr) {
      d_outputBuffer_.reset();
    }
    if (commandBuffer_ != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer_);
    }
    for (auto& shader : shaderModules_) {
      vkDestroyShaderModule(device_, shader, nullptr);
    }
    if (pipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
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

  void initialize() {
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
  }

  void run() {

    initialize();
    
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

    std::cout << "==== Create descriptor set layout & pipeline layout ====" << std::endl;
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings{
      {// uniform buffer
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {// input buffer
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {// output buffer
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
    };
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = uint32_t(layoutBindings.size()),
      .pBindings = layoutBindings.data(),
    };
    CHK(vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout_));

    VkPipelineLayoutCreateInfo pipelineLayoutCI{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptorSetLayout_,
    };
    CHK(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr, &pipelineLayout_));

    std::cout << "==== Create shader module & pipeline ====" << std::endl;
    std::vector<char> computeSpv;
    Load("build/shaders/shader.comp.spv", computeSpv);
    VkShaderModule shader = createShaderModule(device_, computeSpv.data(), computeSpv.size());
    VkPipelineShaderStageCreateInfo computeStageCI {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader,
      .pName = "main",
    };
    shaderModules_.push_back(shader);
    VkComputePipelineCreateInfo computePipelineCI{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = computeStageCI,
      .layout = pipelineLayout_,
    };
    CHK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &pipeline_));

    std::cout << "==== Allocate descriptor set ====" << std::endl;
    std::vector<VkDescriptorSetLayout> dsLayouts{ descriptorSetLayout_ };
    VkDescriptorSetAllocateInfo dsAllocInfoComp = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool_,
      .descriptorSetCount = uint32_t(dsLayouts.size()),
      .pSetLayouts = dsLayouts.data(),
    };

    descriptorSets_.resize(dsLayouts.size());
    CHK(vkAllocateDescriptorSets(device_, &dsAllocInfoComp, descriptorSets_.data()));

    VkDescriptorBufferInfo uniformBufferInfo{
      .buffer = uniformBuffer_->buffer_,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo inputBufferInfo{
      .buffer = d_inputBuffer_->buffer_,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo outputBufferInfo{
      .buffer = d_outputBuffer_->buffer_,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    std::vector<VkWriteDescriptorSet> writeDescriptorSets{
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptorSets_[0],
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &uniformBufferInfo
      },
      {// input buffer
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptorSets_[0],
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &inputBufferInfo,
      },
      {// output buffer
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptorSets_[0],
        .dstBinding = 2,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &outputBufferInfo,
      },
    };
    vkUpdateDescriptorSets(device_, uint32_t(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

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

    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSets_[0], 0, nullptr);
    // group count x,y,z
    vkCmdDispatch(commandBuffer_, numElements / 32, 1, 1);

    vkCmdCopyBuffer(commandBuffer_, d_outputBuffer_->buffer_, outputBuffer_->buffer_, 1, &copyRegion);
    CHK(vkEndCommandBuffer(commandBuffer_));

    VkSubmitInfo VkSubmitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer_,
    };
    // Submit the command buffer to the specified queue.
    CHK(vkQueueSubmit(computeQueue_, 1/*submitCount*/, &VkSubmitInfo, VK_NULL_HANDLE/*fence*/));
    CHK(vkQueueWaitIdle(computeQueue_));

    for (uint32_t i = 0; i < numElements; ++i) {
      std::cout << reinterpret_cast<int32_t*>(inputBuffer_->mapped_)[i] << " ";
    }
    std::cout << std::endl;

    for (uint32_t i = 0; i < numElements; ++i) {
      std::cout << reinterpret_cast<int32_t*>(outputBuffer_->mapped_)[i] << " ";
    }
    std::cout << std::endl;
  }

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

  std::unique_ptr<UniformBuffer> uniformBuffer_;
  std::unique_ptr<StagingBuffer> inputBuffer_;
  std::unique_ptr<DeviceBuffer> d_inputBuffer_;
  std::unique_ptr<StagingBuffer> outputBuffer_;
  std::unique_ptr<DeviceBuffer> d_outputBuffer_;
  VkDescriptorSetLayout descriptorSetLayout_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline pipeline_;
  std::vector<VkDescriptorSet> descriptorSets_;
};

int main(int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  Application app;
  app.run();
  return 0;
}

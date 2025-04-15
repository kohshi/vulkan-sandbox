#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>

// #define VOLK_IMPLEMENTATION
// #include "volk/volk.h"

#define CHK(result) \
  if (result != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error: %d at %u %s\n", result, __LINE__, __FILE__); \
    exit(-1); \
  }

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

uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& props,
                        const uint32_t memoryTypeBits,
                        const VkMemoryPropertyFlags flags)
{
  uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if (memoryTypeBits & (1 << i)) {
      // Check if the memory type is suitable
      if ((props.memoryTypes[i].propertyFlags & flags) == flags) {
        memoryTypeIndex = i;
        break;
      }
    }
  }
  // Check if memory type is found
  CHK(((memoryTypeIndex == VK_MAX_MEMORY_TYPES) ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS));
  return memoryTypeIndex;
}

VkResult createBuffer(VkDevice &device, 
                      const VkPhysicalDeviceMemoryProperties& physMemProps,
                      const VkDeviceSize bufferSize,
                      const VkBufferUsageFlags usage,
                      const VkMemoryPropertyFlags propFlags,
                      VkBuffer& buffer,
                      VkDeviceMemory& memory)
{
  VkBufferCreateInfo ci{
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = bufferSize,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  CHK(vkCreateBuffer(device, &ci, nullptr, &buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  uint32_t memoryTypeIndex =
  findMemoryType(physMemProps,
                 memRequirements.memoryTypeBits,
                 propFlags);
  const VkMemoryAllocateInfo memoryAllocateInfo {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    0,
    .allocationSize = memRequirements.size,
    .memoryTypeIndex = memoryTypeIndex,
  };
  CHK(vkAllocateMemory(device, &memoryAllocateInfo, 0, &memory));
  CHK(vkBindBufferMemory(device, buffer, memory, 0));
  return VK_SUCCESS;
}
}// namespace {

struct PhysicalDeviceMemory
{
  PhysicalDeviceMemory(VkPhysicalDevice& device) {
    vkGetPhysicalDeviceMemoryProperties(device, &props_);
  }

  VkPhysicalDeviceMemoryProperties props_;
};

struct StagingBuffer
{
  StagingBuffer(VkDevice& device, VkPhysicalDeviceMemoryProperties& props) :
  device_(device),
  physMemProps_(props),
  buffer(VK_NULL_HANDLE),
  memory(VK_NULL_HANDLE) {}
  ~StagingBuffer() {
    if (mapped != nullptr) {
      vkUnmapMemory(device_, memory);
      mapped = nullptr;
    }
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory, nullptr);
    }
  }

  VkDevice device_;
  VkPhysicalDeviceMemoryProperties physMemProps_;;
  VkBuffer buffer;
  VkDeviceMemory memory;
  void* mapped;

  VkResult allocate(VkDevice device, const size_t bufferSize);
};

VkResult StagingBuffer::allocate(VkDevice device, const size_t bufferSize) {
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  createBuffer(device, physMemProps_,
    bufferSize, usage, props,
    buffer, memory);
  // Map the buffer memory
  CHK(vkMapMemory(device, memory, 0/*offset*/, bufferSize, 0/*flags*/, &mapped));
  return VK_SUCCESS;
}

struct DeviceBuffer
{
  DeviceBuffer(VkDevice& device, VkPhysicalDeviceMemoryProperties& props) :
  device_(device),
  physMemProps_(props),
  buffer(VK_NULL_HANDLE),
  memory(VK_NULL_HANDLE) {}
  ~DeviceBuffer() {
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory, nullptr);
      memory = VK_NULL_HANDLE;
    }
  }

  VkDevice device_;
  VkPhysicalDeviceMemoryProperties physMemProps_;;
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkResult allocate(VkDevice device, const size_t bufferSize);
};

VkResult DeviceBuffer::allocate(VkDevice device, const size_t bufferSize) {
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  createBuffer(device, physMemProps_,
    bufferSize, usage, props,
    buffer, memory);
  return VK_SUCCESS;
}

class Application {
public:
  Application() :
  vkInstance_(VK_NULL_HANDLE),
  vkPhysicalDevice_(VK_NULL_HANDLE),
  vkDevice_(VK_NULL_HANDLE),
  commandPool_(VK_NULL_HANDLE),
  descriptorPool_(VK_NULL_HANDLE),
  commandBuffer_(VK_NULL_HANDLE) {}
  ~Application() {
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
      vkFreeCommandBuffers(vkDevice_, commandPool_, 1, &commandBuffer_);
    }
    if (descriptorPool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(vkDevice_, descriptorPool_, nullptr);
    }
    if (commandPool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(vkDevice_, commandPool_, nullptr);
    }
    if (vkDevice_ != VK_NULL_HANDLE) {
      CHK(vkDeviceWaitIdle(vkDevice_));
      std::cout << "==== Destroy device ====" << std::endl;
      vkDestroyDevice(vkDevice_, nullptr);
    }
    if (vkInstance_ != VK_NULL_HANDLE) {
      std::cout << "==== Destroy vulkan instance ====" << std::endl;
      vkDestroyInstance(vkInstance_, nullptr);
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

    // std::vector<const char*> layers;
    // std::vector<const char*> extensions;
    // if (gUseValidation)
    // {
    //   layers.push_back("VK_LAYER_KHRONOS_validation");
    //   extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    // }

    // ci.enabledExtensionCount = uint32_t(extensions.size());
    // ci.ppEnabledExtensionNames = extensions.data();
    // ci.enabledLayerCount = uint32_t(layers.size());
    // ci.ppEnabledLayerNames = layers.data();
    CHK(vkCreateInstance(&ci, nullptr, &vkInstance_));

    uint32_t version = 0;
    vkEnumerateInstanceVersion(&version);
    std::cout << "Vulkan Instance version: " <<
          VK_VERSION_MAJOR(version) << "." <<
          VK_VERSION_MINOR(version) << "." <<
          VK_VERSION_PATCH(version) << std::endl;

    std::cout << "==== Create physical device ====" << std::endl;
    uint32_t count = 0;
    CHK(vkEnumeratePhysicalDevices(vkInstance_, &count, nullptr));
    std::vector<VkPhysicalDevice> physDevs(count);
    CHK(vkEnumeratePhysicalDevices(vkInstance_, &count, physDevs.data()));
    for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(physDevs[i], &props);
      std::cout << "GPU " << i << ": " << props.deviceName << " "
                << VK_VERSION_MAJOR(props.apiVersion) << "."
                << VK_VERSION_MINOR(props.apiVersion) << "." 
                << VK_VERSION_PATCH(props.apiVersion) << std::endl;
    }

    // use gpu[0]
    vkPhysicalDevice_ = physDevs[0];

    // Get memory properties
    physDevMemory_.reset(new PhysicalDeviceMemory(vkPhysicalDevice_));

    // Get queue family properties
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProps(count);
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &count, queueFamilyProps.data());
    uint32_t computeQueueIndex = 0;
    for (uint32_t i = 0; i < count; ++i) {
      if (queueFamilyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeQueueIndex = i;
      }
    }

    std::cout << "==== Create device ====" << std::endl;
    const float queuePrioritory = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCI{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = computeQueueIndex,
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

    CHK(vkCreateDevice(vkPhysicalDevice_, &deviceCI, nullptr, &vkDevice_));

    vkGetDeviceQueue(vkDevice_, computeQueueIndex, 0, &computeQueue_);

    std::cout << "==== Create command pool ====" << std::endl;
    VkCommandPoolCreateInfo commandPoolCI{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = computeQueueIndex,
    };
    CHK(vkCreateCommandPool(vkDevice_, &commandPoolCI, nullptr, &commandPool_));

    std::cout << "==== Create descriptor pool ====" << std::endl;
    // A descriptor pool holds a sufficient 
    // number of descriptors to be used by the application, 
    // and when a descriptor set is allocated, it is cut out of the pool.
    const uint32_t descCount = 100;
    std::vector<VkDescriptorPoolSize> poolSizes = { {
      {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = descCount,
      },
    } };
    VkDescriptorPoolCreateInfo descriptorPoolCI{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = descCount,
      .poolSizeCount = uint32_t(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
    };
    CHK(vkCreateDescriptorPool(vkDevice_, &descriptorPoolCI, nullptr, &descriptorPool_));
  }

  void run() {

    initialize();
    
    std::cout << "==== Allocate buffer & memory ====" << std::endl;
    // Allocate input and output buffers
    const uint32_t numElements = 32;
	  const uint32_t bufferSize = numElements * sizeof(int32_t);

    const VkDeviceSize memorySize = bufferSize;
    
    inputBuffer_.reset(new StagingBuffer(vkDevice_, physDevMemory_->props_));
    outputBuffer_.reset(new StagingBuffer(vkDevice_, physDevMemory_->props_));
    d_inputBuffer_.reset(new DeviceBuffer(vkDevice_, physDevMemory_->props_));
    d_outputBuffer_.reset(new DeviceBuffer(vkDevice_, physDevMemory_->props_));

    inputBuffer_->allocate(vkDevice_, memorySize);
    outputBuffer_->allocate(vkDevice_, memorySize);

    d_inputBuffer_->allocate(vkDevice_, memorySize);
    d_outputBuffer_->allocate(vkDevice_, memorySize);

    // Set input data.
    for (uint32_t i = 0; i < numElements; ++i) {
      reinterpret_cast<int32_t*>(inputBuffer_->mapped)[i] = i;
    }

    std::cout << "==== Create descriptor set layout & pipeline layout ====" << std::endl;
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings{
      {// input buffer
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {// output buffer
        .binding = 1,
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
    CHK(vkCreateDescriptorSetLayout(vkDevice_, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout_));

    VkPipelineLayoutCreateInfo pipelineLayoutCI{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptorSetLayout_,
    };
    CHK(vkCreatePipelineLayout(vkDevice_, &pipelineLayoutCI, nullptr, &pipelineLayout_));

    std::cout << "==== Create shader module & pipeline ====" << std::endl;
    std::vector<char> computeSpv;
    Load("build/shaders/shader.comp.spv", computeSpv);
    VkPipelineShaderStageCreateInfo computeStageCI {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = createShaderModule(vkDevice_, computeSpv.data(), computeSpv.size()),
      .pName = "main",
    };
    VkComputePipelineCreateInfo computePipelineCI{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = computeStageCI,
      .layout = pipelineLayout_,
    };
    CHK(vkCreateComputePipelines(vkDevice_, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &pipeline_));

    std::cout << "==== Allocate descriptor set ====" << std::endl;
    std::vector<VkDescriptorSetLayout> dsLayouts{ descriptorSetLayout_ };
    VkDescriptorSetAllocateInfo dsAllocInfoComp = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool_,
      .descriptorSetCount = uint32_t(dsLayouts.size()),
      .pSetLayouts = dsLayouts.data(),
    };

    descriptorSets_.resize(dsLayouts.size());
    CHK(vkAllocateDescriptorSets(vkDevice_, &dsAllocInfoComp, descriptorSets_.data()));

    VkDescriptorBufferInfo inputBufferInfo{
      .buffer = d_inputBuffer_->buffer,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo outputBufferInfo{
      .buffer = d_outputBuffer_->buffer,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    std::vector<VkWriteDescriptorSet> writeDescriptorSets{
      {// input buffer
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptorSets_[0],
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &inputBufferInfo,
      },
      {// output buffer
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptorSets_[0],
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &outputBufferInfo,
      },
    };
    vkUpdateDescriptorSets(vkDevice_, uint32_t(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

    std::cout << "==== Create command buffer ====" << std::endl;
  
    VkCommandBufferAllocateInfo commandAI{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = commandPool_,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
    };
    CHK(vkAllocateCommandBuffers(vkDevice_, &commandAI, &commandBuffer_));

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
    vkCmdCopyBuffer(commandBuffer_, inputBuffer_->buffer, d_inputBuffer_->buffer, 1, &copyRegion);

    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSets_[0], 0, nullptr);
    vkCmdDispatch(commandBuffer_, numElements / 32, 1, 1);

    vkCmdCopyBuffer(commandBuffer_, d_outputBuffer_->buffer, outputBuffer_->buffer, 1, &copyRegion);
    CHK(vkEndCommandBuffer(commandBuffer_));

    VkSubmitInfo VkSubmitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer_,
    };
    CHK(vkQueueSubmit(computeQueue_, 1, &VkSubmitInfo, VK_NULL_HANDLE));
    CHK(vkQueueWaitIdle(computeQueue_));

    for (uint32_t i = 0; i < numElements; ++i) {
      std::cout << reinterpret_cast<int32_t*>(inputBuffer_->mapped)[i] << " ";
    }
    std::cout << std::endl;

    for (uint32_t i = 0; i < numElements; ++i) {
      std::cout << reinterpret_cast<int32_t*>(outputBuffer_->mapped)[i] << " ";
    }
    std::cout << std::endl;
  }

private:
  VkInstance vkInstance_;
  VkPhysicalDevice vkPhysicalDevice_;
  std::unique_ptr<PhysicalDeviceMemory> physDevMemory_;
  VkDevice vkDevice_;
  VkQueue computeQueue_;
  VkCommandPool commandPool_;
  VkDescriptorPool descriptorPool_;
  VkCommandBuffer commandBuffer_;
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

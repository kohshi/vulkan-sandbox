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
VkShaderModule CreateShaderModule(VkDevice device, const void* code, size_t length)
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

  void run() {
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
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_, &physDevmemoryProps_);

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

    std::cout << "==== Create command buffer ====" << std::endl;
  
    VkCommandBufferAllocateInfo commandAI{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = commandPool_,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
    };
    CHK(vkAllocateCommandBuffers(vkDevice_, &commandAI, &commandBuffer_));



    std::cout << "==== Allocate buffer & memory ====" << std::endl;

    // input buffer
    const uint32_t numElements = 10;
	  const uint32_t bufferSize = numElements * sizeof(int32_t);

    const VkDeviceSize memorySize = bufferSize * 2; // 2 = in + out
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;
    for (uint32_t i = 0; i < physDevmemoryProps_.memoryTypeCount; ++i) {
      if ((physDevmemoryProps_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          (physDevmemoryProps_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
        (memorySize < physDevmemoryProps_.memoryHeaps[physDevmemoryProps_.memoryTypes[i].heapIndex].size)) {
        memoryTypeIndex = i;
        break;
      }
    }

    // Check if memory type is found
    CHK(((memoryTypeIndex == VK_MAX_MEMORY_TYPES) ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS));
    const VkMemoryAllocateInfo memoryAllocateInfo {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      0,
      .allocationSize = memorySize,
      .memoryTypeIndex = memoryTypeIndex,
    };
    VkDeviceMemory memory;
    CHK(vkAllocateMemory(vkDevice_, &memoryAllocateInfo, 0, &memory));

    // use map memory to copy data in easy way
    int32_t *data;
    CHK(vkMapMemory(vkDevice_, memory, 0, memorySize, 0, (void**)&data));
    for (uint32_t i = 0; i < numElements; ++i) {
      data[i] = i;
    }
    vkUnmapMemory(vkDevice_, memory);

    VkBufferCreateInfo bufferCreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bufferSize,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    };
    
    CHK(vkCreateBuffer(vkDevice_, &bufferCreateInfo, nullptr, &inputBuffer_));
    CHK(vkBindBufferMemory(vkDevice_, inputBuffer_, memory, 0));

    // output buffer
    CHK(vkCreateBuffer(vkDevice_, &bufferCreateInfo, nullptr, &outputBuffer_));

    std::cout << "==== Create descriptor set ====" << std::endl;
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
      .module = CreateShaderModule(vkDevice_, computeSpv.data(), computeSpv.size()),
      .pName = "main",
    };
    VkComputePipelineCreateInfo computePipelineCI{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = computeStageCI,
      .layout = pipelineLayout_,
    };
    CHK(vkCreateComputePipelines(vkDevice_, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &pipeline_));


    // std::cout << "==== Allocate descriptor set ====" << std::endl;
    // std::vector<VkDescriptorSetLayout> dsLayouts{ descriptorSetLayout_ };
    // VkDescriptorSetAllocateInfo dsAllocInfoComp = {
    //   .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    //   .descriptorPool = descriptorPool_
    //   .descriptorSetCount = uint32_t(dsLayouts.size()),
    //   .pSetLayouts = dsLayouts.data(),
    // };
    // std::vector<VkDescriptorSet> descriptorSets(dsLayouts.size());
    // CHK(vkAllocateDescriptorSets(vkDevice, &dsAllocInfoComp, descriptorSets.data()));

  }

private:
  VkInstance vkInstance_;
  VkPhysicalDevice vkPhysicalDevice_;
  VkPhysicalDeviceMemoryProperties physDevmemoryProps_;
  VkDevice vkDevice_;
  VkQueue computeQueue_;
  VkCommandPool commandPool_;
  VkDescriptorPool descriptorPool_;
  VkCommandBuffer commandBuffer_;
  VkBuffer inputBuffer_;
  VkBuffer outputBuffer_;
  VkDescriptorSetLayout descriptorSetLayout_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline pipeline_;
  VkDescriptorSet descriptorSet_;
};

int main(int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  Application app;
  app.run();
  return 0;
}

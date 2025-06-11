#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <cstring>

#include "vulkan_utils.hpp"
#include "vk_mem_alloc.h"

namespace vk {

struct Instance {
  Instance(const bool enable_validation = false){
    const VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType
      .pApplicationName = "VulkanCompute",      // Application Name
      .pEngineName = "VulkanCompute",            // Application Version
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),    // Engine Version
      .apiVersion= VK_API_VERSION_1_2    // Vulkan API version
    };
  
    VkInstanceCreateInfo instance_ci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo
    };

    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    
    layers.push_back("VK_LAYER_KHRONOS_synchronization2");
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    if (enable_validation) {
      layers.push_back("VK_LAYER_KHRONOS_validation");
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);  
    }
    instance_ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instance_ci.ppEnabledLayerNames = layers.data();
    instance_ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instance_ci.ppEnabledExtensionNames = extensions.data();

    CHK(vkCreateInstance(
        &instance_ci, nullptr, &instance_));
    
    uint32_t version = 0;
    vkEnumerateInstanceVersion(&version);
    std::cout << "Vulkan Instance version: " <<
          VK_VERSION_MAJOR(version) << "." <<
          VK_VERSION_MINOR(version) << "." <<
          VK_VERSION_PATCH(version) << std::endl;
  }
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&& s) = delete;
  Instance& operator=(Instance&& s) = delete;
  ~Instance(){
    vkDestroyInstance(instance_, nullptr);
  }
  
  VkInstance instance_;
};

struct PhysicalDevice {
  PhysicalDevice(Instance& instance, const int gpu_idx = 0) {
    uint32_t n_phys_dev = 0;
    VkInstance vki = instance.instance_;
    CHK(vkEnumeratePhysicalDevices(vki, &n_phys_dev, nullptr));
    std::vector<VkPhysicalDevice> phys_devs(n_phys_dev);
    std::vector<VkPhysicalDeviceProperties> props(n_phys_dev);
    CHK(vkEnumeratePhysicalDevices(vki, &n_phys_dev, phys_devs.data()));
    for (uint32_t i = 0; i < n_phys_dev; ++i) {
      vkGetPhysicalDeviceProperties(phys_devs[i], &props[i]);
      std::cout << "GPU " << i << ": " << props[i].deviceName << " "
                << VK_VERSION_MAJOR(props[i].apiVersion) << "."
                << VK_VERSION_MINOR(props[i].apiVersion) << "." 
                << VK_VERSION_PATCH(props[i].apiVersion) << std::endl;
    }
    physical_device_ = phys_devs[gpu_idx];
    VkPhysicalDeviceProperties phys_device_prop = props[gpu_idx];

    const VkPhysicalDeviceLimits& limits = phys_device_prop.limits;
    std::cout << "Using GPU " << gpu_idx << ": " << phys_device_prop.deviceName << std::endl;
    std::cout << "==== Physical device limits ====" << std::endl;
    std::cout << "maxUniformBufferRange: " << limits.maxUniformBufferRange << std::endl;
    std::cout << "maxStorageBufferRange: " << limits.maxStorageBufferRange << std::endl;
    std::cout << "maxPushConstantsSize: " << limits.maxPushConstantsSize << std::endl;
    std::cout << "maxMemoryAllocationCount: " << limits.maxMemoryAllocationCount << std::endl;
    std::cout << "maxSamplerAllocationCount: " << limits.maxSamplerAllocationCount << std::endl;

    VkPhysicalDeviceMemoryProperties phys_memory_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &phys_memory_props);

    for (uint32_t i = 0; i < phys_memory_props.memoryTypeCount; ++i) {
      std::cout << "Memory type " << i << ": " << phys_memory_props.memoryTypes[i].heapIndex << std::endl;
      std::cout << "  propertyFlags: " << std::hex << phys_memory_props.memoryTypes[i].propertyFlags << std::dec << std::endl;
    }
  }
  PhysicalDevice() = delete;
  PhysicalDevice(const PhysicalDevice&) = delete;
  PhysicalDevice& operator=(const PhysicalDevice&) = delete;
  PhysicalDevice(PhysicalDevice&& s) = delete;
  PhysicalDevice& operator=(PhysicalDevice&& s) = delete;
  ~PhysicalDevice() = default;
  
  VkPhysicalDevice physical_device_;
};

struct Device {
  Device(Instance& instance, PhysicalDevice& physical_device) {
    // Get queue family properties
    uint32_t n_queue_family_props = 0;
    VkPhysicalDevice vkpd = physical_device.physical_device_;
    vkGetPhysicalDeviceQueueFamilyProperties(vkpd, &n_queue_family_props, nullptr);
    std::vector<VkQueueFamilyProperties> queue_family_props(n_queue_family_props);
    vkGetPhysicalDeviceQueueFamilyProperties(vkpd, &n_queue_family_props, queue_family_props.data());
    compute_queue_family_index_ = 0;
    uint32_t n_compute_queue = 0;
    for (uint32_t i = 0; i < n_queue_family_props; ++i) {
      if (queue_family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        compute_queue_family_index_ = i;
        n_compute_queue = queue_family_props[i].queueCount;
      }
    }
    std::cout << "computeQueueFamilyIndex: " << compute_queue_family_index_ << std::endl;
    std::cout << "computeQueueCount: " << n_compute_queue << std::endl;

    // Get supported extnsions
    uint32_t n_extensions = 0;
    vkEnumerateDeviceExtensionProperties(vkpd, nullptr, &n_extensions, nullptr);
    std::vector<VkExtensionProperties> supported_exts(n_extensions);
    vkEnumerateDeviceExtensionProperties(vkpd, nullptr, &n_extensions, supported_exts.data());

    synchronization2_supported_ = false;
    for (const auto& ext : supported_exts) {
        if (std::strcmp(ext.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) {
          synchronization2_supported_ = true;
            break;
        }
        if (std::strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0) {
          std::cout << "VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME supported" << std::endl;
        }
    }
    std::cout << "==== Supported extensions ====" << std::endl;
    std::cout << "VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME: " << synchronization2_supported_ << std::endl;

    // Create device
    std::cout << "==== Create device ====" << std::endl;
    std::vector<const char*> extensions;
    if (synchronization2_supported_) {
      extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    const float queue_priorities = 1.0f;
    VkPhysicalDeviceSynchronization2FeaturesKHR sync2_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .pNext = nullptr,
      .synchronization2 = VK_TRUE,
    };

    VkDeviceQueueCreateInfo device_queue_ci{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = compute_queue_family_index_,
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

    CHK(vkCreateDevice(vkpd, &device_ci, nullptr, &device_));

    // Get process address
    vkGetMemoryFdKHR_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
    if (synchronization2_supported_) {
      vkCmdPipelineBarrier2KHR_ = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(
      vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2KHR"));
    }

    VkPhysicalDeviceMemoryProperties phys_memory_props;
    vkGetPhysicalDeviceMemoryProperties(vkpd, &phys_memory_props);
    std::vector<VkExternalMemoryHandleTypeFlagsKHR> handle_types(phys_memory_props.memoryTypeCount, 0);
    for (uint32_t i = 0; i < phys_memory_props.memoryTypeCount; ++i) {
      if (phys_memory_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        handle_types[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      }
    }

    const VmaAllocatorCreateInfo allocator_info = {
      .flags = 0,
      .physicalDevice = vkpd,
      .device = device_,
      .instance = instance.instance_,
      .vulkanApiVersion = VK_API_VERSION_1_2,
      .pTypeExternalMemoryHandleTypes = handle_types.data(),
    };
    CHK(vmaCreateAllocator(&allocator_info, &allocator_));
  }
  Device() = delete;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&& s) = delete;
  Device& operator=(Device&& s) = delete;
  ~Device() {
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, nullptr);
  }
  
  VkDevice device_;
  bool synchronization2_supported_;
  uint32_t compute_queue_family_index_;
  VmaAllocator allocator_;

  PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR_ = nullptr;
  PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR_ = nullptr;
};

struct ComputeQueue {
  ComputeQueue(Device& device) {
    vkGetDeviceQueue(device.device_, device.compute_queue_family_index_, 0/*queueIndex*/, &queue_);
  };
  ComputeQueue() = delete;
  ComputeQueue(const ComputeQueue&) = delete;
  ComputeQueue& operator=(const ComputeQueue&) = delete;
  ComputeQueue(ComputeQueue&& s) = delete;
  ComputeQueue& operator=(ComputeQueue&& s) = delete;
  ~ComputeQueue() = default;
  
  VkQueue queue_;
};

struct CommandPool {
  CommandPool(Device& device):
  device_(device) {
    VkCommandPoolCreateInfo command_pool_ci{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = device_.compute_queue_family_index_,
    };
    CHK(vkCreateCommandPool(device_.device_, &command_pool_ci, nullptr, &command_pool_));
  }
  CommandPool() = delete;
  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;
  CommandPool(CommandPool&& s) = delete;
  CommandPool& operator=(CommandPool&& s) = delete;
  ~CommandPool() {
    vkDestroyCommandPool(device_.device_, command_pool_, nullptr);
  }
  
  VkCommandPool command_pool_;
  Device& device_;
};

struct DescriptorPool {
  DescriptorPool(Device& device, const uint32_t n_descriptor):
  device_(device) {
    // A descriptor pool holds a sufficient 
    // number of descriptors to be used by the application, 
    // and when a descriptor set is allocated, it is cut out of the pool.
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
    CHK(vkCreateDescriptorPool(device_.device_, &descriptor_pool_ci, nullptr, &descriptor_pool_));
  }
  DescriptorPool() = delete;
  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;
  DescriptorPool(DescriptorPool&& s) = delete;
  DescriptorPool& operator=(DescriptorPool&& s) = delete;
  ~DescriptorPool() {
    vkDestroyDescriptorPool(device_.device_, descriptor_pool_, nullptr);
  }
  operator VkDescriptorPool() { return descriptor_pool_; }
  
  VkDescriptorPool descriptor_pool_;
  Device& device_;
};

};// namespace vk

#pragma once
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace {
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

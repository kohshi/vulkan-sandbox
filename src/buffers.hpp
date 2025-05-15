#pragma once
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace vk {

namespace {
uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& props,
    const uint32_t memory_types_bits,
    const VkMemoryPropertyFlags flags)
{
  uint32_t memory_types_index = VK_MAX_MEMORY_TYPES;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if (memory_types_bits & (1 << i)) {
      // Check if the memory type is suitable
      if ((props.memoryTypes[i].propertyFlags & flags) == flags) {
        memory_types_index = i;
        break;
      }
    }
  }
  // Check if memory type is found
  CHK(((memory_types_index == VK_MAX_MEMORY_TYPES) ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS));
  return memory_types_index;
}

VkResult createBuffer(VkDevice device, 
  const VkPhysicalDeviceMemoryProperties& phys_mem_props,
  const VkDeviceSize buffer_size,
  const VkBufferUsageFlags usage,
  const VkMemoryPropertyFlags prop_flags,
  VkBuffer& buffer,
  VkDeviceMemory& memory)
{
  VkBufferCreateInfo buffer_ci{
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = buffer_size,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  CHK(vkCreateBuffer(device, &buffer_ci, nullptr, &buffer));

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

  uint32_t memory_type_index =
  findMemoryType(phys_mem_props,
                 mem_requirements.memoryTypeBits,
                 prop_flags);
  const VkMemoryAllocateInfo memory_ai {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    0,
    .allocationSize = mem_requirements.size,
    .memoryTypeIndex = memory_type_index,
  };
  CHK(vkAllocateMemory(device, &memory_ai, 0, &memory));
  CHK(vkBindBufferMemory(device, buffer, memory, 0));
  return VK_SUCCESS;
}
}// namespace {


struct StagingBuffer
{
  StagingBuffer(Device& device,
                const VkPhysicalDeviceMemoryProperties& props,
                const size_t size = 0) :
  device_(&device),
  phys_mem_props_(props) {
    CHK(allocate(size));
  }
  StagingBuffer() = delete;
  StagingBuffer(const StagingBuffer&) = delete;
  StagingBuffer& operator=(const StagingBuffer&) = delete;
  StagingBuffer(StagingBuffer&& s) :
  device_(s.device_),
  phys_mem_props_(s.phys_mem_props_),
  buffer_(s.buffer_),
  memory_(s.memory_),
  mapped_(s.mapped_) {
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.memory_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
  }
  StagingBuffer& operator=(StagingBuffer&& s) {
    if (&s == this) return *this;
    device_ = s.device_;
    buffer_ = s.buffer_;
    memory_ = s.memory_;
    mapped_ = s.mapped_;
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.memory_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
    return *this;
  }
  ~StagingBuffer() {
    if (device_ == nullptr) return;
    VkDevice vkd = device_->device_;
    if (mapped_ != nullptr) {
      vkUnmapMemory(vkd, memory_);
      mapped_ = nullptr;
    }
    if (buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(vkd, buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(vkd, memory_, nullptr);
    }
  }

  Device* device_;
  const VkPhysicalDeviceMemoryProperties phys_mem_props_;
  VkBuffer buffer_;
  VkDeviceMemory memory_;
  void* mapped_;
private:
  VkResult allocate(const size_t size);
};

VkResult StagingBuffer::allocate(const size_t size) {
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    return VK_SUCCESS;
  }
  VkDevice vkd = device_->device_;
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  createBuffer(vkd, phys_mem_props_,
    size, usage, props,
    buffer_, memory_);
  // Map the buffer memory
  CHK(vkMapMemory(vkd, memory_, 0/*offset*/, size, 0/*flags*/, &mapped_));
  return VK_SUCCESS;
}

struct DeviceBuffer
{
  DeviceBuffer(VkDevice device,
               const VkPhysicalDeviceMemoryProperties& props,
               const size_t size = 0) :
  device_(device),
  phys_mem_props_(props) {
    CHK(allocate(size));
  }
  DeviceBuffer() = delete;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& s) :
  device_(s.device_),
  phys_mem_props_(s.phys_mem_props_),
  buffer_(s.buffer_),
  memory_(s.memory_) {
    s.device_  = VK_NULL_HANDLE;
    s.buffer_ = VK_NULL_HANDLE;
    s.memory_ = VK_NULL_HANDLE;
  }
  DeviceBuffer& operator=(DeviceBuffer&& s) {
    if (&s == this) return *this;
    device_ = s.device_;
    buffer_ = s.buffer_;
    memory_ = s.memory_;
    s.device_  = VK_NULL_HANDLE;
    s.buffer_ = VK_NULL_HANDLE;
    s.memory_ = VK_NULL_HANDLE;
    return *this;
  }
  ~DeviceBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer_, nullptr);
      buffer_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory_, nullptr);
      memory_ = VK_NULL_HANDLE;
    }
  }

  VkDevice device_;
  const VkPhysicalDeviceMemoryProperties phys_mem_props_;;
  VkBuffer buffer_;
  VkDeviceMemory memory_;
private:
  VkResult allocate(const size_t size);
};

VkResult DeviceBuffer::allocate(const size_t size) {
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    return VK_SUCCESS;
  }
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  createBuffer(device_, phys_mem_props_,
    size, usage, props,
    buffer_, memory_);
  return VK_SUCCESS;
}

struct UniformBuffer
{
  UniformBuffer(VkDevice device,
                const VkPhysicalDeviceMemoryProperties& props,
                const size_t size = 0) :
  device_(device),
  phys_mem_props_(props) {
    CHK(allocate(size));
  }
  UniformBuffer() = delete;
  UniformBuffer(const UniformBuffer&) = delete;
  UniformBuffer& operator=(const UniformBuffer&) = delete;
  UniformBuffer(UniformBuffer&& s) :
  device_(s.device_),
  phys_mem_props_(s.phys_mem_props_),
  buffer_(s.buffer_),
  memory_(s.memory_),
  mapped_(s.mapped_) {
    s.device_  = VK_NULL_HANDLE;
    s.buffer_ = VK_NULL_HANDLE;
    s.memory_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
  }
  UniformBuffer& operator=(UniformBuffer&& s) {
    if (&s == this) return *this;
    device_ = s.device_;
    buffer_ = s.buffer_;
    memory_ = s.memory_;
    mapped_ = s.mapped_;
    s.device_  = VK_NULL_HANDLE;
    s.buffer_ = VK_NULL_HANDLE;
    s.memory_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
    return *this;
  }
  ~UniformBuffer() {
    if (mapped_ != nullptr) {
      vkUnmapMemory(device_, memory_);
      mapped_ = nullptr;
    }
    if (buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory_, nullptr);
    }
  }

  VkDevice device_;
  const VkPhysicalDeviceMemoryProperties phys_mem_props_;
  VkBuffer buffer_;
  VkDeviceMemory memory_;
  void* mapped_;
private:
  VkResult allocate(const size_t size);
};

VkResult UniformBuffer::allocate(const size_t size) {
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    return VK_SUCCESS;
  }
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  createBuffer(device_, phys_mem_props_,
    size, usage, props,
    buffer_, memory_);
  // Map the buffer memory
  CHK(vkMapMemory(device_, memory_, 0/*offset*/, size, 0/*flags*/, &mapped_));
  return VK_SUCCESS;
}
};// namespace vk

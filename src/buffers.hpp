#pragma once
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

namespace vk {

namespace {

VkResult createBuffer(VmaAllocator allocator,
  const VkDeviceSize buffer_size,
  const VkBufferUsageFlags usage,
  const VkMemoryPropertyFlags prop_flags,
  const VmaAllocatorCreateFlags alloc_flags,
  VkBuffer& buffer,
  VmaAllocation& allocation)
{
  VkBufferCreateInfo buffer_ci{
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = buffer_size,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  VmaAllocationCreateInfo vma_alloc_ci{
    .flags = alloc_flags,
    .usage = VMA_MEMORY_USAGE_AUTO,
    .requiredFlags = prop_flags,
  };
  CHK(vmaCreateBuffer(allocator, &buffer_ci, &vma_alloc_ci, &buffer, &allocation, nullptr));
  return VK_SUCCESS;
}

}// namespace {


struct StagingBuffer
{
  StagingBuffer(Device& device,
                const size_t size = 0) :
  device_(&device) {
    CHK(allocate(size));
  }
  StagingBuffer() = delete;
  StagingBuffer(const StagingBuffer&) = delete;
  StagingBuffer& operator=(const StagingBuffer&) = delete;
  StagingBuffer(StagingBuffer&& s) :
  device_(s.device_),
  buffer_(s.buffer_),
  allocation_(s.allocation_),
  mapped_(s.mapped_) {
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
  }
  StagingBuffer& operator=(StagingBuffer&& s) {
    if (&s == this) return *this;
    device_ = s.device_;
    buffer_ = s.buffer_;
    allocation_ = s.allocation_;
    mapped_ = s.mapped_;
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
    return *this;
  }
  ~StagingBuffer() {
    free();
  }

  Device* device_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
  void* mapped_;
private:
  void free();
  VkResult allocate(const size_t size);
};

void StagingBuffer::free() {
  if (device_ == nullptr) return;
  if (mapped_ != nullptr) {
    vmaUnmapMemory(device_->allocator_, allocation_);
    mapped_ = nullptr;
  }
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(device_->allocator_, buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
  }
}

VkResult StagingBuffer::allocate(const size_t size) {
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    return VK_SUCCESS;
  }
  
  VmaAllocator allocator = device_->allocator_;
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VmaAllocationCreateFlags alloc_flags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  createBuffer(allocator,
    size, usage, props,
    alloc_flags,
    buffer_, allocation_);
  // Map the buffer memory
  CHK(vmaMapMemory(allocator, allocation_, &mapped_));
  return VK_SUCCESS;
}

struct DeviceBuffer
{
  DeviceBuffer(Device& device,
               const size_t size = 0) :
  device_(&device) {
    CHK(allocate(size));
  }
  DeviceBuffer() = delete;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& s) :
  device_(s.device_),
  buffer_(s.buffer_),
  allocation_(s.allocation_) {
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
  }
  DeviceBuffer& operator=(DeviceBuffer&& s) {
    if (&s == this) return *this;
    device_ = s.device_;
    buffer_ = s.buffer_;
    allocation_ = s.allocation_;
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
    return *this;
  }
  ~DeviceBuffer() {
    free();
  }

  Device* device_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
private:
  void free();
  VkResult allocate(const size_t size);
};

void DeviceBuffer::free() {
  if (device_ == nullptr) return;
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(device_->allocator_, buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
  }
}

VkResult DeviceBuffer::allocate(const size_t size) {
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    return VK_SUCCESS;
  }

  VmaAllocator allocator = device_->allocator_;
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VmaAllocationCreateFlags alloc_flags = 0;
  createBuffer(allocator,
    size, usage, props,
    alloc_flags,
    buffer_, allocation_);
  return VK_SUCCESS;
}

struct UniformBuffer
{
  UniformBuffer(Device& device,
                const size_t size = 0) :
  device_(&device) {
    CHK(allocate(size));
  }
  UniformBuffer() = delete;
  UniformBuffer(const UniformBuffer&) = delete;
  UniformBuffer& operator=(const UniformBuffer&) = delete;
  UniformBuffer(UniformBuffer&& s) :
  device_(s.device_),
  buffer_(s.buffer_),
  allocation_(s.allocation_),
  mapped_(s.mapped_) {
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
  }
  UniformBuffer& operator=(UniformBuffer&& s) {
    if (&s == this) return *this;
    device_ = s.device_;
    buffer_ = s.buffer_;
    allocation_ = s.allocation_;
    mapped_ = s.mapped_;
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
    s.mapped_ = nullptr;
    return *this;
  }
  ~UniformBuffer() {
    free();
  }

  Device* device_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
  void* mapped_;
private:
  void free();
  VkResult allocate(const size_t size);
};

void UniformBuffer::free() {
  if (device_ == nullptr) return;
  if (mapped_ != nullptr) {
    vmaUnmapMemory(device_->allocator_, allocation_);
    mapped_ = nullptr;
  }
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(device_->allocator_, buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
  }
}

VkResult UniformBuffer::allocate(const size_t size) {
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    return VK_SUCCESS;
  }
  VmaAllocator allocator = device_->allocator_;
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags props =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VmaAllocationCreateFlags alloc_flags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  createBuffer(allocator,
    size, usage, props,
    alloc_flags,
    buffer_, allocation_);
  // Map the buffer memory
  CHK(vmaMapMemory(allocator, allocation_, &mapped_));
  return VK_SUCCESS;
}
};// namespace vk

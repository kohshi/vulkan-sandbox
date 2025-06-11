#pragma once
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

#include <cassert>

namespace vk {

namespace {

VkResult create_buffer(VmaAllocator allocator,
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

VkResult create_export_device_buffer(VmaAllocator allocator,
  const VkDeviceSize buffer_size,
  VkBuffer& buffer,
  VmaAllocation& allocation)
{
  VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkExternalMemoryBufferCreateInfo external_memory_ci{
    .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
  };

  VkBufferCreateInfo buffer_ci{
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .pNext = &external_memory_ci,
    .size = buffer_size,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VmaAllocationCreateInfo vma_alloc_ci{
    .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,// needed for cuda interop
    .usage = VMA_MEMORY_USAGE_GPU_ONLY,
    .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  CHK(vmaCreateBuffer(allocator, &buffer_ci, &vma_alloc_ci, &buffer, &allocation, nullptr));
  return VK_SUCCESS;
}

}// namespace {


struct StagingBuffer
{
  StagingBuffer(Device& device,
                const size_t size);
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
  ~StagingBuffer();

  Device* device_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
  void* mapped_;
private:
  void free();
  VkResult allocate(const size_t size);
};

StagingBuffer::StagingBuffer(Device& device, const size_t size) :
device_(&device)
{
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    return;
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
  create_buffer(allocator,
    size, usage, props,
    alloc_flags,
    buffer_, allocation_);
  // Map the buffer memory
  CHK(vmaMapMemory(allocator, allocation_, &mapped_));
}

StagingBuffer::~StagingBuffer() {
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

struct DeviceBuffer
{
  DeviceBuffer(Device& device,
              const size_t size,
              const bool export_fd);
  DeviceBuffer() = delete;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& s) :
  device_(s.device_),
  buffer_(s.buffer_),
  allocation_(s.allocation_),
  fd_(s.fd_) {
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
    s.fd_ = -1;
  }
  DeviceBuffer& operator=(DeviceBuffer&& s) {
    if (&s == this) return *this;
    device_ = s.device_;
    buffer_ = s.buffer_;
    allocation_ = s.allocation_;
    fd_ = s.fd_;
    s.device_  = nullptr;
    s.buffer_ = VK_NULL_HANDLE;
    s.allocation_ = VK_NULL_HANDLE;
    s.fd_ = -1;
    return *this;
  }
  ~DeviceBuffer();

  Device* device_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
  int fd_ = -1;
private:
  void free();
  VkResult allocate(const size_t size);
};

DeviceBuffer::DeviceBuffer(Device& device,
  const size_t size,
  const bool export_fd) :
device_(&device)
{
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    return;
  }

  VmaAllocator allocator = device_->allocator_;
  if (export_fd) {
    create_export_device_buffer(allocator,
      size,
      buffer_, allocation_);

    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(allocator, allocation_, &alloc_info);
    VkMemoryGetFdInfoKHR get_fd_info{
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = alloc_info.deviceMemory,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    CHK(device_->vkGetMemoryFdKHR_(device_->device_, &get_fd_info, &fd_));
    assert(fd_ > 0);
  } else {
    VkBufferUsageFlags usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags props =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VmaAllocationCreateFlags alloc_flags = 0;
    create_buffer(allocator,
      size, usage, props,
      alloc_flags,
      buffer_, allocation_);
  }
}

DeviceBuffer::~DeviceBuffer() {
  if (device_ == nullptr) return;
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(device_->allocator_, buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
  }
}

struct UniformBuffer
{
  UniformBuffer(Device& device,
                const size_t size);
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
  ~UniformBuffer();

  Device* device_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
  void* mapped_;
};

UniformBuffer::UniformBuffer(Device& device, const size_t size) :
device_(nullptr)
{
  if (size == 0) {
    // disallowed 0 size memory
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    return;
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
  create_buffer(allocator,
    size, usage, props,
    alloc_flags,
    buffer_, allocation_);
  // Map the buffer memory
  CHK(vmaMapMemory(allocator, allocation_, &mapped_));
}

UniformBuffer::~UniformBuffer() {
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

};// namespace vk

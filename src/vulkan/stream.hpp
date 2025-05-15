#pragma once

#include <vulkan/vulkan.h>

#include "kernel.hpp"

namespace vk {

struct Stream {
  Stream(Device& device, ComputeQueue& queue, CommandPool& command_pool);
  Stream() = delete;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& s) = delete;
  Stream& operator=(Stream&& s) = delete;
  ~Stream() {
    if (timeline_semaphore_ != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_.device_, timeline_semaphore_, nullptr);
    }
  }

  void begin();
  void copy(VkBuffer src, VkBuffer dst, VkDeviceSize size);
  template<typename T>
  void dispatch(ComputeShader<T>& cs, uint32_t group_x, uint32_t group_y, uint32_t group_z);
  void barrier();
  void submit();
  void synchronize();

private:
  void wait(const uint64_t wait_value);
  void check();

public:
  Device& device_;
  ComputeQueue& queue_;
  CommandPool& command_pool_;
  VkCommandBuffer current_command_buf_;
  std::vector<VkCommandBuffer> submitted_command_bufs_;
  VkSemaphore timeline_semaphore_;
  uint64_t timeline_value_;

  const bool synchronize2_supported_ = true;
  PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR_ = nullptr;
};

Stream::Stream(Device& device, ComputeQueue& queue, CommandPool& command_pool) :
  device_(device),
  queue_(queue),
  command_pool_(command_pool),
  current_command_buf_(VK_NULL_HANDLE),
  timeline_semaphore_(VK_NULL_HANDLE),
  timeline_value_(0) {
    VkSemaphoreTypeCreateInfo timeline_semaphore_ci{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
    };
    VkSemaphoreCreateInfo semaphoreCI{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timeline_semaphore_ci,
    };
    CHK(vkCreateSemaphore(device_.device_, &semaphoreCI, nullptr, &timeline_semaphore_));
  
  if (synchronize2_supported_) {
    // Get process addresses
    vkCmdPipelineBarrier2KHR_ = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(
      vkGetDeviceProcAddr(device_.device_, "vkCmdPipelineBarrier2KHR"));
  }
}

void Stream::begin() {
  VkCommandBufferAllocateInfo command_buffer_ci{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = command_pool_.command_pool_,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };
  CHK(vkAllocateCommandBuffers(device_.device_, &command_buffer_ci, &current_command_buf_));
  VkCommandBufferBeginInfo begin_info{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  CHK(vkBeginCommandBuffer(current_command_buf_, &begin_info));
}

void Stream::copy(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
  check();

  VkBufferCopy copy_region{
    .srcOffset = 0,
    .dstOffset = 0,
    .size = size,
  };
  vkCmdCopyBuffer(current_command_buf_, src, dst, 1, &copy_region);
}

template<typename T>
void Stream::dispatch(ComputeShader<T>& cs, uint32_t group_x, uint32_t group_y, uint32_t group_z) {
  check();

  vkCmdBindPipeline(current_command_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, cs.pipeline_);
  vkCmdPushConstants(current_command_buf_,
    cs.pipeline_layout_,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0/*offset*/, sizeof(T), &(cs.push_constants_));
  vkCmdBindDescriptorSets(current_command_buf_,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    cs.pipeline_layout_,
    0 /*firstSet*/, 1/*descriptorSetCount*/,
    &(cs.descriptor_sets_[0]),
    0/*DynamicOffsetCount*/,
    nullptr);
  // group count x,y,z
  vkCmdDispatch(current_command_buf_, group_x, group_y, group_z);
}

void Stream::barrier() {
  check();

  if (synchronize2_supported_) {
    VkMemoryBarrier2KHR memory_barrier {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
      .srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    VkDependencyInfoKHR dependency_info{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memory_barrier,
      .bufferMemoryBarrierCount = 0,
      .pBufferMemoryBarriers = nullptr,
      .imageMemoryBarrierCount = 0,
      .pImageMemoryBarriers = nullptr,
    };
    vkCmdPipelineBarrier2KHR_(current_command_buf_, &dependency_info);
  } else {
    VkMemoryBarrier memory_barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(
      current_command_buf_,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, //srcStageMask
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, //dstStageMask
      0, //dependencyFlags
      1/*memoryBarrierCount*/, &memory_barrier, //pMemoryBarriers
      0/*bufferMemoryBarrierCount*/, nullptr, // pBufferMemoryBarriers
      0/*imageMemoryBarrierCount*/, nullptr);// pImageMemoryBarriers
  }
}

void Stream::submit() {
  check();
  CHK(vkEndCommandBuffer(current_command_buf_));

  uint64_t wait_value = timeline_value_;
  uint64_t signal_value = ++timeline_value_;

  VkTimelineSemaphoreSubmitInfo timeline_info {
    .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
    .waitSemaphoreValueCount = 1,
    .pWaitSemaphoreValues = &wait_value,
    .signalSemaphoreValueCount = 1,
    .pSignalSemaphoreValues = &signal_value,
  };

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  VkSubmitInfo submit_info {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .pNext = &timeline_info,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &timeline_semaphore_,
    .pWaitDstStageMask = &wait_stage,
    .commandBufferCount = 1,
    .pCommandBuffers = &current_command_buf_,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &timeline_semaphore_,
  };

  CHK(vkQueueSubmit(queue_.queue_, 1, &submit_info, VK_NULL_HANDLE));
  submitted_command_bufs_.push_back(current_command_buf_);
  current_command_buf_ = VK_NULL_HANDLE;
}

void Stream::synchronize() {
  wait(timeline_value_);
  // free finished command buffers
  for (auto command_buffer : submitted_command_bufs_) {
    vkFreeCommandBuffers(device_.device_, command_pool_.command_pool_, 1, &command_buffer);
  }
  submitted_command_bufs_.clear();
}

void Stream::wait(const uint64_t wait_value) {
  VkSemaphoreWaitInfo wait_info {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    .semaphoreCount = 1,
    .pSemaphores = &timeline_semaphore_,
    .pValues = &wait_value,
  };

  CHK(vkWaitSemaphores(device_.device_, &wait_info, UINT64_MAX));
  timeline_value_++;
}

void Stream::check() {
  if (current_command_buf_ == VK_NULL_HANDLE) {
    std::cerr << "Command buffer is not started" << std::endl;
    CHK(VK_ERROR_NOT_PERMITTED)
  }
}

} // namespace vk

#pragma once

#include <vulkan/vulkan.h>

#include "kernel.hpp"

namespace vk {

struct Stream {
  Stream(VkDevice device, VkQueue queue, VkCommandPool commandPool);
  ~Stream() {
    if (timelineSemaphore_ != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, timelineSemaphore_, nullptr);
    }
  }

  void begin();
  void copy(VkBuffer src, VkBuffer dst, VkDeviceSize size);
  void dispatch(ComputeShader& cs, uint32_t group_x, uint32_t group_y, uint32_t group_z);
  void barrier();
  void submit();
  void synchronize();

private:
  void wait(const uint64_t wait_value);
  void check();

public:
  VkDevice device_;
  VkQueue queue_;
  VkCommandPool commandPool_;
  VkCommandBuffer currentCommandBuf_;
  std::vector<VkCommandBuffer> submittedCommandBufs_;
  VkSemaphore timelineSemaphore_;
  uint64_t timeline_value_;

  const bool synchronize2Supported_ = true;
  PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR_ = nullptr;
};

Stream::Stream(VkDevice device, VkQueue queue, VkCommandPool commandPool) :
  device_(device),
  queue_(queue),
  commandPool_(commandPool),
  currentCommandBuf_(VK_NULL_HANDLE),
  timelineSemaphore_(VK_NULL_HANDLE),
  timeline_value_(0) {
    VkSemaphoreTypeCreateInfo timelineSemaphoreCI{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
    };
    VkSemaphoreCreateInfo semaphoreCI{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timelineSemaphoreCI,
    };
    CHK(vkCreateSemaphore(device_, &semaphoreCI, nullptr, &timelineSemaphore_));
  
  if (synchronize2Supported_) {
    // Get process addresses
    vkCmdPipelineBarrier2KHR_ = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(
      vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2KHR"));
  }
}

void Stream::begin() {
  VkCommandBufferAllocateInfo commandBufferCI{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = commandPool_,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };
  CHK(vkAllocateCommandBuffers(device_, &commandBufferCI, &currentCommandBuf_));
  VkCommandBufferBeginInfo beginInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  CHK(vkBeginCommandBuffer(currentCommandBuf_, &beginInfo));
}

void Stream::copy(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
  check();

  VkBufferCopy copyRegion{
    .srcOffset = 0,
    .dstOffset = 0,
    .size = size,
  };
  vkCmdCopyBuffer(currentCommandBuf_, src, dst, 1, &copyRegion);
}

void Stream::dispatch(ComputeShader& cs, uint32_t group_x, uint32_t group_y, uint32_t group_z) {
  check();

  vkCmdBindPipeline(currentCommandBuf_, VK_PIPELINE_BIND_POINT_COMPUTE, cs.pipeline_);
  vkCmdBindDescriptorSets(currentCommandBuf_,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    cs.pipelineLayout_,
    0 /*firstSet*/, 1/*descriptorSetCount*/,
    &(cs.descriptorSets_[0]),
    0/*DynamicOffsetCount*/,
    nullptr);
  // group count x,y,z
  vkCmdDispatch(currentCommandBuf_, group_x, group_y, group_z);
}

void Stream::barrier() {
  check();

  if (synchronize2Supported_) {
    VkMemoryBarrier2KHR memoryBarrier {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
      .srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    VkDependencyInfoKHR dependencyInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memoryBarrier,
      .bufferMemoryBarrierCount = 0,
      .pBufferMemoryBarriers = nullptr,
      .imageMemoryBarrierCount = 0,
      .pImageMemoryBarriers = nullptr,
    };
    vkCmdPipelineBarrier2KHR_(currentCommandBuf_, &dependencyInfo);
  } else {
    VkMemoryBarrier memoryBarrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(
      currentCommandBuf_,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, //srcStageMask
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, //dstStageMask
      0, //dependencyFlags
      1/*memoryBarrierCount*/, &memoryBarrier, //pMemoryBarriers
      0/*bufferMemoryBarrierCount*/, nullptr, // pBufferMemoryBarriers
      0/*imageMemoryBarrierCount*/, nullptr);// pImageMemoryBarriers
  }
}

void Stream::submit() {
  check();
  CHK(vkEndCommandBuffer(currentCommandBuf_));

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
    .pWaitSemaphores = &timelineSemaphore_,
    .pWaitDstStageMask = &wait_stage,
    .commandBufferCount = 1,
    .pCommandBuffers = &currentCommandBuf_,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &timelineSemaphore_,
  };

  CHK(vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));
  submittedCommandBufs_.push_back(currentCommandBuf_);
  currentCommandBuf_ = VK_NULL_HANDLE;
}

void Stream::synchronize() {
  wait(timeline_value_);
  // free finished command buffers
  for (auto commandBuffer : submittedCommandBufs_) {
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
  }
  submittedCommandBufs_.clear();
}

void Stream::wait(const uint64_t wait_value) {
  VkSemaphoreWaitInfo wait_info {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    .semaphoreCount = 1,
    .pSemaphores = &timelineSemaphore_,
    .pValues = &wait_value,
  };

  CHK(vkWaitSemaphores(device_, &wait_info, UINT64_MAX));
  timeline_value_++;
}

void Stream::check() {
  if (currentCommandBuf_ == VK_NULL_HANDLE) {
    std::cerr << "Command buffer is not started" << std::endl;
    CHK(VK_ERROR_NOT_PERMITTED)
  }
}

} // namespace vk

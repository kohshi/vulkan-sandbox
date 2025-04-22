#pragma once

#include <vulkan/vulkan.h>

namespace vk {

struct Stream {
  Stream(VkDevice device, VkQueue queue);
  ~Stream() {
    if (timelineSemaphore_ != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, timelineSemaphore_, nullptr);
    }
  }

  void submit(VkCommandBuffer cmd_buf);
  void wait(const uint64_t wait_value);
  void synchronize();

  VkDevice device_;
  VkQueue queue_;
  VkSemaphore timelineSemaphore_;
  uint64_t timeline_value_;
};

Stream::Stream(VkDevice device, VkQueue queue) :
  device_(device),
  queue_(queue),
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
}

void Stream::submit(VkCommandBuffer cmd_buf) {
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
    .pCommandBuffers = &cmd_buf,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &timelineSemaphore_,
  };

  CHK(vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));
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

void Stream::synchronize() {
  wait(timeline_value_);
}

} // namespace vk

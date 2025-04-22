#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <tuple>
#include <filesystem>
#include <fstream>

#include "vulkan_utils.hpp"

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
}// namespace {

namespace vk {

struct ComputeShader {
  ComputeShader(VkDevice device,
    VkDescriptorPool descriptorPool,
    const char* filename);
  ~ComputeShader() {
    if (pipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
    }
    if (shaderModule_ != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device_, shaderModule_, nullptr);
    }
  }

  void bind(std::vector<std::tuple<VkDescriptorType, VkBuffer>>& argTypes);

  VkDevice device_;
  VkDescriptorPool descriptorPool_;
  VkShaderModule shaderModule_;
  VkDescriptorSetLayout descriptorSetLayout_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline pipeline_;
  std::vector<VkDescriptorSet> descriptorSets_;
};

ComputeShader::ComputeShader(VkDevice device,
  VkDescriptorPool descriptorPool,
  const char* filename) :
  device_(device),
  descriptorPool_(descriptorPool),
  shaderModule_(VK_NULL_HANDLE),
  descriptorSetLayout_(VK_NULL_HANDLE),
  pipelineLayout_(VK_NULL_HANDLE),
  pipeline_(VK_NULL_HANDLE) {

  std::vector<char> computeSpv;
  if (!Load(filename, computeSpv)) {
    std::cerr << "Failed to load shader: " << filename << std::endl;
    return;
  }
  shaderModule_ = createShaderModule(device_, computeSpv.data(), computeSpv.size());
  if (shaderModule_ == VK_NULL_HANDLE) {
    std::cerr << "Failed to create shader module" << std::endl;
  }
}

void ComputeShader::bind(std::vector<std::tuple<VkDescriptorType, VkBuffer>>& argTypes) {

  // create DescriptorSetLayout and PipelineLayout
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(argTypes.size());
  for (size_t i = 0; i < argTypes.size(); ++i) {
    VkDescriptorSetLayoutBinding binding{
      .binding = uint32_t(i),
      .descriptorType = std::get<0>(argTypes[i]),
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    bindings.push_back(binding);
  }

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = uint32_t(bindings.size()),
    .pBindings = bindings.data(),
  };
  CHK(vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout_));

  VkPipelineLayoutCreateInfo pipelineLayoutCI{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &descriptorSetLayout_,
  };
  CHK(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr, &pipelineLayout_));

  // Create Pipeline
  VkPipelineShaderStageCreateInfo computeStageCI {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = shaderModule_,
    .pName = "main",
  };
  VkComputePipelineCreateInfo computePipelineCI{
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = computeStageCI,
    .layout = pipelineLayout_,
  };

  CHK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &pipeline_));

  // Update DescriptorSet
  std::cout << "==== Allocate descriptor set ====" << std::endl;
  std::vector<VkDescriptorSetLayout> dsLayouts{descriptorSetLayout_};
  VkDescriptorSetAllocateInfo dsAllocInfoComp = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = descriptorPool_,
    .descriptorSetCount = uint32_t(dsLayouts.size()),
    .pSetLayouts = dsLayouts.data(),
  };

  descriptorSets_.resize(dsLayouts.size());
  CHK(vkAllocateDescriptorSets(device_, &dsAllocInfoComp, descriptorSets_.data()));

  std::cout << "==== Update descriptor set ====" << std::endl;
  std::vector<VkDescriptorBufferInfo> bufferInfos;
  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  bufferInfos.reserve(argTypes.size());
  writeDescriptorSets.reserve(argTypes.size());
  for (size_t i = 0; i < argTypes.size(); ++i) {
    VkDescriptorBufferInfo bufferInfo{
      .buffer = std::get<1>(argTypes[i]),
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    bufferInfos.push_back(bufferInfo);
    VkWriteDescriptorSet writeDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSets_[0],
      .dstBinding = uint32_t(i),
      .descriptorCount = 1,
      .descriptorType = std::get<0>(argTypes[i]),
      .pBufferInfo = &bufferInfos[i],
    };
    writeDescriptorSets.push_back(writeDescriptorSet);
  };
  vkUpdateDescriptorSets(device_, uint32_t(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

}

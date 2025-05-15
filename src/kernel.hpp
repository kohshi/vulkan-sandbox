#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <tuple>
#include <filesystem>
#include <fstream>

#include "vulkan_utils.hpp"

namespace {
  VkShaderModule create_shader_module(VkDevice device, const void* code, size_t length)
  {
    VkShaderModuleCreateInfo ci{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = length,
      .pCode = reinterpret_cast<const uint32_t*>(code),
    };
    VkShaderModule shader_module = VK_NULL_HANDLE;
    CHK(vkCreateShaderModule(device, &ci, nullptr, &shader_module));
    return std::move(shader_module);
  }
  
  bool Load(std::filesystem::path file_path, std::vector<char>& data) {
    if (std::filesystem::exists(file_path))
    {
      std::ifstream infile(file_path, std::ios::binary);
      if (infile)
      {
        auto size = infile.seekg(0, std::ios::end).tellg();
        data.resize(size);
        infile.seekg(0, std::ios::beg).read(data.data(), size);
        return true;
      }
    }
    file_path = std::filesystem::path("../") / file_path;
    if (std::filesystem::exists(file_path))
    {
      std::ifstream infile(file_path, std::ios::binary);
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

template<typename T>
struct ComputeShader {
  ComputeShader(Device& device,
    DescriptorPool& descriptor_pool,
    const char* filename);
  ComputeShader() = delete;
  ComputeShader(const ComputeShader&) = delete;
  ComputeShader& operator=(const ComputeShader&) = delete;
  ComputeShader(ComputeShader&& s) = delete;
  ComputeShader& operator=(ComputeShader&& s) = delete;
  ~ComputeShader() {
    VkDevice vkd = device_.device_;
    if (pipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(vkd, pipeline_, nullptr);
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(vkd, pipeline_layout_, nullptr);
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(vkd, descriptor_set_layout_, nullptr);
    }
    if (shader_module_ != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vkd, shader_module_, nullptr);
    }
  }

  void bind(T& push_constants,
    std::vector<std::tuple<VkDescriptorType, VkBuffer>>& args);

  Device& device_;
  DescriptorPool& descriptor_pool_;
  VkShaderModule shader_module_;
  VkDescriptorSetLayout descriptor_set_layout_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline pipeline_;
  std::vector<VkDescriptorSet> descriptor_sets_;
  VkPushConstantRange push_range_;
  T push_constants_;
};

template<typename T>
ComputeShader<T>::ComputeShader(Device& device,
  DescriptorPool& descriptor_pool,
  const char* filename) :
  device_(device),
  descriptor_pool_(descriptor_pool),
  shader_module_(VK_NULL_HANDLE),
  descriptor_set_layout_(VK_NULL_HANDLE),
  pipeline_layout_(VK_NULL_HANDLE),
  pipeline_(VK_NULL_HANDLE) {

  std::vector<char> compute_spv;
  if (!Load(filename, compute_spv)) {
    std::cerr << "Failed to load shader: " << filename << std::endl;
    return;
  }
  shader_module_ = create_shader_module(device_.device_, compute_spv.data(), compute_spv.size());
  if (shader_module_ == VK_NULL_HANDLE) {
    std::cerr << "Failed to create shader module" << std::endl;
  }
}

template<typename T>
void ComputeShader<T>::bind(
  T& push_constants,
  std::vector<std::tuple<VkDescriptorType, VkBuffer>>& args) {

  VkDevice vkd = device_.device_;
  // create DescriptorSetLayout and PipelineLayout
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    VkDescriptorSetLayoutBinding binding{
      .binding = uint32_t(i),
      .descriptorType = std::get<0>(args[i]),
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    bindings.push_back(binding);
  }

  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_ci{
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = uint32_t(bindings.size()),
    .pBindings = bindings.data(),
  };
  CHK(vkCreateDescriptorSetLayout(vkd, &descriptor_set_layout_ci, nullptr, &descriptor_set_layout_));

  push_range_.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range_.offset = 0;
  push_range_.size = sizeof(T);
  push_constants_ = push_constants;

  VkPipelineLayoutCreateInfo pipeline_layout_ci{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &descriptor_set_layout_,
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_range_,
  };
  CHK(vkCreatePipelineLayout(vkd, &pipeline_layout_ci, nullptr, &pipeline_layout_));

  // Create Pipeline
  VkPipelineShaderStageCreateInfo compute_stage_ci {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = shader_module_,
    .pName = "main",
  };
  VkComputePipelineCreateInfo compute_pipeline_ci{
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = compute_stage_ci,
    .layout = pipeline_layout_,
  };

  CHK(vkCreateComputePipelines(vkd, VK_NULL_HANDLE, 1, &compute_pipeline_ci, nullptr, &pipeline_));

  // Update DescriptorSet
  std::vector<VkDescriptorSetLayout> ds_layouts{descriptor_set_layout_};
  VkDescriptorSetAllocateInfo descriptor_set_ai {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = descriptor_pool_,
    .descriptorSetCount = uint32_t(ds_layouts.size()),
    .pSetLayouts = ds_layouts.data(),
  };

  descriptor_sets_.resize(ds_layouts.size());
  CHK(vkAllocateDescriptorSets(vkd, &descriptor_set_ai, descriptor_sets_.data()));

  std::vector<VkDescriptorBufferInfo> buffer_infos;
  std::vector<VkWriteDescriptorSet> write_descriptor_sets;
  buffer_infos.reserve(args.size());
  write_descriptor_sets.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    VkDescriptorBufferInfo info{
      .buffer = std::get<1>(args[i]),
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    buffer_infos.push_back(info);
    VkWriteDescriptorSet write_descriptor_set{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_sets_[0],
      .dstBinding = uint32_t(i),
      .descriptorCount = 1,
      .descriptorType = std::get<0>(args[i]),
      .pBufferInfo = &buffer_infos[i],
    };
    write_descriptor_sets.push_back(write_descriptor_set);
  };
  vkUpdateDescriptorSets(vkd, uint32_t(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
}

}

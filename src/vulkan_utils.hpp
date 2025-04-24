#pragma once
#include <vulkan/vk_enum_string_helper.h>
#include <iostream>

#define CHK(result) \
  if (result != VK_SUCCESS) { \
    std::cerr << "Vulkan API error: "<< string_VkResult(result) << " " <<  __LINE__ << " at " << __FILE__ << std::endl; \
    throw std::runtime_error(string_VkResult(result)); \
  }



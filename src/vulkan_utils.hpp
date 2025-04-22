#pragma once
#include <iostream>

#define CHK(result) \
  if (result != VK_SUCCESS) { \
    std::cerr << "Vulkan API error: "<< result << " " <<  __LINE__ << " at " << __FILE__ << std::endl; \
    exit(-1); \
  }



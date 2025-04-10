#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <iostream>

// #define VOLK_IMPLEMENTATION
// #include "volk/volk.h"

#define CHK(result) \
  if (result != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error: %d at %u %s\n", result, __LINE__, __FILE__); \
    exit(-1); \
  }

bool gUseValidation = false;

class Application {
public:
  Application() :
  vkInstance_(VK_NULL_HANDLE),
  vkPhysicalDevice_(VK_NULL_HANDLE),
  vkDevice_(VK_NULL_HANDLE) {}
  ~Application() {
    if (vkDevice_ != VK_NULL_HANDLE) {
      vkDestroyDevice(vkDevice_, nullptr);
    }
    if (vkInstance_ != VK_NULL_HANDLE) {
      vkDestroyInstance(vkInstance_, nullptr);
    }
  }

  void run() {
    std::cout << "==== Create Vulkan instance ====" << std::endl;
    const VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType
      .pApplicationName = "VulkanCompute",      // Application Name
      .pEngineName = "VulkanCompute",            // Application Version
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),    // Engine Version
      .apiVersion= VK_API_VERSION_1_3    // Vulkan API version
    };
    
    VkInstanceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo
    };

    // std::vector<const char*> layers;
    // std::vector<const char*> extensions;
    // if (gUseValidation)
    // {
    //   layers.push_back("VK_LAYER_KHRONOS_validation");
    //   extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    // }
  
    // ci.enabledExtensionCount = uint32_t(extensions.size());
    // ci.ppEnabledExtensionNames = extensions.data();
    // ci.enabledLayerCount = uint32_t(layers.size());
    // ci.ppEnabledLayerNames = layers.data();
    CHK(vkCreateInstance(&ci, nullptr, &vkInstance_));

    std::cout << "==== Create physical device ====" << std::endl;
    uint32_t count = 0;
    CHK(vkEnumeratePhysicalDevices(vkInstance_, &count, nullptr));
    std::vector<VkPhysicalDevice> physDevs(count);
    CHK(vkEnumeratePhysicalDevices(vkInstance_, &count, physDevs.data()));

    // use gpu[0]
    vkPhysicalDevice_ = physDevs[0];

    // Get memory properties
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_, &physDevmemoryProps_);

    // Get queue family properties
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProps(count);
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &count, queueFamilyProps.data());
    uint32_t computeQueueIndex = 0;
    for (uint32_t i = 0; i < count; ++i) {
      if (queueFamilyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeQueueIndex = i;
      }
    }

    std::cout << "==== Create device ====" << std::endl;
    const float queuePrioritory = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCI{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = computeQueueIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePrioritory,
    };
    VkDeviceCreateInfo deviceCI{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCI,
      .enabledExtensionCount = 0,
      .ppEnabledExtensionNames = 0,
    };

    CHK(vkCreateDevice(vkPhysicalDevice_, &deviceCI, nullptr, &vkDevice_));
  }

private:
  VkInstance vkInstance_;
  VkPhysicalDevice vkPhysicalDevice_;
  VkPhysicalDeviceMemoryProperties physDevmemoryProps_;
  VkDevice vkDevice_;
};

int main(int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  Application app;
  app.run();
  return 0;
}

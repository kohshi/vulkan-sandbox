python3 - <<EOF
import ctypes as c
lib = c.cdll.LoadLibrary("libvulkan.so.1")

# Vulkan loader version (from loader, not GPU)
ver = c.c_uint32(0)
try: lib.vkEnumerateInstanceVersion(c.byref(ver))
except AttributeError: ver.value = 0x00400000
loader_major, loader_minor, loader_patch = (ver.value >> 22) & 0x3FF, (ver.value >> 12) & 0x3FF, ver.value & 0xFFF

# Create dummy Vulkan instance
ci = (c.c_uint32 * 17)(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
inst = c.c_void_p()
lib.vkCreateInstance(c.byref(ci), None, c.byref(inst))

# Enumerate physical devices
count = c.c_uint32()
lib.vkEnumeratePhysicalDevices(inst, c.byref(count), None)
devs = (c.c_void_p * count.value)()
lib.vkEnumeratePhysicalDevices(inst, c.byref(count), devs)

# Define VkPhysicalDeviceProperties struct
class Props(c.Structure):
    _fields_ = [
        ("apiVersion", c.c_uint32),
        ("driverVersion", c.c_uint32),
        ("vendorID", c.c_uint32),
        ("deviceID", c.c_uint32),
        ("deviceType", c.c_uint32),
        ("deviceName", c.c_char * 256),
        ("uuid", c.c_uint8 * 16),
        ("limits", c.c_uint8 * 512),
        ("sparse", c.c_uint8 * 48)
    ]

# Query first GPU's properties
props = Props()
lib.vkGetPhysicalDeviceProperties(devs[0], c.byref(props))
gpu_api = props.apiVersion
gpu_major, gpu_minor, gpu_patch = (gpu_api >> 22) & 0x3FF, (gpu_api >> 12) & 0x3FF, gpu_api & 0xFFF

# Output
print(f"Loader API version: {loader_major}.{loader_minor}.{loader_patch}")
print(f"GPU Vulkan API version: {gpu_major}.{gpu_minor}.{gpu_patch}")

lib.vkDestroyInstance(inst, None)
EOF
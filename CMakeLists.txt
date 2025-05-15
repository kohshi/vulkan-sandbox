cmake_minimum_required(VERSION 3.13)

project(
    vulkan-sandbox
    VERSION 0.0.1
	DESCRIPTION "A minimal application with Vulkan Compute Shaders"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g")

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/src/ext")

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find VMA
find_package(VulkanMemoryAllocator REQUIRED)

#######################################################################
#                               SHADERS                               #
#######################################################################

find_program(glslc_executable NAMES glslc)
find_program(glslangValidator_executable NAMES glslangValidator)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB SHADERS
  ${SHADER_SOURCE_DIR}/*.vert
  ${SHADER_SOURCE_DIR}/*.frag
  ${SHADER_SOURCE_DIR}/*.comp
  ${SHADER_SOURCE_DIR}/*.geom
  ${SHADER_SOURCE_DIR}/*.tesc
  ${SHADER_SOURCE_DIR}/*.tese
  ${SHADER_SOURCE_DIR}/*.mesh
  ${SHADER_SOURCE_DIR}/*.task
  ${SHADER_SOURCE_DIR}/*.rgen
  ${SHADER_SOURCE_DIR}/*.rchit
  ${SHADER_SOURCE_DIR}/*.rmiss)

add_custom_command(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
  OUTPUT ${SHADER_BINARY_DIR}
  COMMENT "Creating ${SHADER_BINARY_DIR}"
)

if (Vulkan_glslc_FOUND)
  foreach(source IN LISTS SHADERS)
    get_filename_component(FILENAME ${source} NAME)
    add_custom_command(
      COMMAND
        ${glslc_executable}
        -o ${SHADER_BINARY_DIR}/${FILENAME}.spv
        ${source}
      OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
      DEPENDS ${source} ${SHADER_BINARY_DIR}
      COMMENT "Compiling shader(glslc): ${FILENAME}"
    )
    list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
  endforeach()
elseif (Vulkan_glslangValidator_FOUND)
  foreach(source IN LISTS SHADERS)
    get_filename_component(FILENAME ${source} NAME)
    add_custom_command(
      COMMAND
        ${glslangValidator_executable}
        -V
        -o ${SHADER_BINARY_DIR}/${FILENAME}.spv
        ${source}
      OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
      DEPENDS ${source} ${SHADER_BINARY_DIR}
      COMMENT "Compiling shader(glsanvValidator): ${FILENAME}"
    )
    list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
  endforeach()
else()
  message(FATAL_ERROR "No shader compiler found. Please install glslangValidator or glslc.")
endif()

add_custom_target(compile-shaders ALL DEPENDS ${SPV_SHADERS})


#######################################################################
#                           MAIN EXECUTABLE                           #
#######################################################################

# Add the executable
add_executable(main src/exec/main.cpp src/vulkan/vma.cpp)
add_dependencies(main compile-shaders)

# Add the binary tree to the search path for include files
target_include_directories(main PUBLIC
	"${PROJECT_BINARY_DIR}"
	"${PROJECT_SOURCE_DIR}/src/vulkan"
	${Vulkan_INCLUDE_DIRS}
	)

target_link_libraries(main PUBLIC
	${Vulkan_LIBRARIES}
  GPUOpen::VulkanMemoryAllocator)

#*****************************************************************************
# Copyright 2020 NVIDIA Corporation. All rights reserved.
#*****************************************************************************

cmake_minimum_required(VERSION 3.9.6 FATAL_ERROR)

#--------------------------------------------------------------------------------------------------
# Project setting
get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(PROJNAME ${PROJNAME})
project(${PROJNAME} LANGUAGES C CXX)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJNAME}:")

#--------------------------------------------------------------------------------------------------
# C++ target and defines
set(CMAKE_CXX_STANDARD 17)
add_executable(${PROJNAME})



#--------------------------------------------------------------------------------------------------
# look for shared_sources 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES shared_sources/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing shared_sources"
    )
endif()

## Various functions and macros REQUIRED
if(EXISTS ${BASE_DIRECTORY}/shared_sources/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/shared_sources/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/shared_sources/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing shared_sources")
endif() 



#--------------------------------------------------------------------------------------------------
# Resources
#
 download_files(FILENAMES robot_toon.zip EXTRACT)
 download_files(FILENAMES daytime.hdr std_env.hdr)



#--------------------------------------------------------------------------------------------------
# Packages
_add_package_VulkanSDK()
_add_package_ImGUI()
_add_package_FreeImage()
_add_package_NVML()
# This should be added after all packages
_add_shared_sources_lib()



#--------------------------------------------------------------------------------------------------
# Memory Allocation
#   Forcing to use our memory allocator DMA (similar to Vulkan Memory Allocator (VMA))
target_compile_definitions(${PROJNAME} PUBLIC NVVK_ALLOC_DMA)
# target_compile_definitions(${PROJNAME} PUBLIC NVVK_ALLOC_DEDICATED)

#--------------------------------------------------------------------------------------------------
# Default definitions: PROJECT_RELDIRECTORY, ... 
_add_project_definitions(${PROJNAME})



#--------------------------------------------------------------------------------------------------
# For NVML, the DLL is loaded by the application
if(MSVC)
  set_target_properties(${PROJNAME} PROPERTIES LINK_FLAGS "/DELAYLOAD:nvml.dll")
endif()

#--------------------------------------------------------------------------------------------------
# Source files for this project
file(GLOB SOURCE_FILES *.cpp *.c)
file(GLOB HEADER_FILES *.hpp *.h )


#--------------------------------------------------------------------------------------------------
# GLSL to SPIR-V custom build
compile_glsl_directory(
    SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders" 
    DST "${CMAKE_CURRENT_SOURCE_DIR}/autogen" 
    VULKAN_TARGET "vulkan1.2"
    HEADER ON
    DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
    )



#--------------------------------------------------------------------------------------------------
# Sources
target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES} ${HEADER_FILES})
target_sources(${PROJNAME} PUBLIC ${COMMON_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${PACKAGE_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${GLSL_SOURCES})
target_sources(${PROJNAME} PUBLIC ${GLSL_HEADERS})


#--------------------------------------------------------------------------------------------------
# Sub-folders in Visual Studio
#
source_group("Common"         FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group("Sources"        FILES ${SOURCE_FILES})
source_group("Headers"        FILES ${HEADER_FILES})
source_group("Shader Sources" FILES ${GLSL_SOURCES})
source_group("Shader Headers" FILES ${GLSL_HEADERS})



#####################################################################################
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} shared_sources)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target( ${PROJNAME} )


#####################################################################################
# Copy the default scene and images
#
#install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/media" CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
#install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/media" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}")





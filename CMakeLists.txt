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
# look for shared_sources 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find CMakeLists_include.txt yet
if(NOT BASE_DIRECTORY)
   SET(BASE_DIRECTORY "" CACHE FILEPATH "folder containing shared_sources")
  SET(ADD_SUBDIR_BELOW 1)
  find_path(BASE_DIRECTORY2
    NAMES shared_sources
    PATHS  ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}
    REQUIRED
    DOC "Couldn't find shared_source directory'"
    )
  SET(BASE_DIRECTORY ${BASE_DIRECTORY2} )
endif()

## Various functions and macros REQUIRED
include(${BASE_DIRECTORY}/shared_sources/CMakeLists_include.txt)

#--------------------------------------------------------------------------------------------------
# Package shared by all projects
_add_package_VulkanSDK()
_add_package_ImGUI()
_add_package_ZLIB()
_add_shared_sources_lib()




#--------------------------------------------------------------------------------------------------
# C++ target and defines
set(CMAKE_CXX_STANDARD 17)
add_executable(${PROJNAME})

target_compile_definitions(${PROJNAME} PUBLIC NVVK_ALLOC_DMA)
_add_project_definitions(${PROJNAME})


#--------------------------------------------------------------------------------------------------
# Packages 
_add_package_FreeImage()

# Adding NVML 
_add_package_NVML()
if(MSVC)
  set_target_properties(${PROJNAME} PROPERTIES LINK_FLAGS "/DELAYLOAD:nvml.dll")
endif()


#--------------------------------------------------------------------------------------------------
# Source files for this project
file(GLOB SOURCE_FILES *.cpp *.c)
file(GLOB HEADER_FILES *.hpp *.h )
file(GLOB EXTRA_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.*")
list(APPEND COMMON_SOURCE_FILES ${EXTRA_COMMON})
target_include_directories(${PROJNAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)



#--------------------------------------------------------------------------------------------------
# GLSL to SPIR-V custom build
SET(VULKAN_TARGET_ENV vulkan1.2)

file(GLOB_RECURSE GLSL_HEADER_FILES "shaders/*.h" "shaders/*.glsl")

# Macro to add custom build for Spir-V
# _SOURCE can be more than one file (.vert + .frag)
# _OUTPUT is the .spv file, resulting from the linkage
#
macro(_compile_GLSL_to_header _SOURCE _OUTPUT SOURCE_LIST OUTPUT_LIST FILE_NAME)
  LIST(APPEND ${SOURCE_LIST} ${_SOURCE})
  LIST(APPEND ${OUTPUT_LIST} autogen/${_OUTPUT})
  STRING(REPLACE "." "_" VAR_NAME ${FILE_NAME})
  if(GLSLANGVALIDATOR)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/autogen/${_OUTPUT}
      COMMAND echo ${GLSLANGVALIDATOR} -o autogen/${_OUTPUT} -V ${_SOURCE} --vn ${VAR_NAME} --target-env ${VULKAN_TARGET_ENV}
    COMMAND ${GLSLANGVALIDATOR} -g -o autogen/${_OUTPUT} -V ${_SOURCE} --vn ${VAR_NAME} --target-env ${VULKAN_TARGET_ENV}
      MAIN_DEPENDENCY ${_SOURCE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS ${GLSL_HEADER_FILES}
      )
  else(GLSLANGVALIDATOR)
    MESSAGE(WARNING "could not find GLSLANGVALIDATOR to compile shaders")
  endif(GLSLANGVALIDATOR)
endmacro()


UNSET(GLSL_SOURCES)
UNSET(SPV_OUTPUT)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "shaders/*.comp"
    "shaders/*.frag"
    "shaders/*.vert"
    "shaders/*.rchit"
    "shaders/*.rahit"
    "shaders/*.rmiss"
    "shaders/*.rgen"
    )
foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    #_compile_GLSL(${GLSL} "shaders/${FILE_NAME}.spv" GLSL_SOURCES SPV_OUTPUT)
     _compile_GLSL_to_header(${GLSL} "shaders/${FILE_NAME}.h" GLSL_SOURCES SPV_OUTPUT ${FILE_NAME})
endforeach(GLSL)


#list(APPEND GLSL_SOURCES ${GLSL_HEADER_FILES})


# ADD_CUSTOM_TARGET(GLSL_files ALL DEPENDS ${GLSL_HEADER_FILES})
# ADD_DEPENDENCIES(${PROJNAME} GLSL_files)



#--------------------------------------------------------------------------------------------------
# Sources
target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES} ${HEADER_FILES})
target_sources(${PROJNAME} PUBLIC ${COMMON_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${PACKAGE_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${GLSL_SOURCES})
target_sources(${PROJNAME} PUBLIC ${GLSL_HEADER_FILES})
#target_sources(${PROJNAME} PUBLIC ${CUDA_FILES} ${CUBIN_SOURCES})


#--------------------------------------------------------------------------------------------------
# Sub-folders in Visual Studio
#
source_group("Common"       FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group("Sources"      FILES ${SOURCE_FILES})
source_group("Headers"      FILES ${HEADER_FILES})
source_group("Shader Sources" FILES ${GLSL_SOURCES})
source_group("Shader Headers" FILES ${GLSL_HEADER_FILES})



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
_copy_binaries_to_target( ${PROJNAME} )


install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/shaders")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}/shaders")
install(FILES ${CUBIN_SOURCES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
install(FILES ${CUBIN_SOURCES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}")
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/media" CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/media" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}")



#----------------------------------------------------------------------------------------------------
# Copying elements
file(GLOB MDL_DLLs "${MDL_DLL_FOLDER}/*.dll")
# Next to the executable
# target_copy_to_output_dir(TARGET ${PROJECT_NAME} FILES ${MDL_SHADERS_FOLDER})
# target_copy_to_output_dir(TARGET ${PROJECT_NAME} FILES ${MDL_DLLs})
# to sub folder
target_copy_to_output_dir(TARGET ${PROJECT_NAME} DEST_SUBFOLDER "${PROJECT_NAME}/" FILES "${CMAKE_CURRENT_SOURCE_DIR}/media")

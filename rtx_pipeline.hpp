/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include "vulkan/vulkan.hpp"

#include "nvmath/nvmath_glsltypes.h"
#include "nvvk/allocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

using namespace nvmath;
#include "nvvk/profiler_vk.hpp"
#include "renderer.h"
#include "structures.h"

/*

Creating the RtCore renderer 
* Requiring:  
  - Acceleration structure (AccelSctruct / Tlas)
  - An image (Post StoreImage)
  - The glTF scene (vertex, index, materials, ... )

* Usage
  - setup as usual
  - create
  - run
*/
class RtxPipeline : public Renderer
{
public:
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::Allocator* allocator) override;
  void destroy() override;
  void create(const vk::Extent2D& size, const std::vector<vk::DescriptorSetLayout>& rtDescSetLayouts, Scene* scene) override;
  void              updatePipeline(const std::vector<vk::DescriptorSetLayout>& rtDescSetLayouts);
  void              run(const vk::CommandBuffer&              cmdBuf,
                        const vk::Extent2D&                   size,
                        nvvk::ProfilerVK&                     profiler,
                        const std::vector<vk::DescriptorSet>& descSets) override;
  const std::string name() override { return std::string("Rtx"); }

private:
  void createRtShaderBindingTable();


  uint32_t m_nbHit;

private:
  // Setup
  nvvk::Allocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil  m_debug;   // Utility to name objects
  vk::Device       m_device;
  uint32_t         m_queueIndex;


  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR   m_rtProperties;
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  vk::PipelineLayout                                  m_rtPipelineLayout;
  vk::Pipeline                                        m_rtPipeline;
  nvvk::Buffer                                        m_rtSBTBuffer;
};

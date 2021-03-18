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
#include "nvvk/allocator_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "vulkan/vulkan.hpp"

using namespace nvmath;
#include "structures.h"

// Forward declaration
class Scene;

class Renderer
{
public:
  virtual void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::Allocator* allocator) = 0;
  virtual void destroy()                                                = 0;
  virtual void run(const vk::CommandBuffer&              cmdBuf,
                   const vk::Extent2D&                   size,
                   nvvk::ProfilerVK&                     profiler,
                   const std::vector<vk::DescriptorSet>& extraDescSets) = 0;
  virtual void create(const vk::Extent2D&                         size,
                      const std::vector<vk::DescriptorSetLayout>& extraDescSetsLayout,
                      Scene*                                      _scene = nullptr)                          = 0;

  virtual const std::string name() = 0;
  RtxState                  m_state;
};

/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

//////////////////////////////////////////////////////////////////////////
// Raytracing implementation
//
// There are 3 descriptor sets
// (0) - Acceleration structure and result image
// (1) - Various buffers: vertices, indices, matrices, ...
// (2) - Material and Textures
//
//////////////////////////////////////////////////////////////////////////

#include <vulkan/vulkan.hpp>

#include "vkalloc.hpp"

#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/raytraceNV_vk.hpp"


class Raytracer
{

public:
  Raytracer();

  // Initializing the allocator and querying the raytracing properties
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator);


  // Return the rendered image
  const nvvk::Texture& outputImage() const;
  const int            maxFrames() const;

  void destroy();

  // Creating the image where the result is stored
  void createOutputImage(vk::Extent2D size);

  // Create a descriptor set holding the acceleration structure and the output image
  void createDescriptorSet();

  // Will be called when resizing the window
  void updateDescriptorSet();

  // Pipeline with all shaders, including the 3 descriptor layouts.
  void createPipeline(const vk::DescriptorSetLayout& sceneDescSetLayout, const vk::DescriptorSetLayout& matDescSetLayout);

  // The SBT, storing in a buffer the calling handles of each shader group
  void createShadingBindingTable();

  // Executing the raytracing
  void run(const vk::CommandBuffer& cmdBuf, const vk::DescriptorSet& sceneDescSet, const vk::DescriptorSet& matDescSet, int frame = 0);

  // To control the raytracer
  bool uiSetup();

  nvvk::RaytracingBuilderNV& builder() { return m_rtBuilder; }

private:
  std::vector<vk::RayTracingShaderGroupCreateInfoNV> m_groups;
  nvvk::Texture                                      m_raytracingOutput;
  vk::Extent2D                                       m_outputSize;
  nvvk::DescriptorSetBindings                        m_binding;

  vk::Device       m_device;
  nvvk::DebugUtil  m_debug;
  uint32_t         m_queueIndex;
  nvvk::Allocator* m_alloc{nullptr};

  nvvk::Buffer                                       m_rtSBTBuffer;
  vk::PhysicalDeviceRayTracingPropertiesNV           m_rtProperties;
  nvvk::RaytracingBuilderNV                          m_rtBuilder;
  nvvk::DescriptorSetBindings                        m_rtDescSetLayoutBind;
  vk::DescriptorPool                                 m_rtDescPool;
  vk::DescriptorSetLayout                            m_rtDescSetLayout;
  vk::DescriptorSet                                  m_rtDescSet;
  std::vector<vk::RayTracingShaderGroupCreateInfoNV> m_rtShaderGroups;
  vk::PipelineLayout                                 m_rtPipelineLayout;
  vk::Pipeline                                       m_rtPipeline;

  struct PushConstant
  {
    int   frame{0};    // Current frame number
    int   depth{2};    // Max depth
    int   samples{5};  // samples per frame
    float hdrMultiplier{1.f};
  } m_pushC;

  int m_maxFrames{100};  // Max iterations
};

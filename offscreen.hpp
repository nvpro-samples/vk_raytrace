/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once
#include "vulkan/vulkan.hpp"

#include "nvmath/nvmath_glsltypes.h"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"


class Offscreen
{
public:
  struct Tonemapper
  {
    float        brightness{1.0f};
    float        contrast{1.0f};
    float        saturation{1.0f};
    float        vignette{0.0f};
    float        avgLum{1.0f};
    float        zoom{1.0f};
    nvmath::vec2 renderingRatio{1.0f, 1.0f};  // Rendering area without the UI
  } m_tonemapper;

public:
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator);
  void destroy();
  void create(const vk::Extent2D& size, const vk::RenderPass& renderPass);
  void update(const vk::Extent2D& size);
  void run(vk::CommandBuffer cmdBuf);

  vk::DescriptorSetLayout getDescLayout() { return m_postDescSetLayout; }
  vk::DescriptorSet       getDescSet() { return m_postDescSet; }
  vk::RenderPass          getRenderPass() { return m_offscreenRenderPass; }
  vk::Framebuffer         getFrameBuffer() { return m_offscreenFramebuffer; }


private:
  void createOffscreenRender(const vk::Extent2D& size);
  void createPostPipeline(const vk::RenderPass& renderPass);
  void createPostDescriptor();
  void updatePostDescriptorSet();

  vk::DescriptorPool      m_postDescPool;
  vk::DescriptorSetLayout m_postDescSetLayout;
  vk::DescriptorSet       m_postDescSet;
  vk::Pipeline            m_postPipeline;
  vk::PipelineLayout      m_postPipelineLayout;
  vk::RenderPass          m_offscreenRenderPass;
  vk::Framebuffer         m_offscreenFramebuffer;
  nvvk::Texture           m_offscreenColor;
  //vk::Format m_offscreenColorFormat{vk::Format::eR16G16B16A16Sfloat};  // Darkening the scene over 5000 iterations
  vk::Format    m_offscreenColorFormat{vk::Format::eR32G32B32A32Sfloat};
  nvvk::Texture m_offscreenDepth;
  vk::Format    m_offscreenDepthFormat{vk::Format::eX8D24UnormPack32}; // Will be replaced by best supported format


  // Setup
  nvvk::ResourceAllocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil  m_debug;   // Utility to name objects
  vk::Device       m_device;
  uint32_t         m_queueIndex;
};

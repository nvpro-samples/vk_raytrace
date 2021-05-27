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
  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator);
  void destroy();
  void create(const VkExtent2D& size, const VkRenderPass& renderPass);
  void update(const VkExtent2D& size);
  void run(VkCommandBuffer cmdBuf);

  VkDescriptorSetLayout getDescLayout() { return m_postDescSetLayout; }
  VkDescriptorSet       getDescSet() { return m_postDescSet; }
  VkRenderPass          getRenderPass() { return m_offscreenRenderPass; }
  VkFramebuffer         getFrameBuffer() { return m_offscreenFramebuffer; }


private:
  void createOffscreenRender(const VkExtent2D& size);
  void createPostPipeline(const VkRenderPass& renderPass);
  void createPostDescriptor();
  void updatePostDescriptorSet();

  VkDescriptorPool      m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet       m_postDescSet{VK_NULL_HANDLE};
  VkPipeline            m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout      m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass          m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer         m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture         m_offscreenColor;
  nvvk::Texture         m_offscreenDepth;
  //VkFormat m_offscreenColorFormat{VkFormat::eR16G16B16A16Sfloat};  // Darkening the scene over 5000 iterations
  VkFormat m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};  // Will be replaced by best supported format


  // Setup
  nvvk::ResourceAllocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;   // Utility to name objects
  VkDevice                 m_device;
  uint32_t                 m_queueIndex;
};

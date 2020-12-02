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

#include "nvvk/allocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"


class Offscreen
{
public:
  struct TReinhard
  {
    float key{0.5f};
    float Ywhite{5.0f};
    float sat{1.f};
    float invGamma{1.0f / 2.2f};
    float avgLum{1.0f};
  } m_tReinhard;

public:
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::Allocator* allocator);
  void destroy();
  void create(const vk::Extent2D& size, const vk::RenderPass& renderPass);
  void update(const vk::Extent2D& size);
  void run(vk::CommandBuffer cmdBuf, const vk::Extent2D& size);

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
  vk::Format              m_offscreenColorFormat{vk::Format::eR32G32B32A32Sfloat};
  nvvk::Texture           m_offscreenDepth;
  vk::Format              m_offscreenDepthFormat{vk::Format::eD32Sfloat};


  // Setup
  nvvk::Allocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil  m_debug;   // Utility to name objects
  vk::Device       m_device;
  uint32_t         m_queueIndex;
};

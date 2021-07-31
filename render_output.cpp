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


/*
 *  This creates the image in floating point, holding the result of ray tracing.
 *  It also creates a pipeline for drawing this image from HDR to LDR applying a tonemapper
 */


#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "render_output.hpp"
#include "tools.hpp"

// Shaders
#include "autogen/passthrough.vert.h"
#include "autogen/post.frag.h"


void RenderOutput::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);

  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}


void RenderOutput::destroy()
{
  m_pAlloc->destroy(m_offscreenColor);

  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
}

void RenderOutput::create(const VkExtent2D& size, const VkRenderPass& renderPass)
{
  MilliTimer timer;
  LOGI("Create Offscreen");
  createOffscreenRender(size);
  createPostPipeline(renderPass);
  timer.print();
}

void RenderOutput::update(const VkExtent2D& size)
{
  createOffscreenRender(size);
}

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void RenderOutput::createOffscreenRender(const VkExtent2D& size)
{
  m_size = size;
  if(m_offscreenColor.image != VK_NULL_HANDLE)
  {
    m_pAlloc->destroy(m_offscreenColor);
  }

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(
        size, m_offscreenColorFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

    nvvk::Image image = m_pAlloc->createImage(colorCreateInfo);
    NAME_VK(image.image);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.maxLod                          = FLT_MAX;
    m_offscreenColor                        = m_pAlloc->createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }


  createPostDescriptor();
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void RenderOutput::createPostPipeline(const VkRenderPass& renderPass)
{
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);

  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Tonemapper)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.setLayoutCount         = 1;
  pipelineLayoutCreateInfo.pSetLayouts            = &m_postDescSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_postPipelineLayout);

  // Pipeline: completely generic, no vertices
  std::vector<uint32_t> vertexShader(std::begin(passthrough_vert), std::end(passthrough_vert));
  std::vector<uint32_t> fragShader(std::begin(post_frag), std::end(post_frag));

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, renderPass);
  pipelineGenerator.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  CREATE_NAMED_VK(m_postPipeline, pipelineGenerator.createPipeline());
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void RenderOutput::createPostDescriptor()
{
  nvvk::DescriptorSetBindings bind;

  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);

  // This descriptor is passed to the RTX pipeline
  // Ray tracing will write to the binding 1, but the fragment shader will be using binding 0, so it can use a sampler too.
  bind.addBinding({0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT});
  bind.addBinding({1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR});
  m_postDescSetLayout = bind.createLayout(m_device);
  m_postDescPool      = bind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor));  // This is use by the tonemapper
  writes.emplace_back(bind.makeWrite(m_postDescSet, 1, &m_offscreenColor.descriptor));  // This will be used by the ray trace to write the image
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void RenderOutput::run(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Tonemapper), &m_tonemapper);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);
}

//--------------------------------------------------------------------------------------------------
// Generating all pyramid images, the highest level is used for getting the average luminance
// of the image, which is then use to auto-expose.
//
void RenderOutput::genMipmap(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  nvvk::cmdGenerateMipmaps(cmdBuf, m_offscreenColor.image, m_offscreenColorFormat, m_size, nvvk::mipLevels(m_size), 1,
                           VK_IMAGE_LAYOUT_GENERAL);
}

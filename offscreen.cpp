/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "offscreen.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "tools.hpp"


#include "autogen/shaders/passthrough.vert.h"
#include "autogen/shaders/post.frag.h"


void Offscreen::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::Allocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);
}


void Offscreen::destroy()
{
  m_pAlloc->destroy(m_offscreenColor);
  m_pAlloc->destroy(m_offscreenDepth);

  m_device.destroy(m_postPipeline);
  m_device.destroy(m_postPipelineLayout);
  m_device.destroy(m_postDescPool);
  m_device.destroy(m_postDescSetLayout);
  m_device.destroy(m_offscreenRenderPass);
  m_device.destroy(m_offscreenFramebuffer);
}

void Offscreen::create(const vk::Extent2D& size, const vk::RenderPass& renderPass)
{
  MilliTimer timer;
  LOGI("Create Offscreen");
  createOffscreenRender(size);
  createPostPipeline(renderPass);
  timer.print();
}

void Offscreen::update(const vk::Extent2D& size)
{
  createOffscreenRender(size);
}

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void Offscreen::createOffscreenRender(const vk::Extent2D& size)
{
  if(m_offscreenColor.image)
  {
    m_pAlloc->destroy(m_offscreenColor);
    m_pAlloc->destroy(m_offscreenDepth);
  }

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(size, m_offscreenColorFormat,
                                                       vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
                                                           | vk::ImageUsageFlagBits::eStorage);


    nvvk::Image             image           = m_pAlloc->createImage(colorCreateInfo);
    vk::ImageViewCreateInfo ivInfo          = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_offscreenColor                        = m_pAlloc->createTexture(image, ivInfo, vk::SamplerCreateInfo());
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(size, m_offscreenDepthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment);
  {
    nvvk::Image image = m_pAlloc->createImage(depthCreateInfo);

    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.setViewType(vk::ImageViewType::e2D);
    depthStencilView.setFormat(m_offscreenDepthFormat);
    depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
    depthStencilView.setImage(image.image);

    m_offscreenDepth = m_pAlloc->createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageAspectFlagBits::eDepth);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass = nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                                   true, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
  }

  // Creating the frame buffer for offscreen
  std::vector<vk::ImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  m_device.destroy(m_offscreenFramebuffer);
  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_offscreenRenderPass);
  info.setAttachmentCount(2);
  info.setPAttachments(attachments.data());
  info.setWidth(size.width);
  info.setHeight(size.height);
  info.setLayers(1);
  m_offscreenFramebuffer = m_device.createFramebuffer(info);


  createPostDescriptor();
  updatePostDescriptorSet();
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void Offscreen::createPostPipeline(const vk::RenderPass& renderPass)
{
  m_device.destroy(m_postPipeline);
  m_device.destroy(m_postPipelineLayout);

  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(TReinhard)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_postDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_postPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  std::vector<uint32_t> vertexShader(std::begin(passthrough_vert), std::end(passthrough_vert));
  std::vector<uint32_t> fragShader(std::begin(post_frag), std::end(post_frag));

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, renderPass);
  pipelineGenerator.addShader(vertexShader, vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(fragShader, vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_postPipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void Offscreen::createPostDescriptor()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;

  m_device.destroy(m_postDescSetLayout);
  m_device.destroy(m_postDescPool);
  m_postDescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment | vkSS::eCompute | vkSS::eRaygenKHR));
  m_postDescSetLayoutBind.addBinding(vkDS(1, vkDT::eStorageImage, 1, vkSS::eFragment | vkSS::eCompute | vkSS::eRaygenKHR));
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);

  vk::WriteDescriptorSet w1 = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  vk::WriteDescriptorSet w2 = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 1, &m_offscreenColor.descriptor);
  m_device.updateDescriptorSets({w1, w2}, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void Offscreen::updatePostDescriptorSet() {}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void Offscreen::run(vk::CommandBuffer cmdBuf, const vk::Extent2D& size)
{
  m_debug.beginLabel(cmdBuf, "Post");

  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)size.width, (float)size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {size.width, size.height}}});

  cmdBuf.pushConstants<TReinhard>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_tReinhard);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_postPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_postPipelineLayout, 0, m_postDescSet, {});
  cmdBuf.draw(3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

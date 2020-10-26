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
#include <vulkan/vulkan.hpp>

#include "vkalloc.hpp"

#include "nvvk/pipeline_vk.hpp"

// Take as an input an image (RGB32F) and apply a tonemapper
//
// Usage:
// - setup(...)
// - initialize( size of window/image )
// - setInput( image.descriptor )
// - run
// - getOutput
class Tonemapper
{
public:
  Tonemapper() = default;
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
  {
    m_device     = device;
    m_queueIndex = queueIndex;

    m_alloc = allocator;
  }

  void initialize(const VkExtent2D& size)
  {
    createRenderPass();
    createDescriptorSet();
    createPipeline();
    updateRenderTarget(size);
  }

  // Attaching the input image
  void setInput(const vk::DescriptorImageInfo& descriptor)
  {
    m_device.waitIdle();
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets{{m_descriptorSet, 0, 0, 1, vkDT::eCombinedImageSampler, &descriptor}};
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }

  // Updating the output framebuffer when the image size is changing
  void updateRenderTarget(const VkExtent2D& size)
  {
    m_size = size;
    m_alloc->destroy(m_output);


    vk::SamplerCreateInfo samplerCreateInfo;  // default values
    vk::ImageCreateInfo   imageCreateInfo =
        nvvk::makeImage2DCreateInfo(m_size, vk::Format::eR8G8B8A8Unorm,
                                    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    vk::DeviceSize bufferSize = m_size.width * m_size.height * 4;

    {
      nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
      nvvk::Image              image  = m_alloc->createImage(cmdBuf, bufferSize, nullptr, imageCreateInfo);
      vk::ImageViewCreateInfo  ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
      m_output                        = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    }
    m_alloc->finalizeAndReleaseStaging();


    if(m_framebuffer)
    {
      m_device.destroyFramebuffer(m_framebuffer);
    }

    vk::FramebufferCreateInfo info;
    info.setRenderPass(m_renderPass);

    info.setAttachmentCount(1);
    vk::ImageView view = m_output.descriptor.imageView;
    info.setPAttachments(&view);
    info.setWidth(m_size.width);
    info.setHeight(m_size.height);
    info.setLayers(1);

    m_framebuffer = m_device.createFramebuffer(info);
  }

  void destroy()
  {
    m_alloc->destroy(m_output);
    m_device.destroyFramebuffer(m_framebuffer);
    m_device.destroyDescriptorSetLayout(m_descriptorSetLayout);
    m_device.destroyRenderPass(m_renderPass);
    m_device.destroyPipeline(m_pipeline);
    m_device.destroyPipelineLayout(m_pipelineLayout);
    m_device.destroyDescriptorPool(m_descriptorPool);

    m_renderPass  = vk::RenderPass();
    m_framebuffer = vk::Framebuffer();
  }

  nvvk::Texture getOutput() { return m_output; }

  // Executing the the tonemapper
  void run(const vk::CommandBuffer& cmdBuf)
  {
    vk::RenderPassBeginInfo renderPassBeginInfo = {m_renderPass, m_framebuffer, {{}, m_size}, 0, {}};
    cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

    //
    cmdBuf.setViewport(0, {vk::Viewport(0, 0, m_size.width, m_size.height, 0, 1)});
    cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

    cmdBuf.pushConstants<pushConstant>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_pushCnt);
    cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, m_descriptorSet, {});
    cmdBuf.draw(3, 1, 0, 0);
    cmdBuf.endRenderPass();
  }

  // Controlling the tonemapper
  bool uiSetup()
  {
    bool modified = false;
    if(ImGui::CollapsingHeader("Tonemapper"))
    {
      using GuiH = ImGuiH::Control;
      static pushConstant d;

      static const char* tmItem[] = {"Linear", "Uncharted 2", "Hejl Richard", "ACES"};
      modified |= GuiH::Selection("Tonemapper", "", &m_pushCnt.tonemapper, &d.tonemapper, ImGuiH::Control::Flags::Normal,
                                  {"Linear", "Uncharted 2", "Hejl Richard", "ACES"});
      modified |= GuiH::Slider("Exposure", "", &m_pushCnt.exposure, &d.exposure, ImGuiH::Control::Flags::Normal, 0.1f, 100.f);
      modified |= GuiH::Slider("Gamma", "", &m_pushCnt.gamma, &d.gamma, ImGuiH::Control::Flags::Normal, .1f, 3.f);
    }
    return modified;
  }

private:
  // Render pass, no clear, no depth
  void createRenderPass()
  {
    if(m_renderPass)
      m_device.destroyRenderPass(m_renderPass);

    // Color attachment
    vk::AttachmentDescription attachments;
    attachments.setFormat(vk::Format::eR8G8B8A8Unorm);  // image format of the output image
    attachments.setLoadOp(vk::AttachmentLoadOp::eDontCare);
    attachments.setFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::AttachmentReference colorReference{0, vk::ImageLayout::eColorAttachmentOptimal};
    vk::SubpassDescription  subpassDescription;
    subpassDescription.setColorAttachmentCount(1);
    subpassDescription.setPColorAttachments(&colorReference);

    vk::RenderPassCreateInfo renderPassInfo{{}, 1, &attachments, 1, &subpassDescription};
    m_renderPass = m_device.createRenderPass(renderPassInfo);
  }

  // One input image and push constant to control the effect
  void createDescriptorSet()
  {
    vk::DescriptorPoolSize         poolSizes         = {vkDT::eCombinedImageSampler, 1};
    vk::DescriptorSetLayoutBinding setLayoutBindings = {0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment};
    vk::PushConstantRange          push_constants    = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(pushConstant)};

    m_descriptorPool      = m_device.createDescriptorPool({{}, 1, 1, &poolSizes});
    m_descriptorSetLayout = m_device.createDescriptorSetLayout({{}, 1, &setLayoutBindings});
    m_descriptorSet       = m_device.allocateDescriptorSets({m_descriptorPool, 1, &m_descriptorSetLayout})[0];

    m_pipelineLayout = m_device.createPipelineLayout({{}, 1, &m_descriptorSetLayout, 1, &push_constants});
  }

  // Creating the shading pipeline
  void createPipeline()
  {
    std::vector<std::string> paths = defaultSearchPaths;

    // Pipeline: completely generic, no vertices
    nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
    pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
    pipelineGenerator.addShader(nvh::loadFile("shaders/tonemap.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
    pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
    m_pipeline = pipelineGenerator.createPipeline();
  }

  struct pushConstant
  {
    int   tonemapper{1};
    float gamma{2.2f};
    float exposure{1.0f};
  };


  vk::RenderPass          m_renderPass;
  vk::Pipeline            m_pipeline;
  vk::PipelineLayout      m_pipelineLayout;
  vk::DescriptorPool      m_descriptorPool;
  vk::DescriptorSetLayout m_descriptorSetLayout;
  vk::DescriptorSet       m_descriptorSet;
  vk::Device              m_device;
  vk::Framebuffer         m_framebuffer;
  vk::Extent2D            m_size{0, 0};

  nvvk::Texture    m_output;
  uint32_t         m_queueIndex;
  nvvk::Allocator* m_alloc;

  pushConstant m_pushCnt{1, 2.2, 1.0};
};

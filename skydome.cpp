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

#include <vulkan/vulkan.hpp>

#define _USE_MATH_DEFINES
#include <chrono>
#include <iostream>
#include <math.h>

#include "fileformats/stb_image.h"
#include "nvh/fileoperations.hpp"
#include "nvh/nvprint.hpp"
#include "nvmath/nvmath.h"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "skydome.hpp"


using vkDT = vk::DescriptorType;
using vkSS = vk::ShaderStageFlagBits;
extern std::vector<std::string> defaultSearchPaths;

//--------------------------------------------------------------------------------------------------
// Initialize the Skydome with a HDR image
//
void SkydomePbr::create(const vk::DescriptorBufferInfo& sceneBufferDesc, const vk::RenderPass& renderPass)
{
  createCube();
  m_renderPass = renderPass;
  createPipelines(sceneBufferDesc);
}

//--------------------------------------------------------------------------------------------------
// Draw the environment
//
void SkydomePbr::draw(const vk::CommandBuffer& commandBuffer)
{
  // Pipeline to use for rendering the current scene
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

  // The pipeline uses three descriptor set, one for the scene information, one for the matrix of the instance, one for the textures
  std::vector<vk::DescriptorSet> descriptorSets = {m_descriptorSet[eScene], m_descriptorSet[eMaterial]};
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, descriptorSets, {});


  std::vector<vk::Buffer>     buffers{m_vertices.buffer};
  std::vector<vk::DeviceSize> offset{0};

  commandBuffer.bindVertexBuffers(0, 1, buffers.data(), offset.data());
  commandBuffer.bindIndexBuffer(m_indices.buffer, 0, vk::IndexType::eUint32);
  commandBuffer.drawIndexed(36, 1, 0, 0, 0);
}

//--------------------------------------------------------------------------------------------------
// Creating the environment cube
//
void SkydomePbr::createCube()
{

  if(m_vertices.buffer)
    return;  // Was already initialized

  std::vector<nvmath::vec3f> vertexBuffer = {
      {-0.500000000, -0.500000000, 0.500000000},  {0.500000000, -0.500000000, 0.500000000},
      {-0.500000000, 0.500000000, 0.500000000},   {0.500000000, 0.500000000, 0.500000000},
      {-0.500000000, -0.500000000, -0.500000000}, {0.500000000, -0.500000000, -0.500000000},
      {-0.500000000, 0.500000000, -0.500000000},  {0.500000000, 0.500000000, -0.500000000},
  };

  std::vector<uint32_t> indexBuffer = {
      0, 3, 2, 0, 1, 3,  //    6-----7     Y
      1, 7, 3, 1, 5, 7,  //   /|    /|     ^
      5, 4, 7, 6, 7, 4,  //  2-----3 |     |
      0, 6, 4, 0, 2, 6,  //  | 4 --|-5     ---> X
      2, 7, 6, 2, 3, 7,  //  |/    |/     /
      4, 1, 0, 4, 5, 1,  //  0-----1     Z
  };

  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    m_vertices = m_alloc->createBuffer<nvmath::vec3f>(cmdBuf, vertexBuffer, vk::BufferUsageFlagBits::eVertexBuffer);
    m_indices  = m_alloc->createBuffer<uint32_t>(cmdBuf, indexBuffer, vk::BufferUsageFlagBits::eIndexBuffer);

    m_debug.setObjectName(m_vertices.buffer, "SkyVertex");
    m_debug.setObjectName(m_indices.buffer, "SkyIndex");
  }
  m_alloc->finalizeAndReleaseStaging();
}

void SkydomePbr::destroy()
{
  m_alloc->destroy(m_vertices);
  m_alloc->destroy(m_indices);
  m_alloc->destroy(m_textures.txtHdr);
  m_alloc->destroy(m_textures.accelImpSmpl);
  m_alloc->destroy(m_textures.irradianceCube);
  m_alloc->destroy(m_textures.lutBrdf);
  m_alloc->destroy(m_textures.prefilteredCube);

  m_device.destroyPipeline(m_pipeline);
  m_device.destroyPipelineLayout(m_pipelineLayout);
  m_device.destroyDescriptorSetLayout(m_descriptorSetLayout[eScene]);
  m_device.destroyDescriptorSetLayout(m_descriptorSetLayout[eMaterial]);
  m_device.destroyDescriptorPool(m_descriptorpool);
}

//--------------------------------------------------------------------------------------------------
//
//
void SkydomePbr::createPipelines(const vk::DescriptorBufferInfo& sceneBufferDesc)
{
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  {
    // Descriptors
    {
      vk::DescriptorSetLayoutBinding setLayoutBindings = {0, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eFragment};
      m_descriptorSetLayout[eScene] = m_device.createDescriptorSetLayout({{}, 1, &setLayoutBindings});
    }
    {
      vk::DescriptorSetLayoutBinding setLayoutBindings = {0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment};
      m_descriptorSetLayout[eMaterial] = m_device.createDescriptorSetLayout({{}, 1, &setLayoutBindings});
    }

    // Descriptor Pool
    std::vector<vk::DescriptorPoolSize> poolSize = {{vk::DescriptorType::eUniformBuffer, 1}, {vkDT::eCombinedImageSampler, 1}};
    m_descriptorpool = m_device.createDescriptorPool({{}, 2, uint32_t(poolSize.size()), poolSize.data()});

    // Descriptor sets
    {
      m_descriptorSet[eScene] = m_device.allocateDescriptorSets({m_descriptorpool, 1, &m_descriptorSetLayout[eScene]})[0];
      vk::WriteDescriptorSet writeDescriptorSet{
          m_descriptorSet[eScene], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &sceneBufferDesc};
      m_device.updateDescriptorSets(writeDescriptorSet, nullptr);
    }
    {
      m_descriptorSet[eMaterial] = m_device.allocateDescriptorSets({m_descriptorpool, 1, &m_descriptorSetLayout[eMaterial]})[0];
      vk::WriteDescriptorSet writeDescriptorSet;
      writeDescriptorSet.dstSet          = m_descriptorSet[eMaterial];
      writeDescriptorSet.descriptorCount = 1;
      writeDescriptorSet.descriptorType  = vk::DescriptorType::eCombinedImageSampler,
      writeDescriptorSet.pImageInfo      = (vk::DescriptorImageInfo*)&m_textures.txtHdr.descriptor;
      m_device.updateDescriptorSets(writeDescriptorSet, nullptr);
    }

    m_pipelineLayout = m_device.createPipelineLayout({{}, 2, m_descriptorSetLayout.data()});
  }

  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpg(m_device, m_pipelineLayout, m_renderPass);
  gpg.addShader(nvh::loadFile("shaders/skybox.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  gpg.addShader(nvh::loadFile("shaders/skybox.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  gpg.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  gpg.addBindingDescription({0, sizeof(nvmath::vec3f)});
  gpg.addAttributeDescriptions({{0, 0, vk::Format::eR32G32B32Sfloat, 0}});

  m_pipeline = gpg.createPipeline();

  m_debug.setObjectName(m_pipeline, "SkyPipeline");
  m_debug.setObjectName(gpg.getShaderModule(0), "SkyVS");
  m_debug.setObjectName(gpg.getShaderModule(1), "SkyFS");
}

//--------------------------------------------------------------------------------------------------
// Loading the HDR environment texture (HDR )
//
void SkydomePbr::loadEnvironment(const std::string& hrdImage)
{
  int width, height, component;

  float*         pixels     = stbi_loadf(hrdImage.c_str(), &width, &height, &component, STBI_rgb_alpha);
  vk::DeviceSize bufferSize = width * height * 4 * sizeof(float);
  vk::Extent2D   imgSize(width, height);


  vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  vk::Format            format = vk::Format::eR32G32B32A32Sfloat;
  vk::ImageCreateInfo   icInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    nvvk::Image              image  = m_alloc->createImage(cmdBuf, bufferSize, pixels, icInfo);
    vk::ImageViewCreateInfo  ivInfo = nvvk::makeImageViewCreateInfo(image.image, icInfo);
    m_textures.txtHdr               = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
  }
  m_alloc->finalizeAndReleaseStaging();

  createEnvironmentAccelTexture(pixels, imgSize, m_textures.accelImpSmpl);

  stbi_image_free(pixels);

  if(!m_vertices.buffer)
    createCube();


  integrateBrdf(512);
  prefilterDiffuse(128);
  prefilterGlossy(512);

  m_debug.setObjectName(m_textures.txtHdr.image, "SkyHdr");
  m_debug.setObjectName(m_textures.accelImpSmpl.image, "SkyImpSamp");
  m_debug.setObjectName(m_textures.lutBrdf.image, "SkyLut");
  m_debug.setObjectName(m_textures.prefilteredCube.image, "SkyGlossy");
  m_debug.setObjectName(m_textures.irradianceCube.image, "SkyIrradiance");
}

struct Env_accel
{
  uint32_t alias{0};
  float    q{0.f};
  float    pdf{0.f};
  float    _padding{0.f};
};

//--------------------------------------------------------------------------------------------------
// Build alias map for the importance sampling
//
static float build_alias_map(const std::vector<float>& data, std::vector<Env_accel>& accel)
{
  uint32_t size = static_cast<uint32_t>(data.size());

  // create qs (normalized)
  float sum = 0.0f;
  for(float d : data)
    sum += d;

  float fsize = data.size();
  for(uint32_t i = 0; i < data.size(); ++i)
    accel[i].q = fsize * data[i] / sum;

  // create partition table
  std::vector<uint32_t> partition_table(size);
  uint32_t              s = 0u, large = size;
  for(uint32_t i = 0; i < size; ++i)
    partition_table[(accel[i].q < 1.0f) ? (s++) : (--large)] = accel[i].alias = i;

  // create alias map
  for(s = 0; s < large && large < size; ++s)
  {
    const uint32_t j = partition_table[s], k = partition_table[large];
    accel[j].alias = k;
    accel[k].q += accel[j].q - 1.0f;
    large = (accel[k].q < 1.0f) ? (large + 1u) : large;
  }

  return sum;
}

//--------------------------------------------------------------------------------------------------
// Create acceleration data for importance sampling
//
void SkydomePbr::createEnvironmentAccelTexture(const float* pixels, vk::Extent2D& size, nvvk::Texture& accelTex)
{

  const uint32_t rx = size.width;
  const uint32_t ry = size.height;

  // Create importance sampling data
  std::vector<Env_accel> env_accel(rx * ry);
  std::vector<float>     importance_data(rx * ry);
  float                  cos_theta0 = 1.0f;
  const float            step_phi   = float(2.0 * M_PI) / float(rx);
  const float            step_theta = float(M_PI) / float(ry);

  for(uint32_t y = 0; y < ry; ++y)
  {
    const float theta1     = float(y + 1) * step_theta;
    const float cos_theta1 = std::cos(theta1);
    const float area       = (cos_theta0 - cos_theta1) * step_phi;
    cos_theta0             = cos_theta1;

    for(uint32_t x = 0; x < rx; ++x)
    {
      const uint32_t idx   = y * rx + x;
      const uint32_t idx4  = idx * 4;
      importance_data[idx] = area * std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
    }
  }

  m_integral                   = build_alias_map(importance_data, env_accel);
  const float inv_env_integral = 1.0f / m_integral;
  for(uint32_t i = 0; i < rx * ry; ++i)
  {
    const uint32_t idx4 = i * 4;
    env_accel[i].pdf    = std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2])) * inv_env_integral;
  }

  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);

    vk::SamplerCreateInfo samplerCreateInfo{};
    vk::Format            format     = vk::Format::eR32G32B32A32Sfloat;
    vk::ImageCreateInfo   icInfo     = nvvk::makeImage2DCreateInfo({rx, ry}, format);
    vk::DeviceSize        bufferSize = rx * ry * sizeof(Env_accel);

    nvvk::Image             image  = m_alloc->createImage(cmdBuf, bufferSize, env_accel.data(), icInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, icInfo);
    accelTex                       = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
  }
  m_alloc->finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Pre-integrate glossy BRDF, see
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
void SkydomePbr::integrateBrdf(uint32_t dim)
{
  auto tStart = std::chrono::high_resolution_clock::now();

  const vk::Format format = vk::Format::eR16G16Sfloat;
  auto&            target = m_textures.lutBrdf;


  // Image
  vk::ImageCreateInfo imageCI;
  imageCI.setImageType(vk::ImageType::e2D);
  imageCI.setFormat(format);
  imageCI.setExtent(vk::Extent3D(dim, dim, 1));
  imageCI.setMipLevels(1);
  imageCI.setArrayLayers(1);
  imageCI.setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
  nvvk::Image image = m_alloc->createImage(imageCI);

  // Image view
  vk::ImageViewCreateInfo viewCI;
  viewCI.setViewType(vk::ImageViewType::e2D);
  viewCI.setFormat(format);
  viewCI.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  viewCI.setImage(image.image);

  // Sampler
  vk::SamplerCreateInfo samplerCI;
  samplerCI.setMagFilter(vk::Filter::eLinear);
  samplerCI.setMinFilter(vk::Filter::eLinear);
  samplerCI.setMipmapMode(vk::SamplerMipmapMode::eLinear);
  samplerCI.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
  samplerCI.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
  samplerCI.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
  samplerCI.setMaxLod(1.0f);
  samplerCI.setBorderColor(vk::BorderColor::eFloatOpaqueWhite);

  target = m_alloc->createTexture(image, viewCI, samplerCI);
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    nvvk::cmdBarrierImageLayout(cmdBuf, target.image, VK_IMAGE_LAYOUT_UNDEFINED, target.descriptor.imageLayout);
  }
  m_alloc->finalizeAndReleaseStaging();

  // Render pass, one attachment, no depth
  vk::RenderPass renderpass = nvvk::createRenderPass(m_device, {format}, vk::Format::eUndefined, 1, true, true,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);

  // Using texture in framebuffer
  vk::FramebufferCreateInfo framebufferCI;
  framebufferCI.renderPass      = renderpass;
  framebufferCI.attachmentCount = 1;
  framebufferCI.pAttachments    = (vk::ImageView*)&target.descriptor.imageView;
  framebufferCI.width           = dim;
  framebufferCI.height          = dim;
  framebufferCI.layers          = 1;

  vk::Framebuffer framebuffer = m_device.createFramebuffer(framebufferCI);

  // Descriptors
  vk::DescriptorSetLayout descriptorsetlayout = m_device.createDescriptorSetLayout({});
  vk::DescriptorPoolSize  poolSizes{vk::DescriptorType::eCombinedImageSampler, 1};
  vk::DescriptorPool      descriptorpool = m_device.createDescriptorPool({{}, 1, 1, &poolSizes});
  vk::DescriptorSet       descriptorset = m_device.allocateDescriptorSets({descriptorpool, 1, &descriptorsetlayout})[0];

  // Pipeline layout
  vk::PipelineLayout pipelinelayout = m_device.createPipelineLayout({{}, 1, &descriptorsetlayout});

  // Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, pipelinelayout, renderpass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/integrate_brdf.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/integrate_brdf.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  vk::Pipeline pipeline = pipelineGenerator.createPipeline();

  // Render to texture
  vk::ClearValue          clearValues(std::array<float, 4>({0.0f, 0.0f, 0.0f, 1.0f}));
  vk::RenderPassBeginInfo renderPassBeginInfo{renderpass, framebuffer, {{}, {dim, dim}}, 1, &clearValues};

  nvvk::CommandPool        sc(m_device, m_queueIndex);
  const vk::CommandBuffer& cmdBuf = sc.createCommandBuffer();
  cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
  vk::Viewport viewport{0, 0, (float)dim, (float)dim, 0, 1};
  vk::Rect2D   scissor{{}, {dim, dim}};
  cmdBuf.setViewport(0, viewport);
  cmdBuf.setScissor(0, scissor);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
  cmdBuf.draw(3, 1, 0, 0);
  cmdBuf.endRenderPass();
  sc.submitAndWait(cmdBuf);

  // cleanup
  m_device.destroyPipeline(pipeline);
  m_device.destroyPipelineLayout(pipelinelayout);
  m_device.destroyRenderPass(renderpass);
  m_device.destroyFramebuffer(framebuffer);
  m_device.destroyDescriptorSetLayout(descriptorsetlayout);
  m_device.destroyDescriptorPool(descriptorpool);

  auto tEnd  = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  LOGI("Generating BRDF LUT took %f ms \n", tDiff);
}


// Pipeline layout
struct PushBlock
{
  nvmath::mat4f mvp;
  float         roughness{0.f};
  uint32_t      numSamples{32u};
} pushBlock;


//--------------------------------------------------------------------------------------------------
// Create the irradianceCube which is the light contribution
//
void SkydomePbr::prefilterDiffuse(uint32_t dim)
{
  auto tStart = std::chrono::high_resolution_clock::now();

  const vk::Extent2D size{dim, dim};
  vk::Format         format  = vk::Format::eR16G16B16A16Sfloat;
  const uint32_t     numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

  // Result Cubemap
  auto& filteredEnv = m_textures.irradianceCube;
  {
    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.minFilter  = vk::Filter::eLinear;
    samplerCreateInfo.magFilter  = vk::Filter::eLinear;
    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerCreateInfo.maxLod     = numMips;

    vk::ImageCreateInfo imageCreateInfo =
        nvvk::makeImageCubeCreateInfo(size, format, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, true);
    vk::DeviceSize bufferSize = size.width * size.height * 4 * sizeof(float);

    {
      nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
      nvvk::Image              image  = m_alloc->createImage(cmdBuf, bufferSize, nullptr, imageCreateInfo);
      vk::ImageViewCreateInfo  ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo, true);
      filteredEnv                     = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    }

    m_alloc->finalizeAndReleaseStaging();
  }

  // Render pass, one attachment, no depth
  vk::RenderPass renderpass = nvvk::createRenderPass(m_device, {format}, vk::Format::eUndefined, 1, true, true,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

  // Descriptors
  nvvk::DescriptorSetBindings descSetBind;
  descSetBind.addBinding(vk::DescriptorSetLayoutBinding(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  descSetBind.addBinding(vk::DescriptorSetLayoutBinding(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  vk::DescriptorSetLayout descSetLayout = descSetBind.createLayout(m_device);
  vk::DescriptorPool      descPool      = descSetBind.createPool(m_device);
  vk::DescriptorSet       descSet       = nvvk::allocateDescriptorSet(m_device, descPool, descSetLayout);

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(descSetBind.makeWrite(descSet, 0, &m_textures.txtHdr.descriptor));
  writes.emplace_back(descSetBind.makeWrite(descSet, 1, &m_textures.accelImpSmpl.descriptor));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);


  vk::PushConstantRange pushConstantRange{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                          sizeof(PushBlock)};
  vk::PipelineLayout    pipelinelayout = m_device.createPipelineLayout({{}, 1, &descSetLayout, 1, &pushConstantRange});

  // Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, pipelinelayout, renderpass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/filtercube.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/prefilter_diffuse.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  pipelineGenerator.addBindingDescription({0, sizeof(nvmath::vec3f)});
  pipelineGenerator.addAttributeDescriptions({{0, 0, vk::Format::eR32G32B32Sfloat, 0}});
  pipelineGenerator.depthStencilState.setDepthTestEnable(false);
  vk::Pipeline pipeline = pipelineGenerator.createPipeline();
  renderToCube(renderpass, filteredEnv, pipelinelayout, pipeline, descSet, dim, format, numMips);

  // todo: cleanup
  m_device.destroyRenderPass(renderpass, nullptr);

  m_device.destroyDescriptorPool(descPool, nullptr);
  m_device.destroyDescriptorSetLayout(descSetLayout, nullptr);
  m_device.destroyPipeline(pipeline, nullptr);
  m_device.destroyPipelineLayout(pipelinelayout, nullptr);

  auto tEnd  = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  LOGI("Generating irradiance/diffuse cube with %d mip levels took %f ms \n", numMips, tDiff);
}

//--------------------------------------------------------------------------------------------------
// Create the pre-filtered texture for glossy reflections
//
void SkydomePbr::prefilterGlossy(uint32_t dim)
{
  auto tStart = std::chrono::high_resolution_clock::now();

  const vk::Extent2D size{dim, dim};
  const uint32_t     numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;
  vk::Format         format  = vk::Format::eR16G16B16A16Sfloat;

  // Result Cubemap
  auto& filteredEnv = m_textures.prefilteredCube;
  {
    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.minFilter  = vk::Filter::eLinear;
    samplerCreateInfo.magFilter  = vk::Filter::eLinear;
    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerCreateInfo.maxLod     = numMips;

    vk::ImageCreateInfo imageCreateInfo =
        nvvk::makeImageCubeCreateInfo(size, format, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, true);
    vk::DeviceSize bufferSize = size.width * size.height * 4 * sizeof(float);

    {
      nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
      nvvk::Image              image  = m_alloc->createImage(cmdBuf, bufferSize, nullptr, imageCreateInfo);
      vk::ImageViewCreateInfo  ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo, true);
      filteredEnv                     = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    }
    m_alloc->finalizeAndReleaseStaging();
  }

  // Render pass, one attachment, no depth
  vk::RenderPass renderpass = nvvk::createRenderPass(m_device, {format}, vk::Format::eUndefined, 1, true, true,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

  // Descriptors
  nvvk::DescriptorSetBindings descSetBind;
  descSetBind.addBinding(vk::DescriptorSetLayoutBinding(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  descSetBind.addBinding(vk::DescriptorSetLayoutBinding(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  vk::DescriptorSetLayout descSetLayout = descSetBind.createLayout(m_device);
  vk::DescriptorPool      descPool      = descSetBind.createPool(m_device);
  vk::DescriptorSet       descSet       = nvvk::allocateDescriptorSet(m_device, descPool, descSetLayout);

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(descSetBind.makeWrite(descSet, 0, &m_textures.txtHdr.descriptor));
  writes.emplace_back(descSetBind.makeWrite(descSet, 1, &m_textures.accelImpSmpl.descriptor));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);


  vk::PushConstantRange pushConstantRange{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                          sizeof(PushBlock)};
  vk::PipelineLayout    pipelinelayout = m_device.createPipelineLayout({{}, 1, &descSetLayout, 1, &pushConstantRange});

  // Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, pipelinelayout, renderpass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/filtercube.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/prefilter_glossy.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  pipelineGenerator.addBindingDescriptions({{0, sizeof(nvmath::vec3f)}});
  pipelineGenerator.addAttributeDescriptions({{0, 0, vk::Format::eR32G32B32Sfloat, 0}});
  pipelineGenerator.depthStencilState.setDepthTestEnable(false);
  vk::Pipeline pipeline = pipelineGenerator.createPipeline();
  renderToCube(renderpass, filteredEnv, pipelinelayout, pipeline, descSet, dim, format, numMips);

  // todo: cleanup
  m_device.destroyRenderPass(renderpass, nullptr);

  m_device.destroyDescriptorPool(descPool, nullptr);
  m_device.destroyDescriptorSetLayout(descSetLayout, nullptr);
  m_device.destroyPipeline(pipeline, nullptr);
  m_device.destroyPipelineLayout(pipelinelayout, nullptr);

  auto tEnd  = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  LOGI("Generating pre-filtered cube with %d mip levels took %f ms \n", numMips, tDiff);
}

//--------------------------------------------------------------------------------------------------
// Render into all 6 sides of a cube with mipmaping
//
void SkydomePbr::renderToCube(const vk::RenderPass& renderpass,
                              nvvk::Texture&        filteredEnv,
                              vk::PipelineLayout    pipelinelayout,
                              vk::Pipeline          pipeline,
                              vk::DescriptorSet     descSet,
                              uint32_t              dim,
                              vk::Format            format,
                              const uint32_t        numMips)
{
  // Render
  vk::ClearValue clearValues[1];
  clearValues[0].color = std::array<float, 4>({0.0f, 0.0f, 0.2f, 0.0f});

  // Rendering to image, then copy to cube
  Offscreen offscreen = createOffscreen(dim, format, renderpass);

  vk::RenderPassBeginInfo renderPassBeginInfo{
      renderpass, offscreen.framebuffer, {vk::Offset2D{}, vk::Extent2D{(uint32_t)dim, (uint32_t)dim}}, 1, clearValues};

  nvmath::mat4f       mv[6];
  const nvmath::vec3f pos(0.0f, 0.0f, 0.0f);
  mv[0] = nvmath::look_at(pos, nvmath::vec3f(1.0f, 0.0f, 0.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));   // Positive X
  mv[1] = nvmath::look_at(pos, nvmath::vec3f(-1.0f, 0.0f, 0.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));  // Negative X
  mv[2] = nvmath::look_at(pos, nvmath::vec3f(0.0f, -1.0f, 0.0f), nvmath::vec3f(0.0f, 0.0f, -1.0f));  // Positive Y
  mv[3] = nvmath::look_at(pos, nvmath::vec3f(0.0f, 1.0f, 0.0f), nvmath::vec3f(0.0f, 0.0f, 1.0f));    // Negative Y
  mv[4] = nvmath::look_at(pos, nvmath::vec3f(0.0f, 0.0f, 1.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));   // Positive Z
  mv[5] = nvmath::look_at(pos, nvmath::vec3f(0.0f, 0.0f, -1.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));  // Negative Z


  nvvk::CommandPool        sc(m_device, m_queueIndex);
  const vk::CommandBuffer& cmdBuf = sc.createCommandBuffer();

  vk::Viewport viewport{0, 0, (float)dim, (float)dim, 0, 1};
  vk::Rect2D   scissor{vk::Offset2D{}, vk::Extent2D{(uint32_t)dim, (uint32_t)dim}};
  cmdBuf.setViewport(0, viewport);
  cmdBuf.setScissor(0, scissor);
  vk::ImageSubresourceRange subresourceRange{vk::ImageAspectFlagBits::eColor, 0, numMips, 0, 6};
  // Change image layout for all cubemap faces to transfer destination
  nvvk::cmdBarrierImageLayout(cmdBuf, filteredEnv.image, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eTransferDstOptimal, subresourceRange);

  nvmath::mat4f matPers = nvmath::perspectiveVK(90.0f, 1.0f, 0.1f, 10.0f);

  for(uint32_t mip = 0; mip < numMips; mip++)
  {
    for(uint32_t f = 0; f < 6; f++)
    {
      viewport.width  = static_cast<float>(dim * std::pow(0.5f, mip));
      viewport.height = static_cast<float>(dim * std::pow(0.5f, mip));
      cmdBuf.setViewport(0, 1, &viewport);
      // Render scene from cube face's point of view
      cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

      // Update shader push constant block

      float roughness     = (float)mip / (float)(numMips - 1);
      pushBlock.roughness = roughness;
      pushBlock.mvp       = matPers * mv[f];

      cmdBuf.pushConstants<PushBlock>(pipelinelayout,
                                      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushBlock);

      cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
      cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelinelayout, 0, descSet, nullptr);

      std::vector<vk::DeviceSize> offsets{0};
      cmdBuf.bindVertexBuffers(0, {m_vertices.buffer}, {offsets});
      cmdBuf.bindIndexBuffer(m_indices.buffer, 0, vk::IndexType::eUint32);
      cmdBuf.drawIndexed(36, 1, 0, 0, 0);

      cmdBuf.endRenderPass();

      nvvk::cmdBarrierImageLayout(cmdBuf, offscreen.image.image, vk::ImageLayout::eColorAttachmentOptimal,
                                  vk::ImageLayout::eTransferSrcOptimal, vk::ImageAspectFlagBits::eColor);

      // Copy region for transfer from framebuffer to cube face
      vk::ImageCopy copyRegion;
      copyRegion.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
      copyRegion.srcSubresource.baseArrayLayer = 0;
      copyRegion.srcSubresource.mipLevel       = 0;
      copyRegion.srcSubresource.layerCount     = 1;
      copyRegion.srcOffset                     = vk::Offset3D{};
      copyRegion.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
      copyRegion.dstSubresource.baseArrayLayer = f;
      copyRegion.dstSubresource.mipLevel       = mip;
      copyRegion.dstSubresource.layerCount     = 1;
      copyRegion.dstOffset                     = vk::Offset3D{};
      copyRegion.extent.width                  = static_cast<uint32_t>(viewport.width);
      copyRegion.extent.height                 = static_cast<uint32_t>(viewport.height);
      copyRegion.extent.depth                  = 1;

      cmdBuf.copyImage(offscreen.image.image, vk::ImageLayout::eTransferSrcOptimal, filteredEnv.image,
                       vk::ImageLayout::eTransferDstOptimal, copyRegion);

      // Transform framebuffer color attachment back
      nvvk::cmdBarrierImageLayout(cmdBuf, offscreen.image.image, vk::ImageLayout::eTransferSrcOptimal,
                                  vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor);
    }
  }
  nvvk::cmdBarrierImageLayout(cmdBuf, filteredEnv.image, vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal, subresourceRange);
  sc.submitAndWait(cmdBuf);

  // destroy
  m_device.destroyFramebuffer(offscreen.framebuffer, nullptr);
  m_alloc->destroy(offscreen.image);
  m_device.destroyImageView(offscreen.descriptor.imageView);
}

//--------------------------------------------------------------------------------------------------
// Offscreen framebuffer used by the render to cube (one of the face)
//
SkydomePbr::Offscreen SkydomePbr::createOffscreen(int dim, vk::Format format, const vk::RenderPass& renderpass)
{
  Offscreen offscreen;
  // Color attachment
  vk::ImageCreateInfo imageCreateInfo;
  imageCreateInfo.imageType     = vk::ImageType::e2D;
  imageCreateInfo.format        = format;
  imageCreateInfo.extent.width  = dim;
  imageCreateInfo.extent.height = dim;
  imageCreateInfo.extent.depth  = 1;
  imageCreateInfo.mipLevels     = 1;
  imageCreateInfo.arrayLayers   = 1;
  imageCreateInfo.usage         = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
  offscreen.image               = m_alloc->createImage(imageCreateInfo);

  vk::ImageViewCreateInfo colorImageView;
  colorImageView.viewType                    = vk::ImageViewType::e2D;
  colorImageView.format                      = format;
  colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  colorImageView.subresourceRange.levelCount = 1;
  colorImageView.subresourceRange.layerCount = 1;
  colorImageView.image                       = offscreen.image.image;
  offscreen.descriptor.imageView             = m_device.createImageView(colorImageView);

  vk::FramebufferCreateInfo fbufCreateInfo;
  fbufCreateInfo.renderPass      = renderpass;
  fbufCreateInfo.attachmentCount = 1;
  fbufCreateInfo.pAttachments    = &offscreen.descriptor.imageView;
  fbufCreateInfo.width           = dim;
  fbufCreateInfo.height          = dim;
  fbufCreateInfo.layers          = 1;
  offscreen.framebuffer          = m_device.createFramebuffer(fbufCreateInfo);

  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    nvvk::cmdBarrierImageLayout(cmdBuf, offscreen.image.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor);
  }
  m_alloc->finalizeAndReleaseStaging();
  offscreen.descriptor.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;

  return offscreen;
}

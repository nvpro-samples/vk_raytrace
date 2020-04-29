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


#include "raytracer.hpp"
#include "imgui.h"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"

extern std::vector<std::string> defaultSearchPaths;

Raytracer::Raytracer() = default;

//--------------------------------------------------------------------------------------------------
// Initializing the allocator and querying the raytracing properties
//
void Raytracer::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
{
  m_device     = device;
  m_queueIndex = queueIndex;
  m_debug.setup(device);
  m_alloc = allocator;

  // Requesting raytracing properties
  auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPropertiesNV>();


  m_rtBuilder.setup(device, allocator, queueIndex);
}

const nvvk::Texture& Raytracer::outputImage() const
{
  return m_raytracingOutput;
}

const int Raytracer::maxFrames() const
{
  return m_maxFrames;
}

void Raytracer::destroy()
{
  m_alloc->destroy(m_raytracingOutput);
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc->destroy(m_rtSBTBuffer);
}

void Raytracer::createOutputImage(vk::Extent2D size)
{
  m_alloc->destroy(m_raytracingOutput);

  m_outputSize = size;
  auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
  vk::DeviceSize imgSize = size.width * size.height * 4 * sizeof(float);
  vk::Format     format  = vk::Format::eR32G32B32A32Sfloat;

  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    vk::SamplerCreateInfo    samplerCreateInfo;  // default values
    vk::ImageCreateInfo      imageCreateInfo = nvvk::makeImage2DCreateInfo(size, format, usage);

    nvvk::Image image = m_alloc->createImage(cmdBuf, imgSize, nullptr, imageCreateInfo, vk::ImageLayout::eGeneral);
    vk::ImageViewCreateInfo ivInfo            = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_raytracingOutput                        = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    m_raytracingOutput.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }
  m_alloc->finalizeAndReleaseStaging();
}

void Raytracer::createDescriptorSet()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_binding.addBinding(vkDS(0, vkDT::eAccelerationStructureNV, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));
  m_binding.addBinding(vkDS(1, vkDT::eStorageImage, 1, vkSS::eRaygenNV));  // Output image

  m_rtDescPool      = m_binding.createPool(m_device);
  m_rtDescSetLayout = m_binding.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureNV                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureNV descAsInfo{1, &tlas};
  vk::DescriptorImageInfo imageInfo{{}, m_raytracingOutput.descriptor.imageView, vk::ImageLayout::eGeneral};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 0, &descAsInfo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 1, &imageInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  updateDescriptorSet();
}

void Raytracer::updateDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  // (1) Output buffer
  {
    vk::DescriptorImageInfo imageInfo{{}, m_raytracingOutput.descriptor.imageView, vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet  wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
    m_device.updateDescriptorSets(wds, nullptr);
  }
}

void Raytracer::createPipeline(const vk::DescriptorSetLayout& sceneDescSetLayout, const vk::DescriptorSetLayout& matDescSetLayout)
{
  std::vector<std::string> paths = defaultSearchPaths;
  vk::ShaderModule raygenSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytrace.rgen.spv", true, paths));
  vk::ShaderModule missSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytrace.rmiss.spv", true, paths));
  vk::ShaderModule shadowmissSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytraceShadow.rmiss.spv", true, paths));
  vk::ShaderModule chitSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytrace.rchit.spv", true, paths));
  vk::ShaderModule rahitSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytrace.rahit.spv", true, paths));

  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoNV rg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenNV, raygenSM, "main"});
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(rg);
  // Miss
  vk::RayTracingShaderGroupCreateInfoNV mg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, missSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Shadow Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, shadowmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Hit Group - Closest Hit + AnyHit
  vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  stages.push_back({{}, vk::ShaderStageFlagBits::eAnyHitNV, rahitSM, "main"});
  hg.setAnyHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(hg);

  // Push constant: ray depth, ...
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV, 0,
                                     sizeof(PushConstant)};

  // All 3 descriptors
  std::vector<vk::DescriptorSetLayout> allLayouts = {m_rtDescSetLayout, sceneDescSetLayout, matDescSetLayout};
  vk::PipelineLayoutCreateInfo         pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(allLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(allLayouts.data());
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);
  LOGI("createPipelineLayout - Done\n");


  // Assemble the shader stages and recursion depth info into the raytracing pipeline
  vk::RayTracingPipelineCreateInfoNV rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));
  rayPipelineInfo.setPStages(stages.data());
  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(m_groups.size()));
  rayPipelineInfo.setPGroups(m_groups.data());
  rayPipelineInfo.setMaxRecursionDepth(10);
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  LOGI("createRayTracingPipelineNV \n");
  m_rtPipeline = m_device.createRayTracingPipelineNV({}, rayPipelineInfo);
  LOGI("createRayTracingPipelineNV - Done\n");

  m_device.destroyShaderModule(raygenSM);
  m_device.destroyShaderModule(missSM);
  m_device.destroyShaderModule(shadowmissSM);
  m_device.destroyShaderModule(chitSM);
  m_device.destroyShaderModule(rahitSM);
}

void Raytracer::createShadingBindingTable()
{
  auto     groupCount      = static_cast<uint32_t>(m_groups.size());  // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;    // Size of a program identifier

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t             sbtSize = groupCount * groupHandleSize;
  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  m_device.getRayTracingShaderGroupHandlesNV(m_rtPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());

  m_rtSBTBuffer = m_alloc->createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc,
                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT").c_str());

  // Write the handles in the SBT
  void* mapped = m_alloc->map(m_rtSBTBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += groupHandleSize;
  }
  m_alloc->unmap(m_rtSBTBuffer);
}

void Raytracer::run(const vk::CommandBuffer& cmdBuf, const vk::DescriptorSet& sceneDescSet, const vk::DescriptorSet& matDescSet, int frame /*= 0*/)
{
  m_pushC.frame = frame;

  uint32_t progSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_rtPipelineLayout, 0,
                            {m_rtDescSet, sceneDescSet, matDescSet}, {});
  cmdBuf.pushConstants<PushConstant>(m_rtPipelineLayout,
                                     vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV, 0, m_pushC);

  vk::DeviceSize rayGenOffset   = 0 * progSize;
  vk::DeviceSize missOffset     = 1 * progSize;
  vk::DeviceSize missStride     = progSize;
  vk::DeviceSize hitGroupOffset = 3 * progSize;  // Jump over the 2 miss
  vk::DeviceSize hitGroupStride = progSize;

  cmdBuf.traceRaysNV(m_rtSBTBuffer.buffer, rayGenOffset,                    //
                     m_rtSBTBuffer.buffer, missOffset, missStride,          //
                     m_rtSBTBuffer.buffer, hitGroupOffset, hitGroupStride,  //
                     m_rtSBTBuffer.buffer, 0, 0,                            //
                     m_outputSize.width, m_outputSize.height,               //
                     1 /*, NVVKPP_DISPATCHER*/);
}

bool Raytracer::uiSetup()
{
  bool modified = false;
  if(ImGui::CollapsingHeader("Raytracing"))
  {
    modified = ImGui::SliderInt("Max Ray Depth ", &m_pushC.depth, 1, 10);
    modified = ImGui::SliderInt("Samples Per Frame", &m_pushC.samples, 1, 100) || modified;
    modified = ImGui::SliderInt("Max Iteration ", &m_maxFrames, 1, 1000) || modified;
    modified = ImGui::SliderFloat("HDR Multiplier", &m_pushC.hdrMultiplier, 0.f, 10.f, "%.3f", 3.0f) || modified;
  }
  return modified;
}

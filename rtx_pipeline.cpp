/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "vulkan/vulkan.hpp"

#include "nvh/alignment.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"
#include "rtx_pipeline.hpp"
#include "scene.hpp"
#include "tools.hpp"

#include "autogen/pathtrace.rahit.h"
#include "autogen/pathtrace.rchit.h"
#include "autogen/pathtrace.rgen.h"
#include "autogen/pathtrace.rmiss.h"
#include "autogen/pathtraceShadow.rmiss.h"


void RtxPipeline::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::Allocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);

  // Requesting ray tracing properties
  auto properties =
      physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
}

void RtxPipeline::destroy()
{
  m_pAlloc->destroy(m_rtSBTBuffer);

  m_device.destroy(m_rtPipelineLayout);
  m_device.destroy(m_rtPipeline);

  m_rtPipelineLayout = vk::PipelineLayout();
  m_rtPipeline       = vk::Pipeline();
}


void RtxPipeline::create(const vk::Extent2D& size, const std::vector<vk::DescriptorSetLayout>& rtDescSetLayouts, Scene* scene)
{
  MilliTimer timer;
  LOGI("Create RtxPipeline");

  m_nbHit = 1;  //scene->getStat().nbMaterials;

  updatePipeline(rtDescSetLayouts);
  createRtShaderBindingTable();
  timer.print();
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void RtxPipeline::updatePipeline(const std::vector<vk::DescriptorSetLayout>& rtDescSetLayouts)
{
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_rtShaderGroups.clear();

  // --- Pipeline Layout ----
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR
                                         | vk::ShaderStageFlagBits::eMissKHR,
                                     0, sizeof(RtxState)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());
  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);


  // --- Shaders ---
  std::vector<vk::PipelineShaderStageCreateInfo> stages;
  vk::RayTracingShaderGroupCreateInfoKHR group{{}, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

  vk::ShaderModule rgen     = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
  vk::ShaderModule rmiss    = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
  vk::ShaderModule rmissShd = nvvk::createShaderModule(m_device, pathtraceShadow_rmiss, sizeof(pathtraceShadow_rmiss));
  vk::ShaderModule rchit    = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
  vk::ShaderModule rahit    = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));

  // Raygen
  group.setGeneralShader(static_cast<uint32_t>(stages.size()));
  m_rtShaderGroups.push_back(group);
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, rgen, "main"});

  // Miss
  group.setGeneralShader(static_cast<uint32_t>(stages.size()));
  m_rtShaderGroups.push_back(group);
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, rmiss, "main"});

  // Miss - Shadow
  group.setGeneralShader(static_cast<uint32_t>(stages.size()));
  m_rtShaderGroups.push_back(group);
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, rmissShd, "main"});

  // Hit Group
  group.setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
  group.setGeneralShader(VK_SHADER_UNUSED_KHR);
  for(uint32_t i = 0; i < m_nbHit; i++)
  {
    group.setClosestHitShader(static_cast<uint32_t>(stages.size()));
    stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, rchit, "main"});
    group.setAnyHitShader(static_cast<uint32_t>(stages.size()));
    stages.push_back({{}, vk::ShaderStageFlagBits::eAnyHitKHR, rahit, "main"});
    m_rtShaderGroups.push_back(group);
  }

  // --- Pipeline ---
  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
  rayPipelineInfo.setPStages(stages.data());

  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(m_rtShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.setPGroups(m_rtShaderGroups.data());

  rayPipelineInfo.setMaxPipelineRayRecursionDepth(2);  // Ray depth
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  m_rtPipeline = static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo));

  // --- Clean up ---
  m_device.destroy(rgen);
  m_device.destroy(rmiss);
  m_device.destroy(rmissShd);
  m_device.destroy(rchit);
  m_device.destroy(rahit);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void RtxPipeline::createRtShaderBindingTable()
{
  m_pAlloc->destroy(m_rtSBTBuffer);

  auto     groupCount       = static_cast<uint32_t>(m_rtShaderGroups.size());  // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize  = m_rtProperties.shaderGroupHandleSize;            // Size of a program identifier
  uint32_t groupSizeAligned = nvh::align_up(groupHandleSize, m_rtProperties.shaderGroupBaseAlignment);

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * groupSizeAligned;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  auto result = m_device.getRayTracingShaderGroupHandlesKHR(m_rtPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());
  // Write the handles in the SBT
  m_rtSBTBuffer =
      m_pAlloc->createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  NAME_VK(m_rtSBTBuffer.buffer);

  // Write the handles in the SBT
  void* mapped = m_pAlloc->map(m_rtSBTBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += groupSizeAligned;
  }
  m_pAlloc->unmap(m_rtSBTBuffer);


  m_pAlloc->finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void RtxPipeline::run(const vk::CommandBuffer&              cmdBuf,
                      const vk::Extent2D&                   size,
                      nvvk::ProfilerVK&                     profiler,
                      const std::vector<vk::DescriptorSet>& descSets)
{
  LABEL_SCOPE_VK(cmdBuf);

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipelineLayout, 0, descSets, {});
  cmdBuf.pushConstants<RtxState>(m_rtPipelineLayout,
                                 vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR
                                     | vk::ShaderStageFlagBits::eMissKHR,
                                 0, m_state);

  // Size of a program identifier
  uint32_t groupSize   = nvh::align_up(m_rtProperties.shaderGroupHandleSize, m_rtProperties.shaderGroupBaseAlignment);
  uint32_t groupStride = groupSize;
  vk::DeviceAddress sbtAddress = m_device.getBufferAddress({m_rtSBTBuffer.buffer});

  using Stride = vk::StridedDeviceAddressRegionKHR;
  std::array<Stride, 4> strideAddresses{Stride{sbtAddress + 0u * groupSize, groupStride, groupSize * 1},  // raygen
                                        Stride{sbtAddress + 1u * groupSize, groupStride, groupSize * 2},  // miss
                                        Stride{sbtAddress + 3u * groupSize, groupStride, groupSize * 1},  // hit
                                        Stride{0u, 0u, 0u}};

  cmdBuf.traceRaysKHR(&strideAddresses[0], &strideAddresses[1], &strideAddresses[2],
                      &strideAddresses[3],          //
                      size.width, size.height, 1);  //
}

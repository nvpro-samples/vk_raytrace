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


void RtxPipeline::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);

  // Requesting ray tracing properties
  auto properties =
      physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

  m_stbWrapper.setup(device, familyIndex, allocator, m_rtProperties);
}

void RtxPipeline::destroy()
{
  m_stbWrapper.destroy();

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

  createPipelineLayout(rtDescSetLayouts);
  createPipeline();
  timer.print();
}


//--------------------------------------------------------------------------------------------------
// The layout has a push constant and the incoming descriptors are:
// acceleration structure, offscreen image, scene data, hdr
//
void RtxPipeline::createPipelineLayout(const std::vector<vk::DescriptorSetLayout>& rtDescSetLayouts)
{
  m_device.destroy(m_rtPipelineLayout);
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR
                                         | vk::ShaderStageFlagBits::eMissKHR,
                                     0, sizeof(RtxState)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());
  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void RtxPipeline::createPipeline()
{
  m_device.destroy(m_rtPipeline);
  m_rtShaderGroups.clear();

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
    if(m_enableAnyhit)
    {
      group.setAnyHitShader(static_cast<uint32_t>(stages.size()));
      stages.push_back({{}, vk::ShaderStageFlagBits::eAnyHitKHR, rahit, "main"});
    }
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

  auto resultValue = m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo);
  m_rtPipeline = resultValue.value;

  // --- SBT ---
  m_stbWrapper.create(m_rtPipeline, rayPipelineInfo);

  // --- Clean up ---
  m_device.destroy(rgen);
  m_device.destroy(rmiss);
  m_device.destroy(rmissShd);
  m_device.destroy(rchit);
  m_device.destroy(rahit);
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

  auto regions = m_stbWrapper.getRegions();
  cmdBuf.traceRaysKHR(regions[0], regions[1], regions[2], regions[3], size.width, size.height, 1);
}

void RtxPipeline::useAnyHit(bool enable)
{
  m_enableAnyhit = enable;
  createPipeline();
}

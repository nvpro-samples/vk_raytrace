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
 *  Implement the RTX ray tracing pipeline
 */


#include <future>

#include "nvh/alignment.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"
#include "rtx_pipeline.hpp"
#include "scene.hpp"
#include "tools.hpp"

// Shaders
#include "autogen/pathtrace.rahit.h"
#include "autogen/pathtrace.rchit.h"
#include "autogen/pathtrace.rgen.h"
#include "autogen/pathtrace.rmiss.h"
#include "autogen/pathtraceShadow.rmiss.h"

//--------------------------------------------------------------------------------------------------
// Typical resource holder + query for capabilities
//
void RtxPipeline::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);

  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  properties.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(physicalDevice, &properties);

  m_sbtWrapper.setup(device, familyIndex, allocator, m_rtProperties);
}

//--------------------------------------------------------------------------------------------------
// Destroy all allocated resources
//
void RtxPipeline::destroy()
{
  m_sbtWrapper.destroy();

  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);

  m_rtPipelineLayout = VkPipelineLayout();
  m_rtPipeline       = VkPipeline();
}

//--------------------------------------------------------------------------------------------------
// Creation of the pipeline and layout
//
void RtxPipeline::create(const VkExtent2D& size, const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts, Scene* scene)
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
void RtxPipeline::createPipelineLayout(const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts)
{
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);

  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(RtxState)};

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;
  pipelineLayoutCreateInfo.setLayoutCount         = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts            = rtDescSetLayouts.data();
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void RtxPipeline::createPipeline()
{
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eAnyHit,
    eShaderGroupCount
  };

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point

  // Raygen
  stage.module    = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;

  // Miss
  stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;

  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module   = nvvk::createShaderModule(m_device, pathtraceShadow_rmiss, sizeof(pathtraceShadow_rmiss));
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;

  // Hit Group - Closest Hit
  stage.module        = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;

  // Hit Group - Any Hit
  stage.module    = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));
  stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  stages[eAnyHit] = stage;


  // Shader groups
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
  VkRayTracingShaderGroupCreateInfoKHR              group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  groups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  groups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  groups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  if(m_enableAnyhit)
    group.anyHitShader = eAnyHit;
  groups.push_back(group);

  // --- Pipeline ---
  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());  // Stages are shaders
  rayPipelineInfo.pStages    = stages.data();

  rayPipelineInfo.groupCount = static_cast<uint32_t>(groups.size());  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.pGroups    = groups.data();

  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;

  // Create a deferred operation (compiling in parallel)
  bool                   useDeferred{true};
  VkResult               result;
  VkDeferredOperationKHR deferredOp{VK_NULL_HANDLE};
  if(useDeferred)
  {
    result = vkCreateDeferredOperationKHR(m_device, nullptr, &deferredOp);
    assert(result == VK_SUCCESS);
  }

  vkCreateRayTracingPipelinesKHR(m_device, deferredOp, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);

  if(useDeferred)
  {
    // Query the maximum amount of concurrency and clamp to the desired maximum
    uint32_t maxThreads{8};
    uint32_t numLaunches = std::min(vkGetDeferredOperationMaxConcurrencyKHR(m_device, deferredOp), maxThreads);

    std::vector<std::future<void>> joins;
    for(uint32_t i = 0; i < numLaunches; i++)
    {
      VkDevice device{m_device};
      joins.emplace_back(std::async(std::launch::async, [device, deferredOp]() {
        // A return of VK_THREAD_IDLE_KHR should queue another job
        vkDeferredOperationJoinKHR(device, deferredOp);
      }));
    }

    for(auto& f : joins)
    {
      f.get();
    }

    // deferred operation is now complete.  'result' indicates success or failure
    result = vkGetDeferredOperationResultKHR(m_device, deferredOp);
    assert(result == VK_SUCCESS);
    vkDestroyDeferredOperationKHR(m_device, deferredOp, nullptr);
  }


  // --- SBT ---
  m_sbtWrapper.create(m_rtPipeline, rayPipelineInfo);

  // --- Clean up ---
  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void RtxPipeline::run(const VkCommandBuffer& cmdBuf, const VkExtent2D& size, nvvk::ProfilerVK& profiler, const std::vector<VkDescriptorSet>& descSets)
{
  LABEL_SCOPE_VK(cmdBuf);

  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                     0, sizeof(RtxState), &m_state);


  auto& regions = m_sbtWrapper.getRegions();
  vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], size.width, size.height, 1);
}

//--------------------------------------------------------------------------------------------------
// Toggle the usage of Anyhit. Not having anyhit can be faster, but the scene must but fully opaque
//
void RtxPipeline::useAnyHit(bool enable)
{
  m_enableAnyhit = enable;
  createPipeline();
}

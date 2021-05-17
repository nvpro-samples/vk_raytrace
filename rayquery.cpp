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
#include "rayquery.hpp"
#include "scene.hpp"
#include "tools.hpp"

#include "autogen/pathtrace.comp.h"
//--------------------------------------------------------------------------------------------------
//
//
void RayQuery::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
//
//
void RayQuery::destroy()
{
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_pipeline);
  m_pipelineLayout = vk::PipelineLayout{};
  m_pipeline       = vk::Pipeline{};
}

//--------------------------------------------------------------------------------------------------
// Creation of the RQ pipeline
//
void RayQuery::create(const vk::Extent2D& size, const std::vector<vk::DescriptorSetLayout>& rtDescSetLayouts, Scene* scene)
{
  MilliTimer timer;
  LOGI("Create Ray Query Pipeline");

  std::vector<vk::PushConstantRange> push_constants;
  push_constants.push_back({vk::ShaderStageFlagBits::eCompute, 0, sizeof(RtxState)});

  vk::PipelineLayoutCreateInfo layout_info;
  layout_info.setPushConstantRangeCount(static_cast<uint32_t>(push_constants.size()));
  layout_info.setPPushConstantRanges(push_constants.data());
  layout_info.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  layout_info.setPSetLayouts(rtDescSetLayouts.data());
  m_pipelineLayout = m_device.createPipelineLayout(layout_info);

  vk::ComputePipelineCreateInfo computePipelineCreateInfo{{}, {}, m_pipelineLayout};
  computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, pathtrace_comp, sizeof(pathtrace_comp));
  computePipelineCreateInfo.stage.stage  = vk::ShaderStageFlagBits::eCompute;
  computePipelineCreateInfo.stage.pName  = "main";

  auto resultValue = m_device.createComputePipeline({}, computePipelineCreateInfo);
  m_pipeline = resultValue.value;
  m_debug.setObjectName(m_pipeline, "RayQuery");
  m_device.destroy(computePipelineCreateInfo.stage.module);

  timer.print();
}


//--------------------------------------------------------------------------------------------------
// Executing the Ray Query compute shader
//
#define GROUP_SIZE 8  // Same group size as in compute shader
void RayQuery::run(const vk::CommandBuffer&              cmdBuf,
                   const vk::Extent2D&                   size,
                   nvvk::ProfilerVK&                     profiler,
                   const std::vector<vk::DescriptorSet>& descSets)
{
  // Preparing for the compute shader
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, descSets, {});

  // Sending the push constant information
  cmdBuf.pushConstants(m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(RtxState), &m_state);

  // Dispatching the shader
  cmdBuf.dispatch((size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
}

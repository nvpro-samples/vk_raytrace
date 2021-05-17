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


#pragma once
#include "vulkan/vulkan.hpp"

#include "nvmath/nvmath_glsltypes.h"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

using namespace nvmath;
#include "nvvk/profiler_vk.hpp"
#include "renderer.h"
#include "structures.h"

/*

Creating the Compute ray query renderer 
* Requiring:  
  - Acceleration structure (AccelSctruct / Tlas)
  - An image (Post StoreImage)
  - The glTF scene (vertex, index, materials, ... )

* Usage
  - setup as usual
  - create
  - run
*/
class RayQuery : public Renderer
{
public:
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator) override;
  void destroy() override;
  void create(const vk::Extent2D& size, const std::vector<vk::DescriptorSetLayout>& rtDescSetLayouts, Scene* scene) override;
  void              run(const vk::CommandBuffer&              cmdBuf,
                        const vk::Extent2D&                   size,
                        nvvk::ProfilerVK&                     profiler,
                        const std::vector<vk::DescriptorSet>& descSets) override;
  const std::string name() override { return std::string("RQ"); }

private:
  uint32_t m_nbHit;

private:
  // Setup
  nvvk::ResourceAllocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil  m_debug;   // Utility to name objects
  vk::Device       m_device;
  uint32_t         m_queueIndex;


  vk::PipelineLayout m_pipelineLayout;
  vk::Pipeline       m_pipeline;
};

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
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "vulkan/vulkan.hpp"

using namespace nvmath;
#include "structures.h"

// Forward declaration
class Scene;

class Renderer
{
public:
  virtual void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator) = 0;
  virtual void destroy()                                                = 0;
  virtual void run(const vk::CommandBuffer&              cmdBuf,
                   const vk::Extent2D&                   size,
                   nvvk::ProfilerVK&                     profiler,
                   const std::vector<vk::DescriptorSet>& extraDescSets) = 0;
  virtual void create(const vk::Extent2D&                         size,
                      const std::vector<vk::DescriptorSetLayout>& extraDescSetsLayout,
                      Scene*                                      _scene = nullptr)                          = 0;

  virtual const std::string name() = 0;
  RtxState                  m_state;
};

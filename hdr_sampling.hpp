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
//////////////////////////////////////////////////////////////////////////


#include "nvvk/debug_util_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include <array>
#include <vector>

//--------------------------------------------------------------------------------------------------
// Load an environment image (HDR) and create an acceleration structure for
// important light sampling.
class HdrSampling
{
public:
  HdrSampling() = default;

  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator);
  void loadEnvironment(const std::string& hrdImage);


  void  destroy();
  float getIntegral() { return m_integral; }
  float getAverage() { return m_average; }

  // Resources
  nvvk::Texture m_texHdr;
  nvvk::Buffer  m_accelImpSmpl;

private:
  struct EnvAccel
  {
    uint32_t alias{0};
    float    q{0.f};
    float    pdf{0.f};
    float    aliasPdf{0.f};
  };

  VkDevice                 m_device{VK_NULL_HANDLE};
  uint32_t                 m_queueIndex{0};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::DebugUtil          m_debug;

  float m_integral{1.f};
  float m_average{1.f};


  float                 buildAliasmap(const std::vector<float>& data, std::vector<EnvAccel>& accel);
  std::vector<EnvAccel> createEnvironmentAccel(const float* pixels, VkExtent2D& size);
};

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


#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/images_vk.hpp"
#include <array>
#include <vector>
#include <vulkan/vulkan.hpp>

//--------------------------------------------------------------------------------------------------
// Load an environment image (HDR) and create an acceleration structure for
// important light sampling.
class HdrSampling
{
public:
  HdrSampling() = default;

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator);
  void loadEnvironment(const std::string& hrdImage);


  void  destroy();
  float getIntegral() { return m_integral; }
  float getAverage() { return m_average; }

  struct Textures
  {
    nvvk::Texture txtHdr;        // HDR environment texture
    nvvk::Texture accelImpSmpl;  // Importance Sampling
  } m_textures;


private:
  struct Env_accel
  {
    uint32_t alias{0};
    float    q{0.f};
    float    pdf{0.f};
    float    _padding{0.f};
  };

  vk::Device       m_device;
  uint32_t         m_queueIndex{0};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::DebugUtil  m_debug;

  float m_integral{1.f};
  float m_average{1.f};


  float build_alias_map(const std::vector<float>& data, std::vector<Env_accel>& accel);
  void  createEnvironmentAccelTexture(const float* pixels, vk::Extent2D& size, nvvk::Texture& accelTex);
};

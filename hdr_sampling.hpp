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


#pragma once
//////////////////////////////////////////////////////////////////////////


#include "nvvk/allocator_vk.hpp"
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

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::Allocator* allocator);
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
  nvvk::Allocator* m_alloc{nullptr};
  nvvk::DebugUtil  m_debug;

  float m_integral{1.f};
  float m_average{1.f};


  float build_alias_map(const std::vector<float>& data, std::vector<Env_accel>& accel);
  void  createEnvironmentAccelTexture(const float* pixels, vk::Extent2D& size, nvvk::Texture& accelTex);
};

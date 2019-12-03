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

#include "nvvkpp/allocator_dma_vkpp.hpp"
#include "nvvkpp/debug_util_vkpp.hpp"
#include "nvvkpp/images_vkpp.hpp"
#include <array>
#include <vector>
#include <vulkan/vulkan.hpp>

using nvvkBuffer   = nvvkpp::BufferDma;
using nvvkTexture  = nvvkpp::TextureDma;
using nvvkImage    = nvvkpp::ImageDma;
using nvvkAlloc    = nvvkpp::AllocatorDma;
using nvvkMemAlloc = nvvk::DeviceMemoryAllocator;

// Load an environment image (HDR) and create the cubic textures for glossy reflection and diffuse illumination.
// Creates also the BRDF lookup table and an acceleration structure for lights
// It also has the ability to render a cube with the environment, use by the rasterizer.
class SkydomePbr
{
public:
  SkydomePbr() = default;

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvkMemAlloc& memAlloc)
  {
    m_device = device;
    m_alloc.init(device, &memAlloc);
    m_queueIndex = familyIndex;
    m_debug.setup(device);
  }

  void loadEnvironment(const std::string& hrdImage);
  void create(const vk::DescriptorBufferInfo& sceneBufferDesc, const vk::RenderPass& renderPass);
  void draw(const vk::CommandBuffer& commandBuffer);
  void destroy();

  struct Textures
  {
    nvvkTexture txtHdr;           // HDR environment texture
    nvvkTexture lutBrdf;          // BRDF Lookup table
    nvvkTexture accelImpSmpl;     // Importance Sampling
    nvvkTexture irradianceCube;   // Irradiance/light contribution
    nvvkTexture prefilteredCube;  // specular/glossy reflection
  } m_textures;

  enum Descriptors
  {
    eScene,
    eMaterial
  };

  vk::DescriptorSet       m_descriptorSet[2];
  vk::DescriptorSetLayout m_descriptorSetLayout[2];
  vk::DescriptorPool      m_descriptorpool;
  vk::Pipeline            m_pipeline;
  vk::PipelineLayout      m_pipelineLayout;
  vk::RenderPass          m_renderPass;
  vk::Device              m_device;

private:
  nvvkBuffer m_vertices;
  nvvkBuffer m_indices;

  uint32_t          m_queueIndex{0};
  nvvkAlloc         m_alloc;
  nvvkpp::DebugUtil m_debug;

  void createCube();
  void createEnvironmentAccelTexture(const float* pixels, vk::Extent2D& size, nvvkTexture& accelTex);
  void integrateBrdf(uint32_t dim);
  void createPipelines(const vk::DescriptorBufferInfo& sceneBufferDesc);
  void prefilterDiffuse(uint32_t dim);
  void prefilterGlossy(uint32_t dim);
  void renderToCube(const vk::RenderPass& renderpass,
                    nvvkTexture&          filteredEnv,
                    vk::PipelineLayout    pipelinelayout,
                    vk::Pipeline          pipeline,
                    vk::DescriptorSet     descSet,
                    uint32_t              dim,
                    vk::Format            format,
                    const uint32_t        numMips);

  struct Offscreen
  {
    nvvkImage               image;
    vk::DescriptorImageInfo descriptor;
    vk::Framebuffer         framebuffer;
  };

  Offscreen createOffscreen(int dim, vk::Format format, const vk::RenderPass& renderpass);
};

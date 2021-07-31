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
 *  HDR sampling is loading an HDR image and creating an acceleration structure for 
 *  sampling the environment. 
 */


#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>

#include "stb_image.h"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "hdr_sampling.hpp"


void HdrSampling::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_alloc      = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);
}

void HdrSampling::destroy()
{
  m_alloc->destroy(m_texHdr);
  m_alloc->destroy(m_accelImpSmpl);
}


//--------------------------------------------------------------------------------------------------
// Loading the HDR environment texture (HDR) and create the important accel structure
//
void HdrSampling::loadEnvironment(const std::string& hrdImage)
{
  destroy();

  int32_t width{0};
  int32_t height{0};
  int32_t component{0};

  float*       pixels     = stbi_loadf(hrdImage.c_str(), &width, &height, &component, STBI_rgb_alpha);
  VkDeviceSize bufferSize = width * height * 4 * sizeof(float);
  VkExtent2D   imgSize{(uint32_t)width, (uint32_t)height};

  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  // The map is parameterized with the U axis corresponding to the azimuthal angle, and V to the polar angle
  // Therefore, in U the sampler will use VK_SAMPLER_ADDRESS_MODE_REPEAT (default), but V needs to use
  // CLAMP_TO_EDGE to avoid having light leaking from one pole to another.
  samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkFormat          format       = VK_FORMAT_R32G32B32A32_SFLOAT;
  VkImageCreateInfo icInfo       = nvvk::makeImage2DCreateInfo(imgSize, format);

  {
    // We are using a different index (1), to allow loading in a different
    // queue/thread that the display (0)
    VkQueue queue;
    vkGetDeviceQueue(m_device, m_queueIndex, 1, &queue);

    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex, queue);
    nvvk::Image              image  = m_alloc->createImage(cmdBuf, bufferSize, pixels, icInfo);
    VkImageViewCreateInfo    ivInfo = nvvk::makeImageViewCreateInfo(image.image, icInfo);
    m_texHdr                        = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    NAME_VK(m_texHdr.image);

    auto envAccel  = createEnvironmentAccel(pixels, imgSize);
    m_accelImpSmpl = m_alloc->createBuffer(cmdBuf, envAccel, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NAME_VK(m_accelImpSmpl.buffer);
  }
  m_alloc->finalizeAndReleaseStaging();


  stbi_image_free(pixels);
}

//--------------------------------------------------------------------------------------------------
// Build alias map for the importance sampling: Each texel is associated to another texel, or alias,
// so that their combined intensities are a close as possible to the average of the environment map.
// This will later allow the sampling shader to uniformly select a texel in the environment, and
// select either that texel or its alias depending on their relative intensities
//
float HdrSampling::buildAliasmap(const std::vector<float>& data, std::vector<EnvAccel>& accel)
{
  auto size = static_cast<uint32_t>(data.size());

  // Compute the integral of the emitted radiance of the environment map
  // Since each element in data is already weighted by its solid angle
  // the integral is a simple sum
  float sum = std::accumulate(data.begin(), data.end(), 0.f);

  // For each texel, compute the ratio q between the emitted radiance of the texel and the average
  // emitted radiance over the entire sphere
  // We also initialize the aliases to identity, ie. each texel is its own alias
  auto  fSize          = static_cast<float>(size);
  float inverseAverage = fSize / sum;
  for(uint32_t i = 0; i < size; ++i)
  {
    accel[i].q     = data[i] * inverseAverage;
    accel[i].alias = i;
  }

  // Partition the texels according to their emitted radiance ratio wrt. average.
  // Texels with a value q < 1 (ie. below average) are stored incrementally from the beginning of the
  // array, while texels emitting higher-than-average radiance are stored from the end of the array
  std::vector<uint32_t> partitionTable(size);
  uint32_t              s     = 0u;
  uint32_t              large = size;
  for(uint32_t i = 0; i < size; ++i)
  {
    if(accel[i].q < 1.f)
      partitionTable[s++] = i;
    else
      partitionTable[--large] = i;
  }

  // Associate the lower-energy texels to higher-energy ones. Since the emission of a high-energy texel may
  // be vastly superior to the average,
  for(s = 0; s < large && large < size; ++s)
  {
    // Index of the smaller energy texel
    const uint32_t smallEnergyIndex = partitionTable[s];

    // Index of the higher energy texel
    const uint32_t highEnergyIndex = partitionTable[large];

    // Associate the texel to its higher-energy alias
    accel[smallEnergyIndex].alias = highEnergyIndex;

    // Compute the difference between the lower-energy texel and the average
    const float differenceWithAverage = 1.f - accel[smallEnergyIndex].q;

    // The goal is to obtain texel couples whose combined intensity is close to the average.
    // However, some texels may have low energies, while others may have very high intensity
    // (for example a sunset: the sky is quite dark, but the sun is still visible). In this case
    // it may not be possible to obtain a value close to average by combining only two texels.
    // Instead, we potentially associate a single high-energy texel to many smaller-energy ones until
    // the combined average is similar to the average of the environment map.
    // We keep track of the combined average by subtracting the difference between the lower-energy texel and the average
    // from the ratio stored in the high-energy texel.
    accel[highEnergyIndex].q -= differenceWithAverage;

    // If the combined ratio to average of the higher-energy texel reaches 1, a balance has been found
    // between a set of low-energy texels and the higher-energy one. In this case, we will use the next
    // higher-energy texel in the partition when processing the next texel.
    if(accel[highEnergyIndex].q < 1.0f)
      large++;
  }
  // Return the integral of the emitted radiance. This integral will be used to normalize the probability
  // distribution function (PDF) of each pixel
  return sum;
}

// CIE luminance
inline float luminance(const float* color)
{
  return color[0] * 0.2126f + color[1] * 0.7152f + color[2] * 0.0722f;
}

//--------------------------------------------------------------------------------------------------
// Create acceleration data for importance sampling
// See:  https://arxiv.org/pdf/1901.05423.pdf
std::vector<HdrSampling::EnvAccel> HdrSampling::createEnvironmentAccel(const float* pixels, VkExtent2D& size)
{
  const uint32_t rx = size.width;
  const uint32_t ry = size.height;

  // Create importance sampling data
  std::vector<EnvAccel> envAccel(rx * ry);
  std::vector<float>    importanceData(rx * ry);
  float                 cosTheta0 = 1.0f;
  const float           stepPhi   = float(2.0 * M_PI) / float(rx);
  const float           stepTheta = float(M_PI) / float(ry);
  double                total     = 0;

  // For each texel of the environment map, we compute the related solid angle
  // subtended by the texel, and store the weighted luminance in importance_data,
  // representing the amount of energy emitted through each texel.
  // Also compute the average CIE luminance to drive the tonemapping of the final image
  for(uint32_t y = 0; y < ry; ++y)
  {
    const float theta1    = float(y + 1) * stepTheta;
    const float cosTheta1 = std::cos(theta1);
    const float area      = (cosTheta0 - cosTheta1) * stepPhi;  // solid angle
    cosTheta0             = cosTheta1;

    for(uint32_t x = 0; x < rx; ++x)
    {
      const uint32_t idx          = y * rx + x;
      const uint32_t idx4         = idx * 4;
      float          cieLuminance = luminance(&pixels[idx4]);
      importanceData[idx]         = area * std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
      total += cieLuminance;
    }
  }

  m_average = static_cast<float>(total) / static_cast<float>(rx * ry);

  // Build the alias map, which aims at creating a set of texel couples
  // so that all couples emit roughly the same amount of energy. To this aim,
  // each smaller radiance texel will be assigned an "alias" with higher emitted radiance
  // As a byproduct this function also returns the integral of the radiance emitted by the environment
  m_integral = buildAliasmap(importanceData, envAccel);

  // We deduce the PDF of each texel by normalizing its emitted radiance by the radiance integral
  const float invEnvIntegral = 1.0f / m_integral;
  for(uint32_t i = 0; i < rx * ry; ++i)
  {
    const uint32_t idx4 = i * 4;
    envAccel[i].pdf     = std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2])) * invEnvIntegral;
  }

  // At runtime a texel will be uniformly chosen. Whether that texel or its alias is
  // selected depends on the relative emitted radiances of the two texels.
  // We store the PDF of the alias together with the PDF of the first member, so that both PDFs are
  // available in a single lookup
  for(uint32_t i = 0; i < rx * ry; ++i)
  {
    const uint32_t aliasIdx = envAccel[i].alias;
    envAccel[i].aliasPdf    = envAccel[aliasIdx].pdf;
  }

  return envAccel;
}

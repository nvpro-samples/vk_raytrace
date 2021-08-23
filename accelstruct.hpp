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
#include "nvh/gltfscene.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"


/*
 
 This is for uploading a glTF scene to an acceleration structure.
 - setup as usual
 - create passing the glTF scene and the buffer of vertices and indices pre-constructed
 - retrieve the TLAS with getTlas
 - get the descriptor set and layout 

*/
class AccelStructure
{
public:
  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator);
  void destroy();
  void create(nvh::GltfScene& gltfScene, const std::vector<nvvk::Buffer>& vertex, const std::vector<nvvk::Buffer>& index);

  VkAccelerationStructureKHR getTlas() { return m_rtBuilder.getAccelerationStructure(); }
  VkDescriptorSetLayout      getDescLayout() { return m_rtDescSetLayout; }
  VkDescriptorSet            getDescSet() { return m_rtDescSet; }

private:
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::GltfPrimMesh& prim, VkBuffer vertex, VkBuffer index);
  void                                  createBottomLevelAS(nvh::GltfScene& gltfScene, const std::vector<nvvk::Buffer>& vertex, const std::vector<nvvk::Buffer>& index);
  void                                  createTopLevelAS(nvh::GltfScene& gltfScene);
  void                                  createRtDescriptorSet();


  // Setup
  nvvk::ResourceAllocator* m_pAlloc{nullptr};  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;            // Utility to name objects
  VkDevice                 m_device{nullptr};
  uint32_t                 m_queueIndex{0};

  nvvk::RaytracingBuilderKHR m_rtBuilder;

  VkDescriptorPool      m_rtDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_rtDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet       m_rtDescSet{VK_NULL_HANDLE};
};

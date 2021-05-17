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

#include <string>

#include "vulkan/vulkan.hpp"

#include "nvh/gltfscene.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "structures.h"


/*
  - Loading and storing the glTF scene
  - Creates the buffers and descriptor set for the scene
*/
class Scene
{
public:
  enum EBuffer
  {
    eCameraMat,
    eMaterial,
    eInstData,
    eLights,
  };


  enum EBuffers
  {
    eVertex,
    eIndex,
    eLast_elem
  };

public:
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator);
  bool load(const std::string& filename);

  void createInstanceDataBuffer(vk::CommandBuffer cmdBuf, nvh::GltfScene& gltf);

  void createVertexBuffer(vk::CommandBuffer cmdBuf, const nvh::GltfScene& gltf);

  void setCameraFromScene(const std::string& filename, const nvh::GltfScene& gltf);

  bool loadGltfScene(const std::string& filename, tinygltf::Model& tmodel);

  void createLightBuffer(vk::CommandBuffer cmdBuf, const nvh::GltfScene& gltf);
  void createMaterialBuffer(vk::CommandBuffer cmdBuf, const nvh::GltfScene& gltf);
  void destroy();
  void updateCamera(const vk::CommandBuffer& cmdBuf, float aspectRatio);


  vk::DescriptorSetLayout          getDescLayout() { return m_descSetLayout; }
  vk::DescriptorSet                getDescSet() { return m_descSet; }
  nvh::GltfScene&                  getScene() { return m_gltf; }
  nvh::GltfStats&                  getStat() { return m_stats; }
  const std::vector<nvvk::Buffer>& getBuffers(EBuffers b) { return m_buffers[b]; }
  const std::string&               getSceneName() const { return m_sceneName; }
  SceneCamera&                     getCamera() { return m_camera; }

private:
  void createTextureImages(vk::CommandBuffer cmdBuf, tinygltf::Model& gltfModel);
  void createDescriptorSet(const nvh::GltfScene& gltf);

  nvh::GltfScene m_gltf;
  nvh::GltfStats m_stats;

  std::string m_sceneName;
  SceneCamera m_camera{};

  // Setup
  nvvk::ResourceAllocator *m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil    m_debug;   // Utility to name objects
  vk::Device         m_device;
  uint32_t           m_queueFamilyIndex;

  // Resources
  std::array<nvvk::Buffer, 5>                              m_buffer;           // For single buffer
  std::array<std::vector<nvvk::Buffer>, 2>                 m_buffers;          // For array of buffers (vertex/index)
  std::vector<nvvk::Texture>                               m_textures;         // vector of all textures of the scene
  std::vector<std::pair<nvvk::Image, vk::ImageCreateInfo>> m_images;           // vector of all images of the scene
  std::vector<size_t>                                      m_defaultTextures;  // for cleanup

  vk::DescriptorPool      m_descPool;
  vk::DescriptorSetLayout m_descSetLayout;
  vk::DescriptorSet       m_descSet;
};

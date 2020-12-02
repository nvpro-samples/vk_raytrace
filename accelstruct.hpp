/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include "vulkan/vulkan.hpp"

#pragma once
#include "nvh/gltfscene.hpp"
#include "nvvk/allocator_vk.hpp"
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
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::Allocator* allocator);
  void destroy();
  void create(nvh::GltfScene& gltfScene, vk::Buffer vertex, vk::Buffer index);

  vk::AccelerationStructureKHR getTlas() { return m_rtBuilder.getAccelerationStructure(); }
  vk::DescriptorSetLayout      getDescLayout() { return m_rtDescSetLayout; }
  vk::DescriptorSet            getDescSet() { return m_rtDescSet; }

private:
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::GltfPrimMesh& prim);
  void                                  createBottomLevelAS(nvh::GltfScene& gltfScene);
  void                                  createTopLevelAS(nvh::GltfScene& gltfScene);
  void                                  createRtDescriptorSet();


  // Setup
  nvvk::Allocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil  m_debug;   // Utility to name objects
  vk::Device       m_device;
  uint32_t         m_queueIndex;

  nvvk::RaytracingBuilderKHR m_rtBuilder;
  vk::Buffer                 m_vertexBuffer;
  vk::Buffer                 m_indexBuffer;


  vk::DescriptorPool      m_rtDescPool;
  vk::DescriptorSetLayout m_rtDescSetLayout;
  vk::DescriptorSet       m_rtDescSet;
};

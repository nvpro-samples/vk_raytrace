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
 *	The Acceleration structure class will holds the scene made of BLASes an TLASes.
 * - It expect a scene in a format of GltfScene  
 * - Each glTF primitive mesh will be in a separate BLAS
 * - All BLASes are using one single Hit shader
 * - It creates a descriptorSet holding the TLAS
 * 
 */


#include "accelstruct.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "shaders/host_device.h"
#include "tools.hpp"

#include <sstream>
#include <ios>

void AccelStructure::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);
  m_rtBuilder.setup(m_device, allocator, familyIndex);
}

void AccelStructure::destroy()
{
  m_rtBuilder.destroy();
  vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);
}

void AccelStructure::create(nvh::GltfScene& gltfScene, const std::vector<nvvk::Buffer>& vertex, const std::vector<nvvk::Buffer>& index)
{
  MilliTimer timer;
  LOGI("Create acceleration structure \n");
  destroy();  // reset

  createBottomLevelAS(gltfScene, vertex, index);
  createTopLevelAS(gltfScene);
  createRtDescriptorSet();
  timer.print();
}


//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
nvvk::RaytracingBuilderKHR::BlasInput AccelStructure::primitiveToGeometry(const nvh::GltfPrimMesh& prim, VkBuffer vertex, VkBuffer index)
{
  // Building part
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer                   = vertex;
  VkDeviceAddress vertexAddress = vkGetBufferDeviceAddress(m_device, &info);
  info.buffer                   = index;
  VkDeviceAddress indexAddress  = vkGetBufferDeviceAddress(m_device, &info);

  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(VertexAttributes);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = indexAddress;
  triangles.maxVertex                = prim.vertexCount;
  //triangles.transformData = ({});

  // Setting up the build info of the acceleration
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;  // For AnyHit
  asGeom.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset;
  offset.firstVertex     = 0;
  offset.primitiveCount  = prim.indexCount / 3;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

//--------------------------------------------------------------------------------------------------
//
//
void AccelStructure::createBottomLevelAS(nvh::GltfScene&                  gltfScene,
                                         const std::vector<nvvk::Buffer>& vertex,
                                         const std::vector<nvvk::Buffer>& index)
{
  // BLAS - Storing each primitive in a geometry
  uint32_t                                           prim_idx{0};
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(gltfScene.m_primMeshes.size());
  for(nvh::GltfPrimMesh& primMesh : gltfScene.m_primMeshes)
  {
    auto geo = primitiveToGeometry(primMesh, vertex[prim_idx].buffer, index[prim_idx].buffer);
    allBlas.push_back({geo});
    prim_idx++;
  }
  LOGI(" BLAS(%d)", allBlas.size());
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                     | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
//
//
void AccelStructure::createTopLevelAS(nvh::GltfScene& gltfScene)
{
  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  tlas.reserve(gltfScene.m_nodes.size());

  for(auto& node : gltfScene.m_nodes)
  {
    // Flags
    VkGeometryInstanceFlagsKHR flags{};
    nvh::GltfPrimMesh&         primMesh = gltfScene.m_primMeshes[node.primMesh];
    nvh::GltfMaterial&         mat      = gltfScene.m_materials[primMesh.materialIndex];

    // Always opaque, no need to use anyhit (faster)
    if(mat.alphaMode == 0 || (mat.baseColorFactor.w == 1.0f && mat.baseColorTexture == -1))
      flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    // Need to skip the cull flag in traceray_rtx for double sided materials
    if(mat.doubleSided == 1)
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(node.worldMatrix);
    rayInst.instanceCustomIndex            = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.primMesh);
    rayInst.flags                          = flags;
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    rayInst.mask                                   = 0xFF;
    tlas.emplace_back(rayInst);
  }
  LOGI(" TLAS(%d)", tlas.size());
  m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// Descriptor set holding the TLAS
//
void AccelStructure::createRtDescriptorSet()
{
  VkShaderStageFlags flags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                             | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  nvvk::DescriptorSetBindings bind;
  bind.addBinding({AccelBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, flags});  // TLAS

  m_rtDescPool = bind.createPool(m_device);
  CREATE_NAMED_VK(m_rtDescSetLayout, bind.createLayout(m_device));
  CREATE_NAMED_VK(m_rtDescSet, nvvk::allocateDescriptorSet(m_device, m_rtDescPool, m_rtDescSetLayout));


  VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();

  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_rtDescSet, AccelBindings::eTlas, &descASInfo));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

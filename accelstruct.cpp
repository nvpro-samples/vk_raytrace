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


#include "accelstruct.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "structures.h"
#include "tools.hpp"


/*
 *	The Acceleration structure class will holds the scene made of BLASes an TLASes.
 * - It expect a scene in a format of GltfScene  
 * - Each glTF primitive mesh will be in a separate BLAS
 * - All BLASes are using one single Hit shader, the values of the material as to be
 *   done in the shader.
 * - It creates a descriptorSet holding the TLAS
 * 
 */


void AccelStructure::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
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
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
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
nvvk::RaytracingBuilderKHR::BlasInput AccelStructure::primitiveToGeometry(const nvh::GltfPrimMesh& prim, vk::Buffer vertex, vk::Buffer index)
{
  // Building part
  vk::DeviceAddress vertexAddress = m_device.getBufferAddress({vertex});
  vk::DeviceAddress indexAddress  = m_device.getBufferAddress({index});

  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setVertexData(vertexAddress);
  triangles.setVertexStride(sizeof(VertexAttributes));
  triangles.setIndexType(vk::IndexType::eUint32);
  triangles.setIndexData(indexAddress);
  triangles.setTransformData({});
  triangles.setMaxVertex(prim.vertexCount);

  // Setting up the build info of the acceleration
  vk::AccelerationStructureGeometryKHR asGeom;
  asGeom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  asGeom.setFlags(vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation);  // For AnyHit
  asGeom.geometry.setTriangles(triangles);

  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(0);
  offset.setPrimitiveCount(prim.indexCount / 3);
  offset.setPrimitiveOffset(0);
  offset.setTransformOffset(0);

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
  m_rtBuilder.buildBlas(allBlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                                     | vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction);
}

//--------------------------------------------------------------------------------------------------
//
//
void AccelStructure::createTopLevelAS(nvh::GltfScene& gltfScene)
{
  std::vector<nvvk::RaytracingBuilderKHR::Instance> tlas;
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

    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform        = node.worldMatrix;
    rayInst.instanceCustomId = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
    rayInst.blasId           = node.primMesh;
    rayInst.flags            = flags;
    rayInst.hitGroupId       = 0;  // We will use the same hit group for all objects
    tlas.emplace_back(rayInst);
  }
  LOGI(" TLAS(%d)", tlas.size());
  m_rtBuilder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// Descriptor set holding the TLAS
//
void AccelStructure::createRtDescriptorSet()
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  nvvk::DescriptorSetBindings bind;
  bind.addBinding(vkDSLB(0, vkDT::eAccelerationStructureKHR, 1,
                         vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eCompute | vkSS::eFragment));  // TLAS

  m_rtDescPool      = bind.createPool(m_device);
  m_rtDescSetLayout = bind.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&tlas);

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_rtDescSet, 0, &descASInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

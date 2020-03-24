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

//--------------------------------------------------------------------------------------------------
// This example is loading a glTF scene and raytrace it with a very simple material
//
#include <iostream>

#include <vulkan/vulkan.hpp>

#include "example.hpp"
#include <nvvk/profiler_vk.hpp>
#include <nvvkpp/descriptorsets_vkpp.hpp>
#include <nvvkpp/pipeline_vkpp.hpp>
#include <nvvkpp/renderpass_vkpp.hpp>
#include <nvvkpp/utilities_vkpp.hpp>

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include <fileformats/tiny_gltf.h>

#include "imgui_impl_glfw.h"
#include "nvh/fileoperations.hpp"
#include "nvvkpp/commands_vkpp.hpp"
#include "shaders/binding.h"
#include <imgui/imgui_orient.h>

extern std::vector<std::string> defaultSearchPaths;

nvvk::ProfilerVK g_profiler;
struct Stats
{
  double loadTime{0};
  double sceneTime{0};
  double recordTime{0};
} s_stats;

using vkDT = vk::DescriptorType;
using vkSS = vk::ShaderStageFlagBits;
using vkCB = vk::CommandBufferUsageFlagBits;
using vkBU = vk::BufferUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;
using vkIU = vk::ImageUsageFlagBits;
using vkDS = vk::DescriptorSetLayoutBinding;

//--------------------------------------------------------------------------------------------------
// Overridden function that is called after the base class create()
//
void VkRtExample::initExample()
{
  g_profiler.init(m_device, m_physicalDevice);
  g_profiler.setAveragingSize(5);

  // Loading the glTF file, it will allocate 3 buffers: vertex, index and matrices
  tinygltf::Model    gltfModel;
  tinygltf::TinyGLTF gltfContext;
  std::string        warn, error;
  bool               fileLoaded = false;
  {
    s_stats.loadTime = -g_profiler.getMicroSeconds();
    fileLoaded       = gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warn, m_filename);
    if(!warn.empty())
    {
      LOGE("Warning loading %s: %s", m_filename.c_str(), warn.c_str());
    }
    if(!error.empty())
    {
      LOGE("Error loading %s: %s", m_filename.c_str(), error.c_str());
    }
    assert(fileLoaded && error.empty() && error.c_str());
    s_stats.loadTime += g_profiler.getMicroSeconds();
  }

  s_stats.sceneTime = -g_profiler.getMicroSeconds();

  // From tinyGLTF to our glTF representation
  {
    m_gltfScene.getMaterials(gltfModel);
    m_vertices.attributes["NORMAL"]     = {0, 1, 0};  // Attributes we are interested in
    m_vertices.attributes["TEXCOORD_0"] = {0, 0};
    m_gltfScene.loadMeshes(gltfModel, m_indices, m_vertices);
    m_gltfScene.loadNodes(gltfModel);
    m_gltfScene.computeSceneDimensions();
    createEmptyTexture();
    loadImages(gltfModel);
  }

  // Set the camera as to see the model
  fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max);

  m_skydome.loadEnvironment(m_hdrFilename);

  s_stats.sceneTime += g_profiler.getMicroSeconds();


  // Lights
  m_sceneUbo.nbLights           = 1;
  m_sceneUbo.lights[0].position = nvmath::vec4f(150, 80, -150, 1);
  m_sceneUbo.lights[0].color    = nvmath::vec4f(1, 1, 1, 10000);
  m_sceneUbo.lights[1].position = nvmath::vec4f(-10, 15, 10, 1);
  m_sceneUbo.lights[1].color    = nvmath::vec4f(1, 1, 1, 1000);

  LOGI("prepareUniformBuffers\n");
  prepareUniformBuffers();
  LOGI("createDescriptorFinal\n");
  createDescriptorFinal();
  LOGI("createDescriptorMaterial\n");
  createDescriptorMaterial();
  LOGI("createPipeline\n");
  createPipeline();  // How the quad will be rendered

  // Raytracing
  {
    std::vector<uint32_t>                            blassOffset;
    std::vector<std::vector<vk::GeometryNV>>         blass;
    std::vector<nvvkpp::RaytracingBuilder::Instance> rayInst;
    m_primitiveOffsets.reserve(m_gltfScene.m_linearNodes.size());

    // BLAS - Storing each primitive in a geometry
    LOGI("Preparing geometry for Blas\n");
    uint32_t blassID = 0;
    for(auto& mesh : m_gltfScene.m_linearMeshes)
    {
      blassOffset.push_back(blassID);  // use by the TLAS to find the BLASS ID from the mesh ID

      for(auto& primitive : mesh->m_primitives)
      {
        ++blassID;
        auto geo = primitiveToGeometry(primitive);
        blass.push_back({geo});
      }
    }

    // TLASS - Top level for each valid mesh
    LOGI("Preparing Tlas\n");
    uint32_t instID = 0;
    for(auto& node : m_gltfScene.m_linearNodes)
    {
      if(node->m_mesh != ~0u)
      {
        nvvkpp::RaytracingBuilder::Instance inst;
        inst.transform = node->worldMatrix();

        // Same transform for each primitive of the mesh
        int primID = 0;
        for(auto& primitive : m_gltfScene.m_linearMeshes[node->m_mesh]->m_primitives)
        {
          inst.instanceId = uint32_t(instID++);  // gl_InstanceID
          inst.blasId     = blassOffset[node->m_mesh] + primID++;
          rayInst.emplace_back(inst);
          // The following is use to find the geometry information in the CHIT
          m_primitiveOffsets.push_back({primitive.m_firstIndex, primitive.m_vertexOffset, primitive.m_materialIndex});
        }
      }
    }

    // Uploading the geometry information
    {
      nvvkpp::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
      m_primitiveInfoBuffer = m_alloc.createBuffer(cmdBuf, m_primitiveOffsets, vk::BufferUsageFlagBits::eStorageBuffer);
      m_debug.setObjectName(m_primitiveInfoBuffer.buffer, "PrimitiveInfo");
    }
    m_alloc.flushStaging();

    LOGI("Creating Blas\n");
    m_raytracer.builder().buildBlas(blass);
    LOGI("Creating Tlas\n");
    m_raytracer.builder().buildTlas(rayInst);
    m_raytracer.createOutputImage(m_size);
    LOGI("Creating Descriptor\n");
    createDescriptorRaytrace();
    m_raytracer.createDescriptorSet();
    LOGI("Creating raytrace pipeline\n");
    m_raytracer.createPipeline(m_descSetLayout[eRaytrace], m_descSetLayout[eMaterial]);
    LOGI("Creating raytrace SBT\n");
    m_raytracer.createShadingBindingTable();
  }

  // Using -SPACE- to pick an object
  vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  m_rayPicker.initialize(m_raytracer.builder().getAccelerationStructure(), sceneDesc);
  // Post-process tonemapper
  m_tonemapper.initialize(m_size);

  // Using the output of the tonemapper to display
  updateDescriptor(m_tonemapper.getOutput().descriptor);

  // Other elements
  m_axis.init(m_device, m_renderPass, 0, 40.f);
}


//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLASS
//
vk::GeometryNV VkRtExample::primitiveToGeometry(const nvh::gltf::Primitive& prim)
{
  vk::GeometryTrianglesNV triangles;
  triangles.setVertexData(m_vertexBuffer.buffer);
  triangles.setVertexOffset(prim.m_vertexOffset * sizeof(nvmath::vec3f));
  triangles.setVertexCount(/*prim.m_indexCount);*/ static_cast<uint32_t>(m_vertices.position.size()));
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);  // 3xfloat32 for vertices
  triangles.setIndexData(m_indexBuffer.buffer);
  triangles.setIndexOffset(prim.m_firstIndex * sizeof(uint32_t));
  triangles.setIndexCount(prim.m_indexCount);
  triangles.setIndexType(vk::IndexType::eUint32);  // 32-bit indices
  vk::GeometryDataNV geoData;
  geoData.setTriangles(triangles);
  vk::GeometryNV geometry;
  geometry.setGeometry(geoData);
  geometry.setFlags(vk::GeometryFlagBitsNV::eNoDuplicateAnyHitInvocation);
  return geometry;
}


//--------------------------------------------------------------------------------------------------
// Overridden function called on shutdown
//
void VkRtExample::destroy()
{
  m_device.waitIdle();

  m_gltfScene.destroy();
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_matrixBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_primitiveInfoBuffer);
  m_alloc.destroy(m_sceneBuffer);

  g_profiler.deinit();

  m_device.destroyRenderPass(m_renderPassUI);
  m_renderPassUI = vk::RenderPass();


  m_device.destroyPipeline(m_pipeline);
  m_device.destroyPipelineLayout(m_pipelineLayout);
  for(int i = 0; i < Dset::Total; i++)
  {
    m_device.destroyDescriptorSetLayout(m_descSetLayout[i]);
    m_device.destroyDescriptorPool(m_descPool[i]);
  }
  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }
  m_alloc.destroy(m_emptyTexture[0]);
  m_alloc.destroy(m_emptyTexture[1]);

  m_rayPicker.destroy();
  m_axis.destroy();
  m_skydome.destroy();
  m_tonemapper.destroy();
  m_raytracer.destroy();

  m_dmaAllocator.deinit();


  AppBase::destroy();
}


//--------------------------------------------------------------------------------
// Called at each frame, as fast as possible
//
void VkRtExample::display()
{
  updateFrame();

  g_profiler.beginFrame();
  drawUI();

  // render the scene
  prepareFrame();
  const vk::CommandBuffer& cmdBuf = m_commandBuffers[m_curFramebuffer];
  cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  // Updating the matrices of the camera
  updateCameraBuffer(cmdBuf);

  vk::ClearValue clearValues[2];
  clearValues[0].setColor(nvvkpp::util::clearColor({0.1f, 0.1f, 0.4f, 0.f}));
  clearValues[1].setDepthStencil({1.0f, 0});

  {
    auto scope = g_profiler.timeRecurring("frame", cmdBuf);

    // Raytracing
    if(m_frameNumber < m_raytracer.maxFrames())
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "raytracing");
      m_raytracer.run(cmdBuf, m_descSet[eRaytrace], m_descSet[eMaterial], m_frameNumber);
    }

    // Apply tonemapper, its output is set in the descriptor set
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "tonemapping");
      m_tonemapper.setInput(m_raytracer.outputImage().descriptor);
      m_tonemapper.run(cmdBuf);
    }

    // Drawing a quad (pass through + final.frag)
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "display");

      vk::RenderPassBeginInfo renderPassBeginInfo = {m_renderPass, m_framebuffers[m_curFramebuffer], {{}, m_size}, 2, clearValues};
      cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
      setViewport(cmdBuf);

      cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
      cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, m_descSet[Dset::eFinal], {});

      cmdBuf.draw(3, 1, 0, 0);
    }

    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "ImGui");

      // Drawing GUI
      ImGui::Render();
      ImDrawData* imguiDrawData = ImGui::GetDrawData();
      ImGui::RenderDrawDataVK(cmdBuf, imguiDrawData);
      ImGui::EndFrame();
    }

    // Rendering axis in same render pass
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "Axes");

      m_axis.display(cmdBuf, CameraManip.getMatrix(), m_size);
    }
    cmdBuf.endRenderPass();
  }


  // End of the frame and present the one which is ready
  cmdBuf.end();
  submitFrame();

  g_profiler.endFrame();
}


//--------------------------------------------------------------------------------------------------
// Return the current frame number
// Check if the camera matrix has changed, if yes, then reset the frame to 0
// otherwise, increment
//
void VkRtExample::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         fov = 0;

  auto& m = CameraManip.getMatrix();
  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || fov != CameraManip.getFov())
  {
    resetFrame();
    refCamMatrix = m;
    fov          = CameraManip.getFov();
  }
  m_frameNumber++;
}

void VkRtExample::resetFrame()
{
  m_frameNumber = -1;
}

//--------------------------------------------------------------------------------------------------
// Creating the Uniform Buffers, only for the scene camera matrices
// The one holding all all matrices of the scene nodes was created in glTF.load()
//
void VkRtExample::prepareUniformBuffers()
{
  {
    nvvkpp::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    m_sceneBuffer = m_alloc.createBuffer(cmdBuf, sizeof(SceneUBO), nullptr, vkBU::eUniformBuffer);

    // Creating the GPU buffer of the vertices
    m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_vertices.position, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_vertices.attributes["NORMAL"], vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_uvBuffer = m_alloc.createBuffer(cmdBuf, m_vertices.attributes["TEXCOORD_0"], vkBU::eVertexBuffer | vkBU::eStorageBuffer);

    // Creating the GPU buffer of the indices
    m_indexBuffer = m_alloc.createBuffer(cmdBuf, m_indices, vkBU::eIndexBuffer | vkBU::eStorageBuffer);

    // Adding all node matrices of the scene in a single buffer (mesh primitives are duplicated)
    std::vector<nvh::gltf::NodeMatrices> allMatrices;
    allMatrices.reserve(m_gltfScene.m_linearNodes.size());
    for(auto& node : m_gltfScene.m_linearNodes)
    {
      if(node->m_mesh != ~0u)
      {
        nvh::gltf::NodeMatrices nm;
        nm.world   = node->worldMatrix();
        nm.worldIT = nm.world;
        nm.worldIT = nvmath::transpose(nvmath::invert(nm.worldIT));
        auto& mesh = m_gltfScene.m_linearMeshes[node->m_mesh];
        for(size_t i = 0; i < mesh->m_primitives.size(); ++i)
        {
          allMatrices.push_back(nm);
        }
      }
    }
    m_matrixBuffer = m_alloc.createBuffer(cmdBuf, allMatrices, vkBU::eStorageBuffer);


    // Materials - Storing all material colors and information
    std::vector<nvh::gltf::Material::PushC> allMaterials;
    allMaterials.reserve(m_gltfScene.m_materials.size());
    for(const auto& mat : m_gltfScene.m_materials)
    {
      allMaterials.push_back(mat.m_mat);
    }
    m_materialBuffer = m_alloc.createBuffer(cmdBuf, allMaterials, vkBU::eStorageBuffer);
  }
  m_alloc.flushStaging();

  m_debug.setObjectName(m_sceneBuffer.buffer, "SceneUbo");
  m_debug.setObjectName(m_vertexBuffer.buffer, "Vertex");
  m_debug.setObjectName(m_indexBuffer.buffer, "Index");
  m_debug.setObjectName(m_uvBuffer.buffer, "UV");
  m_debug.setObjectName(m_normalBuffer.buffer, "Normal");
  m_debug.setObjectName(m_matrixBuffer.buffer, "Matrix");
  m_debug.setObjectName(m_materialBuffer.buffer, "Material");
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void VkRtExample::createPipeline()
{
  std::vector<std::string> paths = defaultSearchPaths;

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_descSetLayout[eFinal]);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  nvvkpp::GraphicsPipelineGenerator pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/final.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_pipeline = pipelineGenerator.create();
}


//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void VkRtExample::createDescriptorFinal()
{
  m_descSetLayoutBind[eFinal].emplace_back(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_descSetLayout[eFinal] = nvvkpp::util::createDescriptorSetLayout(m_device, m_descSetLayoutBind[eFinal]);
  m_descPool[eFinal]      = nvvkpp::util::createDescriptorPool(m_device, m_descSetLayoutBind[eFinal]);
  m_descSet[eFinal]       = nvvkpp::util::createDescriptorSet(m_device, m_descPool[eFinal], m_descSetLayout[eFinal]);
}

//--------------------------------------------------------------------------------------------------
// Create all descriptors for the Material (set 2)
// - to find the textures in the closest hit shader
//
void VkRtExample::createDescriptorMaterial()
{
  uint32_t nbMat = static_cast<uint32_t>(m_gltfScene.m_materials.size());
  uint32_t nbTxt = nbMat;

  m_descSetLayoutBind[eMaterial].emplace_back(
      vkDS(0, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // albedo
  m_descSetLayoutBind[eMaterial].emplace_back(vkDS(1, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitNV));  // normal
  m_descSetLayoutBind[eMaterial].emplace_back(vkDS(2, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitNV));  // occlusion
  m_descSetLayoutBind[eMaterial].emplace_back(vkDS(3, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitNV));  // metallic/roughness
  m_descSetLayoutBind[eMaterial].emplace_back(vkDS(4, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitNV));  // emission

  m_descSetLayout[eMaterial] = nvvkpp::util::createDescriptorSetLayout(m_device, m_descSetLayoutBind[eMaterial]);
  m_descPool[eMaterial]      = nvvkpp::util::createDescriptorPool(m_device, m_descSetLayoutBind[eMaterial], nbMat);
  m_descSet[eMaterial] = nvvkpp::util::createDescriptorSet(m_device, m_descPool[eMaterial], m_descSetLayout[eMaterial]);


  // The layout of the descriptor is done, but we have to allocate a descriptor from this template
  // a set the pBufferInfo to the buffer descriptor that was previously allocated (see prepareUniformBuffers)
  std::vector<vk::WriteDescriptorSet>  writes;
  std::vector<vk::DescriptorImageInfo> idBaseColor(nbMat);
  std::vector<vk::DescriptorImageInfo> idNormal(nbMat);
  std::vector<vk::DescriptorImageInfo> idOcclusion(nbMat);
  std::vector<vk::DescriptorImageInfo> idMetallic(nbMat);
  std::vector<vk::DescriptorImageInfo> idEmissive(nbMat);

  uint32_t matId = 0;
  for(auto& material : m_gltfScene.m_materials)
  {
    idBaseColor[matId] =
        (material.m_baseColorTexture ? m_gltfScene.getDescriptor(material.m_baseColorTexture) : m_emptyTexture[1].descriptor);
    idNormal[matId] =
        (material.m_normalTexture ? m_gltfScene.getDescriptor(material.m_normalTexture) : m_emptyTexture[0].descriptor);
    idOcclusion[matId] =
        (material.m_occlusionTexture ? m_gltfScene.getDescriptor(material.m_occlusionTexture) : m_emptyTexture[1].descriptor);
    idMetallic[matId] = (material.m_metallicRoughnessTexture ? m_gltfScene.getDescriptor(material.m_metallicRoughnessTexture) :
                                                               m_emptyTexture[1].descriptor);
    idEmissive[matId] =
        (material.m_emissiveTexture ? m_gltfScene.getDescriptor(material.m_emissiveTexture) : m_emptyTexture[0].descriptor);

    const auto& descSet = m_descSet[eMaterial];
    writes.emplace_back(descSet, 0, matId, 1, vkDT::eCombinedImageSampler, &idBaseColor[matId]);
    writes.emplace_back(descSet, 1, matId, 1, vkDT::eCombinedImageSampler, &idNormal[matId]);
    writes.emplace_back(descSet, 2, matId, 1, vkDT::eCombinedImageSampler, &idOcclusion[matId]);
    writes.emplace_back(descSet, 3, matId, 1, vkDT::eCombinedImageSampler, &idMetallic[matId]);
    writes.emplace_back(descSet, 4, matId, 1, vkDT::eCombinedImageSampler, &idEmissive[matId]);

    ++matId;
  }

  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creates all descriptors for raytracing (set 1)
//
void VkRtExample::createDescriptorRaytrace()
{
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  auto& bind = m_descSetLayoutBind[eRaytrace];
  bind.emplace_back(vkDSLB(B_SCENE, vkDT::eUniformBuffer, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));  // Scene, camera
  bind.emplace_back(vkDSLB(B_PRIM_INFO, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // Primitive info
  bind.emplace_back(vkDSLB(B_VERTICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));   // Vertices
  bind.emplace_back(vkDSLB(B_INDICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));    // Indices
  bind.emplace_back(vkDSLB(B_NORMALS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                      // Normals
  bind.emplace_back(vkDSLB(B_TEXCOORDS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // UVs
  bind.emplace_back(vkDSLB(B_MATRICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                     // Matrix
  bind.emplace_back(vkDSLB(B_MATERIAL, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));   // material
  bind.emplace_back(vkDSLB(B_HDR, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV | vkSS::eMissNV));   // skydome
  bind.emplace_back(vkDSLB(B_FILTER_DIFFUSE, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV));        // irradiance
  bind.emplace_back(vkDSLB(B_LUT_BRDF, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV));              // lutBrdf
  bind.emplace_back(vkDSLB(B_FILTER_GLOSSY, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV | vkSS::eMissNV));  // prefilterdEnv
  bind.emplace_back(vkDSLB(B_IMPORT_SMPL, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV));  // importance sampling

  m_descPool[eRaytrace]      = nvvkpp::util::createDescriptorPool(m_device, m_descSetLayoutBind[eRaytrace]);
  m_descSetLayout[eRaytrace] = nvvkpp::util::createDescriptorSetLayout(m_device, m_descSetLayoutBind[eRaytrace]);
  m_descSet[eRaytrace] = m_device.allocateDescriptorSets({m_descPool[eRaytrace], 1, &m_descSetLayout[eRaytrace]})[0];


  vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo primitiveInfoDesc{m_primitiveInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo vertexDesc{m_vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo indexDesc{m_indexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo normalDesc{m_normalBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo uvDesc{m_uvBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo matrixDesc{m_matrixBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo materialDesc{m_materialBuffer.buffer, 0, VK_WHOLE_SIZE};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_SCENE], &sceneDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_PRIM_INFO], &primitiveInfoDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_VERTICES], &vertexDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_INDICES], &indexDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_NORMALS], &normalDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_TEXCOORDS], &uvDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_MATRICES], &matrixDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_MATERIAL], &materialDesc));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_HDR], &m_skydome.m_textures.txtHdr.descriptor));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_FILTER_DIFFUSE],
                                                &m_skydome.m_textures.irradianceCube.descriptor));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_LUT_BRDF], &m_skydome.m_textures.lutBrdf.descriptor));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_FILTER_GLOSSY],
                                                &m_skydome.m_textures.prefilteredCube.descriptor));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet[eRaytrace], bind[B_IMPORT_SMPL],
                                                &m_skydome.m_textures.accelImpSmpl.descriptor));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void VkRtExample::updateDescriptor(const vk::DescriptorImageInfo& descriptor)
{
  vk::WriteDescriptorSet writeDescriptorSets =
      nvvkpp::util::createWrite(m_descSet[eFinal], m_descSetLayoutBind[eFinal][0], &descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating an empty texture which is used for when the material as no texture. We cannot pass NULL
//
void VkRtExample::createEmptyTexture()
{
  std::vector<uint8_t> black      = {0, 0, 0, 0};
  std::vector<uint8_t> white      = {255, 255, 255, 255};
  VkDeviceSize         bufferSize = 4;
  vk::Extent2D         imgSize(1, 1);

  {
    nvvkpp::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    vk::SamplerCreateInfo      samplerCreateInfo;  // default values
    vk::ImageCreateInfo        imageCreateInfo = nvvkpp::image::create2DInfo(imgSize);

    m_emptyTexture[0]            = m_alloc.createImage(cmdBuf, bufferSize, black.data(), imageCreateInfo);
    m_emptyTexture[0].descriptor = nvvkpp::image::create2DDescriptor(m_device, m_emptyTexture[0].image, samplerCreateInfo);

    m_emptyTexture[1]            = m_alloc.createImage(cmdBuf, bufferSize, white.data(), imageCreateInfo);
    m_emptyTexture[1].descriptor = nvvkpp::image::create2DDescriptor(m_device, m_emptyTexture[1].image, samplerCreateInfo);

    m_debug.setObjectName(m_emptyTexture[0].image, "Black");
    m_debug.setObjectName(m_emptyTexture[1].image, "White");
  }

  m_alloc.flushStaging();
}

//--------------------------------------------------------------------------------------------------
// When the frames are redone, we also need to re-record the command buffer
//
void VkRtExample::onResize(int w, int h)
{
  m_raytracer.createOutputImage(m_size);
  m_raytracer.updateDescriptorSet();
  m_tonemapper.updateRenderTarget(m_size);
  updateDescriptor(m_tonemapper.getOutput().descriptor);
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Setting which scene to load. Check arguments in main.cpp
//
void VkRtExample::setScene(const std::string& filename)
{
  m_filename = filename;
}

void VkRtExample::setEnvironmentHdr(const std::string& hdrFilename)
{
  m_hdrFilename = hdrFilename;
}


//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void VkRtExample::updateCameraBuffer(const vk::CommandBuffer& cmdBuffer)
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  float       nearPlane   = m_gltfScene.m_dimensions.radius / 10.0f;
  float       farPlane    = m_gltfScene.m_dimensions.radius * 50.0f;

  m_sceneUbo.model      = CameraManip.getMatrix();
  m_sceneUbo.projection = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, nearPlane, farPlane);
  nvmath::vec3f pos, center, up;
  CameraManip.getLookat(pos, center, up);
  m_sceneUbo.cameraPosition = pos;

  cmdBuffer.updateBuffer<VkRtExample::SceneUBO>(m_sceneBuffer.buffer, 0, m_sceneUbo);
}

//--------------------------------------------------------------------------------------------------
// The display will render the recorded command buffer, then in a sub-pass, render the UI
//
void VkRtExample::createRenderPass()
{
  m_renderPass   = nvvkpp::util::createRenderPass(m_device, {m_swapChain.colorFormat}, m_depthFormat, 1, true, true);
  m_renderPassUI = nvvkpp::util::createRenderPass(m_device, {m_swapChain.colorFormat}, m_depthFormat, 1, false, false);
}

//--------------------------------------------------------------------------------------------------
// Overload callback when a key gets hit
// - Pressing 'F' to move the camera to see the scene bounding box
//
void VkRtExample::onKeyboardChar(unsigned char key)
{
  AppBase::onKeyboardChar(key);

  if(key == 'f')
  {
    fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, false);
  }

  if(key == ' ')
  {
  }
}


//--------------------------------------------------------------------------------------------------
// IMGUI UI display
//
void VkRtExample::drawUI()
{
  static int e = m_upVector;

  // Update imgui configuration
  ImGui_ImplGlfw_NewFrame();

  ImGui::NewFrame();
  ImGui::SetNextWindowBgAlpha(0.8);
  ImGui::SetNextWindowSize(ImVec2(450, 0), ImGuiCond_FirstUseEver);

  ImGui::Begin("Hello, Vulkan!", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::Text("%s", m_physicalDevice.getProperties().deviceName);

  if(ImGui::CollapsingHeader("Camera Up Vector"))
  {

    ImGui::RadioButton("X", &e, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Y", &e, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Z", &e, 2);
    if(e != m_upVector)
    {
      nvmath::vec3f eye, center, up;
      CameraManip.getLookat(eye, center, up);
      CameraManip.setLookat(eye, center, nvmath::vec3f(e == 0, e == 1, e == 2));
      m_upVector = e;
    }
  }
  bool modified = false;
  modified |= uiLights();
  modified |= m_raytracer.uiSetup();
  m_tonemapper.uiSetup();

  if(modified)
  {
    resetFrame();
  }

  if(ImGui::CollapsingHeader("Debug"))
  {
    static const char* dbgItem[] = {"None",     "Metallic", "Normal", "Base Color", "Occlusion",
                                    "Emissive", "F0",       "Alpha",  "Roughness"};
    // ImGui::Combo("Debug Mode", &m_sceneUbo.materialMode, dbgItem, 9);
  }

  // Adding performance
  static float  valuesFPS[90] = {0};
  static float  valuesRnd[90] = {0};
  static float  valueMax      = 0;
  static float  valueMSMax    = 0;
  static int    values_offset = 0;
  static int    nbFrames      = 0;
  static double perfTime      = g_profiler.getMicroSeconds();

  nbFrames++;
  double diffTime = g_profiler.getMicroSeconds() - perfTime;
  if(diffTime > 50000)
  {
    double frameCpu, frameGpu;
    g_profiler.getAveragedValues("frame", frameCpu, frameGpu);
    double curTime = perfTime = g_profiler.getMicroSeconds();
    valuesFPS[values_offset]  = nbFrames / diffTime * 1000000.0f;  // to seconds
    valuesRnd[values_offset]  = frameGpu / 1000.0;                 // to milliseconds
    valueMax                  = std::min(std::max(valueMax, valuesFPS[values_offset]), 1000.0f);
    valueMSMax                = std::max(valueMSMax, valuesRnd[values_offset]);
    values_offset             = (values_offset + 1) % IM_ARRAYSIZE(valuesFPS);
    nbFrames                  = 0;
  }

  if(ImGui::CollapsingHeader("Performance"))
  {
    char strbuf[80];
    int  last = (values_offset - 1) % IM_ARRAYSIZE(valuesFPS);
    sprintf(strbuf, "Render\n%3.2fms", valuesRnd[last]);
    ImGui::PlotLines(strbuf, valuesRnd, IM_ARRAYSIZE(valuesFPS), values_offset, nullptr, 0.0f, valueMSMax, ImVec2(0, 80));
    sprintf(strbuf, "FPS\n%3.1f", valuesFPS[last]);
    ImGui::PlotLines(strbuf, valuesFPS, IM_ARRAYSIZE(valuesFPS), values_offset, nullptr, 0.0f, valueMax, ImVec2(0, 80));
    if(ImGui::TreeNode("Extra"))
    {
      ImGui::Text("Scene loading time:     %3.2f ms", s_stats.loadTime);
      ImGui::Text("Scene preparation time: %3.2f ms", s_stats.sceneTime);
      ImGui::Text("Scene recording time:   %3.2f ms", s_stats.recordTime);
      ImGui::TreePop();
    }
  }

  if(ImGui::CollapsingHeader("Statistics"))
  {
    ImGui::Text("Nb instances  : %zu", m_gltfScene.m_linearNodes.size());
    ImGui::Text("Nb meshes     : %zu", m_gltfScene.m_linearMeshes.size());
    ImGui::Text("Nb materials  : %zu", m_gltfScene.m_materials.size());
    ImGui::Text("Nb triangles  : %zu", m_indices.size() / 3);
  }

  AppBase::uiDisplayHelp();

  ImGui::End();
  ImGui::Render();
}

//--------------------------------------------------------------------------------------------------
// UI for lights
//
bool VkRtExample::uiLights()
{
  bool modified = false;
  if(ImGui::CollapsingHeader("Lights"))
  {
    for(int nl = 0; nl < m_sceneUbo.nbLights; nl++)
    {
      ImGui::PushID(nl);
      if(ImGui::TreeNode("##light", "Light %d", nl))
      {
        modified = ImGui::DragFloat3("Position", &m_sceneUbo.lights[nl].position.x) || modified;
        modified = ImGui::InputFloat("Intensity", &m_sceneUbo.lights[nl].color.w) || modified;
        modified = ImGui::ColorEdit3("Color", (float*)&m_sceneUbo.lights[nl].color.x) || modified;
        ImGui::Separator();
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
  }
  return modified;
}


//--------------------------------------------------------------------------------------------------
// Convert all images to textures
//
void VkRtExample::loadImages(tinygltf::Model& gltfModel)
{
  m_textures.resize(gltfModel.images.size());

  vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  vk::Format            format{vk::Format::eR8G8B8A8Unorm};
  samplerCreateInfo.maxLod = FLT_MAX;

  nvvkpp::SingleCommandBuffer sc(m_device, m_graphicsQueueIndex);

  auto cmdBuf = sc.createCommandBuffer();
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto&        gltfimage       = gltfModel.images[i];
    void*        buffer          = &gltfimage.image[0];
    VkDeviceSize bufferSize      = gltfimage.image.size();
    auto         imgSize         = vk::Extent2D(gltfimage.width, gltfimage.height);
    auto         imageCreateInfo = nvvkpp::image::create2DInfo(imgSize, format, vkIU::eSampled, true);

    m_textures[i] = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvkpp::image::generateMipmaps(cmdBuf, m_textures[i].image, format, imgSize, imageCreateInfo.mipLevels);
    m_textures[i].descriptor = nvvkpp::image::create2DDescriptor(m_device, m_textures[i].image, samplerCreateInfo);

    m_gltfScene.m_textureDescriptors[i] = m_textures[i].descriptor;
    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)).c_str());
  }
  sc.flushCommandBuffer(cmdBuf);
  m_alloc.flushStaging();
}

//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
// - Home key: fit all, the camera will move to see the entire scene bounding box
// - Space: Trigger ray picking and set the interest point at the intersection
//          also return all information under the cursor
//
void VkRtExample::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvkpp::AppBase::onKeyboard(key, scancode, action, mods);

  if(key == GLFW_KEY_HOME)
  {
    // Set the camera as to see the model
    fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, false);
  }

  if(key == GLFW_KEY_SPACE && action == 1)
  {
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);

    // Set the camera as to see the model
    nvvkpp::SingleCommandBuffer sc(m_device, m_graphicsQueueIndex);
    vk::CommandBuffer           cmdBuf = sc.createCommandBuffer();
    float                       px     = x / float(m_size.width);
    float                       py     = y / float(m_size.height);
    m_rayPicker.run(cmdBuf, px, py);
    sc.flushCommandBuffer(cmdBuf);

    nvvkpp::RayPicker::PickResult pr = m_rayPicker.getResult();

    if(pr.intanceID == ~0)
    {
      LOGI("Not Hit\n");
      return;
    }

    std::stringstream o;
    LOGI("\n Instance:  %d", pr.intanceID);
    LOGI("\n Primitive: %d", pr.primitiveID);
    LOGI("\n Distance:  %f", nvmath::length(pr.worldPos - m_sceneUbo.cameraPosition));
    uint  indexOffset = m_primitiveOffsets[pr.intanceID].indexOffset + (3 * pr.primitiveID);
    ivec3 ind         = ivec3(m_indices[indexOffset + 0], m_indices[indexOffset + 1], m_indices[indexOffset + 2]);
    LOGI("\n Position: %f, %f, %f \n", pr.worldPos.x, pr.worldPos.y, pr.worldPos.z);

    // Set the interest position
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, pr.worldPos, up, false);
  }
}

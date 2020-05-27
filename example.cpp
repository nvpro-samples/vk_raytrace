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
#include "imgui/imgui_orient.h"
#include "imgui_impl_glfw.h"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "shaders/binding.h"

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "nvmath/nvmath_types.h"
#include <fileformats/tiny_gltf.h>


extern std::vector<std::string> defaultSearchPaths;


void VkRtExample::setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex)
{
  AppBase::setup(instance, device, physicalDevice, graphicsQueueIndex);
  m_debug.setup(device);

#ifdef NVVK_ALLOC_DMA
  m_memAllocator.init(device, physicalDevice);
  m_alloc.init(device, physicalDevice, &m_memAllocator);
#elif defined(NVVK_ALLOC_DEDICATED)
  m_alloc.init(device, physicalDevice);
#endif

  m_raytracer.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_rayPicker.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_tonemapper.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_skydome.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
}

//--------------------------------------------------------------------------------------------------
// Overridden function that is called after the base class create()
//
void VkRtExample::loadScene(const std::string& filename)
{
  // Loading the glTF file, it will allocate 3 buffers: vertex, index and matrices
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;
  bool               fileLoaded = false;
  MilliTimer         timer;
  {
    LOGI("Loading glTF: %s\n", filename.c_str());

    fileLoaded = tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename);
    if(!warn.empty())
    {
      LOGE("Warning loading %s: %s", filename.c_str(), warn.c_str());
    }
    if(!error.empty())
    {
      LOGE("Error loading %s: %s", filename.c_str(), error.c_str());
    }
    assert(fileLoaded && error.empty() && error.c_str());
    LOGI(" --> (%5.3f ms)\n", timer.elapse());
  }

  // From tinyGLTF to our glTF representation
  {
    LOGI("Importing Scene\n");
    m_gltfScene.importMaterials(tmodel);
    m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0
                                                | nvh::GltfAttributes::Tangent);
    m_sceneStats = m_gltfScene.getStatistics(tmodel);
    LOGI(" --> (%5.3f ms)\n", timer.elapse());
  }

  // Set the camera to see the scene
  if(m_gltfScene.m_cameras.empty())
    fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max);
  else
  {
    nvmath::vec3f eye;
    m_gltfScene.m_cameras[0].worldMatrix.get_translation(eye);
    float len = nvmath::length(m_gltfScene.m_dimensions.center - eye);
    CameraManip.setMatrix(m_gltfScene.m_cameras[0].worldMatrix, true, len);
    CameraManip.setFov(rad2deg(m_gltfScene.m_cameras[0].cam.perspective.yfov));
  }


  {
    LOGI("Importing %d images\n", tmodel.images.size());
    importImages(tmodel);
    LOGI(" --> (%5.3f ms)\n", timer.elapse());
  }

  // Lights
  m_sceneUbo.nbLights           = 1;
  m_sceneUbo.lights[0].position = nvmath::vec4f(150, 80, -150, 1);
  m_sceneUbo.lights[0].color    = nvmath::vec4f(1, 1, 1, 10000);

  // Create buffers with all scene information: vertex, normal, material, ...
  createSceneBuffers();
  createSceneDescriptors();

  // Using the output of the tonemapper to display
  updateDescriptor(m_tonemapper.getOutput().descriptor);

  // Raytracing
  {
    LOGI("Creating BLAS and TLAS\n");

    // BLAS - Storing each primitive in a geometry
    std::vector<std::vector<VkGeometryNV>> blass;
    std::vector<RtPrimitiveLookup>         primLookup;
    for(auto& primMesh : m_gltfScene.m_primMeshes)
    {
      auto geo = primitiveToGeometry(primMesh);
      blass.push_back({geo});

      // The following is use to find the primitive mesh information in the CHIT
      primLookup.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
    }
    m_raytracer.builder().buildBlas(blass);
    m_raytracer.setPrimitiveLookup(primLookup);

    // TLAS - Top level for each valid mesh
    std::vector<nvvk::RaytracingBuilderNV::Instance> rayInst;
    uint32_t                                         instID = 0;
    for(auto& node : m_gltfScene.m_nodes)
    {
      nvvk::RaytracingBuilderNV::Instance inst;
      inst.transform  = node.worldMatrix;
      inst.instanceId = node.primMesh;  // gl_InstanceCustomIndexNV
      inst.blasId     = node.primMesh;
      rayInst.emplace_back(inst);

      auto& mesh = m_gltfScene.m_primMeshes[node.primMesh];
    }
    m_raytracer.builder().buildTlas(rayInst);

    m_raytracer.createOutputImage(m_size);
    m_raytracer.createDescriptorSet();
    m_raytracer.createPipeline(m_descSetLayout[eScene]);
    m_raytracer.createShadingBindingTable();

    m_alloc.finalizeAndReleaseStaging();
    LOGI(" --> (%5.3f ms)\n", timer.elapse());
  }

  // Using -SPACE- to pick an object
  vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  m_rayPicker.initialize(m_raytracer.builder().getAccelerationStructure(), sceneDesc);
}


//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive to VkGeometryNV used to create a BLAS
//
vk::GeometryNV VkRtExample::primitiveToGeometry(const nvh::GltfPrimMesh& prim)
{
  vk::GeometryTrianglesNV triangles;
  triangles.setVertexData(m_vertexBuffer.buffer);
  triangles.setVertexOffset(prim.vertexOffset * sizeof(nvmath::vec3f));
  triangles.setVertexCount(prim.vertexCount);
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setIndexData(m_indexBuffer.buffer);
  triangles.setIndexOffset(prim.firstIndex * sizeof(uint32_t));
  triangles.setIndexCount(prim.indexCount);
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
  m_alloc.destroy(m_tangentBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_sceneBuffer);

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

  m_rayPicker.destroy();
  m_axis.deinit();
  m_skydome.destroy();
  m_tonemapper.destroy();
  m_raytracer.destroy();

  m_alloc.deinit();
#ifdef NVVK_ALLOC_DMA
  m_memAllocator.deinit();
#endif  // NVVK_ALLOC_DMA


  AppBase::destroy();
}


//--------------------------------------------------------------------------------
// Called at each frame, as fast as possible
//
void VkRtExample::display()
{
  updateFrame();
  drawUI();

  // render the scene
  prepareFrame();
  const vk::CommandBuffer& cmdBuf = m_commandBuffers[getCurFrame()];
  cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  // Updating the matrices of the camera
  updateCameraBuffer(cmdBuf);

  vk::ClearValue clearValues[2];
  clearValues[0].setColor(std::array<float, 4>({0.1f, 0.1f, 0.4f, 0.f}));
  clearValues[1].setDepthStencil({1.0f, 0});

  {
    // Raytracing
    if(m_frameNumber < m_raytracer.maxFrames())
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "raytracing");
      m_raytracer.run(cmdBuf, m_descSet[eScene], m_frameNumber);
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

      vk::RenderPassBeginInfo renderPassBeginInfo = {m_renderPass, m_framebuffers[getCurFrame()], {{}, m_size}, 2, clearValues};
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
void VkRtExample::createSceneBuffers()
{
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    m_sceneBuffer = m_alloc.createBuffer(cmdBuf, sizeof(SceneUBO), nullptr, vkBU::eUniformBuffer);

    // Creating the GPU buffer of the vertices
    m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_uvBuffer     = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_tangentBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_tangents, vkBU::eVertexBuffer | vkBU::eStorageBuffer);

    // Creating the GPU buffer of the indices
    m_indexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices, vkBU::eIndexBuffer | vkBU::eStorageBuffer);

    // Materials - Storing all material colors and information
    m_materialBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_materials, vkBU::eStorageBuffer);
  }
  m_alloc.finalizeAndReleaseStaging();

  m_debug.setObjectName(m_sceneBuffer.buffer, "SceneUbo");
  m_debug.setObjectName(m_vertexBuffer.buffer, "Vertex");
  m_debug.setObjectName(m_indexBuffer.buffer, "Index");
  m_debug.setObjectName(m_uvBuffer.buffer, "UV");
  m_debug.setObjectName(m_tangentBuffer.buffer, "Tangent");
  m_debug.setObjectName(m_normalBuffer.buffer, "Normal");
  m_debug.setObjectName(m_materialBuffer.buffer, "Material");
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
// This one is for displaying the ray traced image on a quad
//
void VkRtExample::createFinalPipeline()
{
  std::vector<std::string> paths = defaultSearchPaths;

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_descSetLayout[eFinal]);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/final.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_pipeline = pipelineGenerator.createPipeline();
}


//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void VkRtExample::createDescriptorFinal()
{
  m_descSetLayoutBind[eFinal].clear();
  m_descSetLayoutBind[eFinal].addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_descSetLayout[eFinal] = m_descSetLayoutBind[eFinal].createLayout(m_device);
  m_descPool[eFinal]      = m_descSetLayoutBind[eFinal].createPool(m_device);
  m_descSet[eFinal]       = nvvk::allocateDescriptorSet(m_device, m_descPool[eFinal], m_descSetLayout[eFinal]);
}

//--------------------------------------------------------------------------------------------------
// Creates all descriptors for raytracing (set 1)
//
void VkRtExample::createSceneDescriptors()
{
  using vkDSLB = vk::DescriptorSetLayoutBinding;
  m_descSetLayoutBind[eScene].clear();

  auto& bind = m_descSetLayoutBind[eScene];
  bind.addBinding(vkDSLB(B_SCENE, vkDT::eUniformBuffer, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));     // Scene, camera
  bind.addBinding(vkDSLB(B_VERTICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // Vertices
  bind.addBinding(vkDSLB(B_INDICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));   // Indices
  bind.addBinding(vkDSLB(B_NORMALS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                     // Normals
  bind.addBinding(vkDSLB(B_TEXCOORDS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // UVs
  bind.addBinding(vkDSLB(B_TANGENTS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));   // Tangents
  bind.addBinding(vkDSLB(B_MATERIAL, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));   // material
  bind.addBinding(vkDSLB(B_HDR, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV | vkSS::eMissNV));   // skydome
  bind.addBinding(vkDSLB(B_FILTER_DIFFUSE, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV));        // irradiance
  bind.addBinding(vkDSLB(B_LUT_BRDF, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV));              // lutBrdf
  bind.addBinding(vkDSLB(B_FILTER_GLOSSY, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV | vkSS::eMissNV));  // prefilterdEnv
  bind.addBinding(vkDSLB(B_IMPORT_SMPL, vkDT::eCombinedImageSampler, 1, vkSS::eClosestHitNV));  // importance sampling
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  bind.addBinding(vkDSLB(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures,
                         vkSS::eFragment | vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // all textures


  m_descPool[eScene]      = m_descSetLayoutBind[eScene].createPool(m_device);
  m_descSetLayout[eScene] = m_descSetLayoutBind[eScene].createLayout(m_device);
  m_descSet[eScene]       = m_device.allocateDescriptorSets({m_descPool[eScene], 1, &m_descSetLayout[eScene]})[0];


  vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo vertexDesc{m_vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo indexDesc{m_indexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo normalDesc{m_normalBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo uvDesc{m_uvBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo tangentDesc{m_tangentBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo materialDesc{m_materialBuffer.buffer, 0, VK_WHOLE_SIZE};

  std::vector<vk::DescriptorImageInfo> dbiImages;
  for(const auto& imageDesc : m_textures)
    dbiImages.emplace_back(imageDesc.descriptor);


  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_SCENE, &sceneDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_VERTICES, &vertexDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_INDICES, &indexDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_NORMALS, &normalDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_TEXCOORDS, &uvDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_TANGENTS, &tangentDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_MATERIAL, &materialDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_HDR, &m_skydome.m_textures.txtHdr.descriptor));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_FILTER_DIFFUSE, &m_skydome.m_textures.irradianceCube.descriptor));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_LUT_BRDF, &m_skydome.m_textures.lutBrdf.descriptor));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_FILTER_GLOSSY, &m_skydome.m_textures.prefilteredCube.descriptor));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_IMPORT_SMPL, &m_skydome.m_textures.accelImpSmpl.descriptor));

  for(int i = 0; i < dbiImages.size(); i++)
    writes.emplace_back(m_descSet[eScene], B_TEXTURES, i, 1, vk::DescriptorType::eCombinedImageSampler, &dbiImages[i]);

  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void VkRtExample::updateDescriptor(const vk::DescriptorImageInfo& descriptor)
{
  vk::WriteDescriptorSet writeDescriptorSets = m_descSetLayoutBind[eFinal].makeWrite(m_descSet[eFinal], 0, &descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
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
//
//
void VkRtExample::loadEnvironmentHdr(const std::string& hdrFilename)
{
  MilliTimer timer;
  LOGI("Loading HDR and converting %s\n", hdrFilename.c_str());
  m_skydome.loadEnvironment(hdrFilename);
  LOGI(" --> (%5.3f ms)\n", timer.elapse());

  m_raytracer.m_pushC.fireflyClampThreshold = m_skydome.getIntegral() * 4.f;  //magic
}


void VkRtExample::createTonemapper()
{
  m_tonemapper.initialize(m_size);
}

void VkRtExample::createAxis()
{
  m_axis.init(m_device, m_renderPass, 0, 40.f);
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
  m_renderPass   = nvvk::createRenderPass(m_device, {getColorFormat()}, m_depthFormat, 1, true, true);
  m_renderPassUI = nvvk::createRenderPass(m_device, {getColorFormat()}, m_depthFormat, 1, false, false);
}

// Formating with local number representation (1,000,000.23 or 1.000.000,23)
template <class T>
std::string FormatNumbers(T value)
{
  std::stringstream ss;
  ss.imbue(std::locale(""));
  ss << std::fixed << value;
  return ss.str();
}

//--------------------------------------------------------------------------------------------------
// IMGUI UI display
//
void VkRtExample::drawUI()
{
  // Update imgui configuration
  ImGui_ImplGlfw_NewFrame();

  ImGui::NewFrame();
  ImGui::SetNextWindowBgAlpha(0.8);
  ImGui::SetNextWindowSize(ImVec2(450, 0), ImGuiCond_FirstUseEver);

  ImGui::Begin("Ray Tracing Example", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::Text("%s", &m_physicalDevice.getProperties().deviceName[0]);

  bool changed{false};

  if(ImGui::CollapsingHeader("Camera"))
  {
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    changed |= ImGui::DragFloat3("Position", &eye.x);
    changed |= ImGui::DragFloat3("Center", &center.x);
    changed |= ImGui::DragFloat3("Up", &up.x, .1f, 0.0f, 1.0f);
    float fov(CameraManip.getFov());
    if(ImGui::SliderFloat("FOV", &fov, 1, 150))
      CameraManip.setFov(fov);
    if(changed)
      CameraManip.setLookat(eye, center, up);
  }

  //  modified |= uiLights();
  changed |= m_raytracer.uiSetup();
  m_tonemapper.uiSetup();

  if(ImGui::CollapsingHeader("Debug"))
  {
    static const char* dbgItem[] = {"None", "Metallic", "Normal",    "Base Color", "Occlusion", "Emissive",
                                    "F0",   "Alpha",    "Roughness", "UV",         "Tangent"};
    changed |= ImGui::Combo("Debug Mode", &m_sceneUbo.debugMode, dbgItem, 11);
  }

  if(changed)
  {
    resetFrame();
  }

  if(ImGui::CollapsingHeader("Statistics"))
  {
    ImGui::Text("Camera      : %d", m_sceneStats.nbCameras);
    ImGui::Text("Images      : %d", m_sceneStats.nbImages);
    ImGui::Text("Textures    : %d", m_sceneStats.nbTextures);
    ImGui::Text("Materials   : %d", m_sceneStats.nbMaterials);
    ImGui::Text("Samplers    : %d", m_sceneStats.nbSamplers);
    ImGui::Text("Nodes       : %d", m_sceneStats.nbNodes);
    ImGui::Text("Meshes      : %d", m_sceneStats.nbMeshes);
    ImGui::Text("Lights      : %d", m_sceneStats.nbLights);
    ImGui::Text("Unique Tri  : %s", FormatNumbers(m_sceneStats.nbUniqueTriangles).c_str());
    ImGui::Text("Total Tri   : %s", FormatNumbers(m_sceneStats.nbTriangles).c_str());
    ImGui::Text("Image memory (bytes) : %s", FormatNumbers(m_sceneStats.imageMem).c_str());
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
void VkRtExample::importImages(tinygltf::Model& gltfModel)
{
  if(gltfModel.images.empty())
  {
    // Make dummy image(1,1), needed as we cannot have an empty array
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};
    m_textures.emplace_back(m_alloc.createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures[0].image, "dummy");
    return;
  }

  m_textures.resize(gltfModel.images.size());

  vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  vk::Format            format{vk::Format::eR8G8B8A8Unorm};
  samplerCreateInfo.maxLod = FLT_MAX;

  nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);

  vk::CommandBuffer cmdBuf = sc.createCommandBuffer();
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto&               gltfimage       = gltfModel.images[i];
    void*               buffer          = &gltfimage.image[0];
    VkDeviceSize        bufferSize      = gltfimage.image.size();
    auto                imgSize         = vk::Extent2D(gltfimage.width, gltfimage.height);
    vk::ImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures[i]                  = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)).c_str());
  }
  sc.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
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
}


//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
// - Home key: fit all, the camera will move to see the entire scene bounding box
// - Space: Trigger ray picking and set the interest point at the intersection
//          also return all information under the cursor
//
void VkRtExample::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvk::AppBase::onKeyboard(key, scancode, action, mods);
  if(action == GLFW_RELEASE)
    return;

  if(key == GLFW_KEY_HOME)
  {
    // Set the camera as to see the model
    if(m_gltfScene.m_cameras.empty())
      fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, false);
    else
    {
      nvmath::vec3f eye;
      m_gltfScene.m_cameras[0].worldMatrix.get_translation(eye);
      float len = nvmath::length(m_gltfScene.m_dimensions.center - eye);
      CameraManip.setMatrix(m_gltfScene.m_cameras[0].worldMatrix, false, len);
    }
  }

  if(key == GLFW_KEY_SPACE)
  {
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);

    // Set the camera as to see the model
    nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);
    vk::CommandBuffer cmdBuf = sc.createCommandBuffer();
    float             px     = x / float(m_size.width);
    float             py     = y / float(m_size.height);
    m_rayPicker.run(cmdBuf, px, py);
    sc.submitAndWait(cmdBuf);

    RayPicker::PickResult pr = m_rayPicker.getResult();

    if(pr.intanceID == ~0)
    {
      LOGI("Not Hit\n");
      return;
    }

    std::stringstream o;
    LOGI("\n Node:  %d", pr.intanceID);
    LOGI("\n PrimMesh:  %d", pr.intanceCustomID);
    LOGI("\n Triangle: %d", pr.primitiveID);
    LOGI("\n Distance:  %f", nvmath::length(pr.worldPos - m_sceneUbo.cameraPosition));
    LOGI("\n Position: %f, %f, %f \n", pr.worldPos.x, pr.worldPos.y, pr.worldPos.z);

    // Set the interest position
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, pr.worldPos, up, false);
  }
}

//--------------------------------------------------------------------------------------------------
// Drag and dropping a glTF file
//
void VkRtExample::onFileDrop(const char* filename)
{
  m_device.waitIdle();

  // Destroy all allocation: buffers, images
  m_gltfScene.destroy();
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_tangentBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  //  m_alloc.destroy(m_primitiveInfoBuffer);
  m_alloc.destroy(m_sceneBuffer);
  for(auto& t : m_textures)
    m_alloc.destroy(t);
  m_textures.clear();

  // Destroy descriptor layout, number of images might change
  m_device.destroy(m_descSetLayout[eScene]);
  m_device.destroy(m_descPool[eScene]);

  // Destroy Raytracer data: blas, tlas, descriptorsets
  m_rayPicker.destroy();
  m_raytracer.destroy();

  loadScene(filename);
  resetFrame();
}

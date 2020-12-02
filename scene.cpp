/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include "scene.hpp"
#include "binding.h"
#include "fileformats/tiny_gltf.h"
#include "fileformats/tiny_gltf_freeimage.h"
#include "imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "structures.h"
#include "tools.hpp"

using vkBU = vk::BufferUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;
using vkDS = vk::DescriptorSetLayoutBinding;
using vkDT = vk::DescriptorType;
using vkSS = vk::ShaderStageFlagBits;
using vkIU = vk::ImageUsageFlagBits;


namespace fs = std::filesystem;

//--------------------------------------------------------------------------------------------------
// Loading a GLTF Scene, allocate buffers and create descriptor set for all resources
//
bool Scene::load(const std::string& filename)
{
  destroy();
  m_gltfScene = {};

  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;
  MilliTimer         timer;

  // Loading the scene using tinygltf, but don't load textures with it
  // because it is faster to use FreeImage
  tcontext.SetImageLoader(nullptr, nullptr);
  LOGI("Loading scene: %s", filename.c_str());
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
    LOGE(error.c_str());
    assert(!"Error while loading scene");
    return false;
  }
  LOGW(warn.c_str());
  timer.print();

  // Loading images in parallel using FreeImage
  LOGI("Loading %d external images", tmodel.images.size());
  tinygltf::loadExternalImages(&tmodel, filename);
  timer.print();

  // Extracting GLTF information to our format and adding, if missing, attributes such as tangent
  LOGI("Convert to internal GLTF");
  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0 | nvh::GltfAttributes::Tangent);
  timer.print();

  // Setting all cameras found in the scene, such that they appears in the
  // Camera GUI helper
  ImGuiH::SetCameraJsonFile(fs::path(filename).stem().string());
  if(!m_gltfScene.m_cameras.empty())
  {
    auto& c = m_gltfScene.m_cameras[0];
    CameraManip.setCamera({c.eye, c.center, c.up, (float)rad2deg(c.cam.perspective.yfov)});
    ImGuiH::SetHomeCamera({c.eye, c.center, c.up, (float)rad2deg(c.cam.perspective.yfov)});

    for(auto& c : m_gltfScene.m_cameras)
    {
      ImGuiH::AddCamera({c.eye, c.center, c.up, (float)rad2deg(c.cam.perspective.yfov)});
    }
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    CameraManip.fit(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, true);
  }


  // Keeping statistics
  m_stats = m_gltfScene.getStatistics(tmodel);

  // Create scene information buffers and copy on the Device
  // vertices, indices, materials and all other scene attributes
  nvvk::CommandPool cmdBufGet(m_device, m_queueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();
  {
    m_buffers[eCameraMat] =
        m_pAlloc->createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer | vkBU::eTransferDst, vkMP::eDeviceLocal);
    m_buffers[eVertex] = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_positions, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
    m_buffers[eIndex] = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_indices, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
    m_buffers[eNormal]   = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_normals, vkBU::eStorageBuffer);
    m_buffers[eTexCoord] = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_texcoords0, vkBU::eStorageBuffer);
    m_buffers[eTangent]  = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_tangents, vkBU::eStorageBuffer);
    m_buffers[eMaterial] = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_materials, vkBU::eStorageBuffer);

    // Matrices of all instances
    // #TODO - This may not be in used, possible to avoid them using RTX hit information
    std::vector<InstanceMatrices> nodeMatrices;
    for(auto& node : m_gltfScene.m_nodes)
    {
      InstanceMatrices mat;
      mat.object2World = node.worldMatrix;
      mat.world2Object = invert(node.worldMatrix);
      nodeMatrices.emplace_back(mat);
    }
    m_buffers[eMatrix] = m_pAlloc->createBuffer(cmdBuf, nodeMatrices, vkBU::eStorageBuffer);

    // The following is used to find the primitive mesh information, offsets in buffers
    std::vector<RtPrimitiveLookup> primLookup;
    for(auto& primMesh : m_gltfScene.m_primMeshes)
      primLookup.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
    m_buffers[ePrimLookup] = m_pAlloc->createBuffer(cmdBuf, primLookup, vk::BufferUsageFlagBits::eStorageBuffer);

    // Debugging names
    m_debug.setObjectName(m_buffers[eCameraMat].buffer, "cameraMat");
    m_debug.setObjectName(m_buffers[eVertex].buffer, "Vertex");
    m_debug.setObjectName(m_buffers[eIndex].buffer, "Index");
    m_debug.setObjectName(m_buffers[eNormal].buffer, "Normal");
    m_debug.setObjectName(m_buffers[eTexCoord].buffer, "TexCoord");
    m_debug.setObjectName(m_buffers[eTangent].buffer, "Tangents");
    m_debug.setObjectName(m_buffers[eMaterial].buffer, "Material");
    m_debug.setObjectName(m_buffers[eMatrix].buffer, "Matrix");
    m_debug.setObjectName(m_buffers[ePrimLookup].buffer, "PrimLookup");

    // Creates all textures found
    createTextureImages(cmdBuf, tmodel);
    cmdBufGet.submitAndWait(cmdBuf);
    m_pAlloc->finalizeAndReleaseStaging();
  }

  // Descriptor set for all elements
  createDescriptorSet();

  return true;
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocated resources
//
void Scene::destroy()
{
  for(auto& b : m_buffers)
  {
    m_pAlloc->destroy(b);
    b = {};
  }

  for(auto& t : m_textures)
  {
    m_pAlloc->destroy(t);
    t = {};
  }
  m_textures.clear();

  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);

  m_gltfScene     = {};
  m_stats         = {};
  m_descPool      = vk::DescriptorPool();
  m_descSetLayout = vk::DescriptorSetLayout();
  m_descSet       = vk::DescriptorSet();
}


//--------------------------------------------------------------------------------------------------
// Uploading all images to the GPU
//
void Scene::createTextureImages(vk::CommandBuffer cmdBuf, tinygltf::Model& gltfModel)
{
  vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  vk::Format            format = vk::Format::eB8G8R8A8Unorm;
  samplerCreateInfo.setMaxLod(FLT_MAX);

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [this]() {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};
    m_textures.emplace_back(m_pAlloc->createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    addDefaultTexture();
    return;
  }

  m_textures.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto&        gltfimage  = gltfModel.images[i];
    void*        buffer     = &gltfimage.image[0];
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = vk::Extent2D(gltfimage.width, gltfimage.height);

    if(bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
    {
      addDefaultTexture();
      continue;
    }


    // Creating an image, the sampler and generating mipmaps
    vk::ImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

    nvvk::Image image = m_pAlloc->createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures.emplace_back(m_pAlloc->createTexture(image, ivInfo, samplerCreateInfo));


    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)).c_str());
  }
}

//--------------------------------------------------------------------------------------------------
// Creating the descriptor for the scene
//
void Scene::createDescriptorSet()
{
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  auto flag       = vkSS::eClosestHitKHR | vkSS::eAnyHitKHR | vkSS::eCompute | vkSS::eFragment;

  nvvk::DescriptorSetBindings bind;
  bind.addBinding(vkDS(B_CAMERA, vkDT::eUniformBuffer, 1, vkSS::eRaygenKHR | flag));
  bind.addBinding(vkDS(B_VERTICES, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_INDICES, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_NORMALS, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_TEXCOORDS, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_TANGENTS, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_MATERIALS, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_MATRICES, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures, flag));
  bind.addBinding(vkDS(B_PRIMLOOKUP, vkDT::eStorageBuffer, 1, flag));

  m_descSetLayout = bind.createLayout(m_device);
  m_descPool      = bind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
  m_debug.setObjectName(m_descSetLayout, "scene");
  m_debug.setObjectName(m_descSet, "scene");

  std::array<vk::DescriptorBufferInfo, 10> dbi;
  dbi[B_CAMERA]     = vk::DescriptorBufferInfo{m_buffers[eCameraMat].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_VERTICES]   = vk::DescriptorBufferInfo{m_buffers[eVertex].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_INDICES]    = vk::DescriptorBufferInfo{m_buffers[eIndex].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_NORMALS]    = vk::DescriptorBufferInfo{m_buffers[eNormal].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_TEXCOORDS]  = vk::DescriptorBufferInfo{m_buffers[eTexCoord].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_MATERIALS]  = vk::DescriptorBufferInfo{m_buffers[eMaterial].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_MATRICES]   = vk::DescriptorBufferInfo{m_buffers[eMatrix].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_TANGENTS]   = vk::DescriptorBufferInfo{m_buffers[eTangent].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_PRIMLOOKUP] = vk::DescriptorBufferInfo{m_buffers[ePrimLookup].buffer, 0, VK_WHOLE_SIZE};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_descSet, B_CAMERA, &dbi[B_CAMERA]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_VERTICES, &dbi[B_VERTICES]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_INDICES, &dbi[B_INDICES]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_NORMALS, &dbi[B_NORMALS]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_TEXCOORDS, &dbi[B_TEXCOORDS]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_TANGENTS, &dbi[B_TANGENTS]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_MATERIALS, &dbi[B_MATERIALS]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_MATRICES, &dbi[B_MATRICES]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_PRIMLOOKUP, &dbi[B_PRIMLOOKUP]));

  // All texture samplers
  std::vector<vk::DescriptorImageInfo> diit;
  for(auto& texture : m_textures)
    diit.emplace_back(texture.descriptor);
  writes.emplace_back(bind.makeWriteArray(m_descSet, B_TEXTURES, diit.data()));

  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Updating camera matrix
//
void Scene::updateCamera(const vk::CommandBuffer& cmdBuf)
{
  const float aspectRatio = CameraManip.getWidth() / static_cast<float>(CameraManip.getHeight());

  CameraMatrices ubo = {};
  ubo.view           = CameraManip.getMatrix();
  ubo.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.001f, 100000.0f);
  ubo.viewInverse    = nvmath::invert(ubo.view);
  ubo.projInverse    = nvmath::invert(ubo.proj);

  cmdBuf.updateBuffer<CameraMatrices>(m_buffers[eCameraMat].buffer, 0, ubo);

  // Making sure the matrix buffer will be available
  vk::MemoryBarrier mb{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead};
  cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                         vk::DependencyFlagBits::eDeviceGroup, {mb}, {}, {});
}

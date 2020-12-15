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


  LOGI("Loading scene: %s", filename.c_str());
  bool        result;
  std::string extension = fs::path(filename).extension().string();
  m_sceneName           = fs::path(filename).stem().string();
  if(extension == ".gltf")
  {
    // Loading the scene using tinygltf, but don't load textures with it
    // because it is faster to use FreeImage
    tcontext.SetImageLoader(nullptr, nullptr);
    result = tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename);
    if(result)
    {
      // Loading images in parallel using FreeImage
      LOGI("Loading %d external images", tmodel.images.size());
      tinygltf::loadExternalImages(&tmodel, filename);
      timer.print();
    }
  }
  else
  {
    // Binary loader
    tcontext.SetImageLoader(&tinygltf::LoadFreeImageData, nullptr);
    result = tcontext.LoadBinaryFromFile(&tmodel, &error, &warn, filename);
  }

  if(result == false)
  {
    LOGE(error.c_str());
    assert(!"Error while loading scene");
    return false;
  }
  LOGW(warn.c_str());
  timer.print();


  // Extracting GLTF information to our format and adding, if missing, attributes such as tangent
  LOGI("Convert to internal GLTF");
  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0
                                              | nvh::GltfAttributes::Tangent | nvh::GltfAttributes::Color_0);
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
  // We are using a different index (1), to allow loading in a different
  // queue/thread that the display (0)
  auto              queue = m_device.getQueue(m_queueFamillyIndex, 1);
  nvvk::CommandPool cmdBufGet(m_device, m_queueFamillyIndex, vk::CommandPoolCreateFlagBits::eTransient, queue);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();
  {
    m_buffers[eCameraMat] =
        m_pAlloc->createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer | vkBU::eTransferDst, vkMP::eDeviceLocal);
    m_buffers[eVertex] = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_positions, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
    m_buffers[eIndex] = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_indices, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
    m_buffers[eNormal]   = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_normals, vkBU::eStorageBuffer);
    m_buffers[eTexCoord] = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_texcoords0, vkBU::eStorageBuffer);
    m_buffers[eTangent]  = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_tangents, vkBU::eStorageBuffer);
    m_buffers[eColor]    = m_pAlloc->createBuffer(cmdBuf, m_gltfScene.m_colors0, vkBU::eStorageBuffer);


    std::vector<GltfShadeMaterial> shadeMaterials;
    for(auto& m : m_gltfScene.m_materials)
    {
      GltfShadeMaterial smat;
      smat.pbrBaseColorFactor           = m.pbrBaseColorFactor;
      smat.pbrBaseColorTexture          = m.pbrBaseColorTexture;
      smat.pbrMetallicFactor            = m.pbrMetallicFactor;
      smat.pbrRoughnessFactor           = m.pbrRoughnessFactor;
      smat.pbrMetallicRoughnessTexture  = m.pbrMetallicRoughnessTexture;
      smat.khrDiffuseFactor             = m.khrDiffuseFactor;
      smat.khrSpecularFactor            = m.khrSpecularFactor;
      smat.khrDiffuseTexture            = m.khrDiffuseTexture;
      smat.shadingModel                 = m.shadingModel;
      smat.khrGlossinessFactor          = m.khrGlossinessFactor;
      smat.khrSpecularGlossinessTexture = m.khrSpecularGlossinessTexture;
      smat.emissiveTexture              = m.emissiveTexture;
      smat.emissiveFactor               = m.emissiveFactor;
      smat.alphaMode                    = m.alphaMode;
      smat.alphaCutoff                  = m.alphaCutoff;
      smat.doubleSided                  = m.doubleSided;
      smat.normalTexture                = m.normalTexture;
      smat.normalTextureScale           = m.normalTextureScale;
      smat.uvTransform                  = m.uvTransform;
      smat.unlit                        = m.unlit;
      smat.anisotropy                   = m.anisotropy;
      smat.anisotropyDirection          = m.anisotropyDirection;


      shadeMaterials.emplace_back(smat);
    }
    m_buffers[eMaterial] = m_pAlloc->createBuffer(cmdBuf, shadeMaterials, vkBU::eStorageBuffer);

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
    m_debug.setObjectName(m_buffers[eColor].buffer, "Color");
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


vk::SamplerCreateInfo gltfSamplerToVulkan(tinygltf::Sampler& tsampler)
{
  vk::SamplerCreateInfo vk_sampler;

  std::map<int, vk::Filter> filters;
  filters[9728] = vk::Filter::eNearest;  // NEAREST
  filters[9729] = vk::Filter::eLinear;   // LINEAR
  filters[9984] = vk::Filter::eNearest;  // NEAREST_MIPMAP_NEAREST
  filters[9985] = vk::Filter::eLinear;   // LINEAR_MIPMAP_NEAREST
  filters[9986] = vk::Filter::eNearest;  // NEAREST_MIPMAP_LINEAR
  filters[9987] = vk::Filter::eLinear;   // LINEAR_MIPMAP_LINEAR

  std::map<int, vk::SamplerMipmapMode> mipmap;
  mipmap[9728] = vk::SamplerMipmapMode::eNearest;  // NEAREST
  mipmap[9729] = vk::SamplerMipmapMode::eNearest;  // LINEAR
  mipmap[9984] = vk::SamplerMipmapMode::eNearest;  // NEAREST_MIPMAP_NEAREST
  mipmap[9985] = vk::SamplerMipmapMode::eNearest;  // LINEAR_MIPMAP_NEAREST
  mipmap[9986] = vk::SamplerMipmapMode::eLinear;   // NEAREST_MIPMAP_LINEAR
  mipmap[9987] = vk::SamplerMipmapMode::eLinear;   // LINEAR_MIPMAP_LINEAR

  std::map<int, vk::SamplerAddressMode> addressMode;
  addressMode[33071] = vk::SamplerAddressMode::eClampToEdge;
  addressMode[33648] = vk::SamplerAddressMode::eMirroredRepeat;
  addressMode[10497] = vk::SamplerAddressMode::eRepeat;

  vk_sampler.setMagFilter(filters[tsampler.magFilter]);
  vk_sampler.setMinFilter(filters[tsampler.minFilter]);
  vk_sampler.setMipmapMode(mipmap[tsampler.minFilter]);

  vk_sampler.setAddressModeU(addressMode[tsampler.wrapS]);
  vk_sampler.setAddressModeV(addressMode[tsampler.wrapT]);
  vk_sampler.setAddressModeW(addressMode[tsampler.wrapR]);

  // Always allow LOD
  vk_sampler.maxLod = FLT_MAX;
  return vk_sampler;
}


//--------------------------------------------------------------------------------------------------
// Uploading all images to the GPU
//
void Scene::createTextureImages(vk::CommandBuffer cmdBuf, tinygltf::Model& gltfModel)
{
  vk::Format format = vk::Format::eB8G8R8A8Unorm;

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [this, cmdBuf]() {
    std::array<uint8_t, 4> white = {255, 255, 255, 255};
    m_textures.emplace_back(m_pAlloc->createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    // No images, add a default one.
    addDefaultTexture();
    return;
  }

  m_textures.reserve(gltfModel.textures.size());
  for(size_t i = 0; i < gltfModel.textures.size(); i++)
  {
    int sourceImage = gltfModel.textures[i].source;
    if(sourceImage >= gltfModel.images.size() || sourceImage < 0)
    {
      // Incorrect source image
      addDefaultTexture();
      continue;
    }

    auto& gltfimage = gltfModel.images[sourceImage];
    if(gltfimage.width == -1 || gltfimage.height == -1 || gltfimage.image.empty())
    {
      // Image not present or incorrectly loaded (image.empty)
      addDefaultTexture();
      continue;
    }

    void*        buffer     = &gltfimage.image[0];
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = vk::Extent2D(gltfimage.width, gltfimage.height);

    // Sampler
    vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
    if(gltfModel.textures[i].sampler > -1)
    {
      // Retrieve the texture sampler
      auto gltfSampler  = gltfModel.samplers[gltfModel.textures[i].sampler];
      samplerCreateInfo = gltfSamplerToVulkan(gltfSampler);
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
  bind.addBinding(vkDS(B_COLORS, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_MATERIALS, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_MATRICES, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures, flag));
  bind.addBinding(vkDS(B_PRIMLOOKUP, vkDT::eStorageBuffer, 1, flag));

  m_descSetLayout = bind.createLayout(m_device);
  m_descPool      = bind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
  m_debug.setObjectName(m_descSetLayout, "scene");
  m_debug.setObjectName(m_descSet, "scene");

  std::array<vk::DescriptorBufferInfo, 11> dbi;
  dbi[B_CAMERA]     = vk::DescriptorBufferInfo{m_buffers[eCameraMat].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_VERTICES]   = vk::DescriptorBufferInfo{m_buffers[eVertex].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_INDICES]    = vk::DescriptorBufferInfo{m_buffers[eIndex].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_NORMALS]    = vk::DescriptorBufferInfo{m_buffers[eNormal].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_TEXCOORDS]  = vk::DescriptorBufferInfo{m_buffers[eTexCoord].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_MATERIALS]  = vk::DescriptorBufferInfo{m_buffers[eMaterial].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_MATRICES]   = vk::DescriptorBufferInfo{m_buffers[eMatrix].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_TANGENTS]   = vk::DescriptorBufferInfo{m_buffers[eTangent].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_COLORS]     = vk::DescriptorBufferInfo{m_buffers[eColor].buffer, 0, VK_WHOLE_SIZE};
  dbi[B_PRIMLOOKUP] = vk::DescriptorBufferInfo{m_buffers[ePrimLookup].buffer, 0, VK_WHOLE_SIZE};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_descSet, B_CAMERA, &dbi[B_CAMERA]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_VERTICES, &dbi[B_VERTICES]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_INDICES, &dbi[B_INDICES]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_NORMALS, &dbi[B_NORMALS]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_TEXCOORDS, &dbi[B_TEXCOORDS]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_TANGENTS, &dbi[B_TANGENTS]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_COLORS, &dbi[B_COLORS]));
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

  CameraMatrices hostUBO = {};
  hostUBO.view           = CameraManip.getMatrix();
  hostUBO.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.001f, 100000.0f);
  hostUBO.viewInverse    = nvmath::invert(hostUBO.view);
  hostUBO.projInverse    = nvmath::invert(hostUBO.proj);

  // UBO on the device
  vk::Buffer deviceUBO = m_buffers[eCameraMat].buffer;

  // Ensure that the modified UBO is not visible to previous frames.
  vk::BufferMemoryBarrier beforeBarrier;
  beforeBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  beforeBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  beforeBarrier.setBuffer(deviceUBO);
  beforeBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                         vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eDeviceGroup, {}, {beforeBarrier}, {});

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  cmdBuf.updateBuffer<CameraMatrices>(deviceUBO, 0, hostUBO);

  // Making sure the updated UBO will be visible.
  vk::BufferMemoryBarrier afterBarrier;
  afterBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  afterBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  afterBarrier.setBuffer(deviceUBO);
  afterBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                         vk::DependencyFlagBits::eDeviceGroup, {}, {afterBarrier}, {});
}

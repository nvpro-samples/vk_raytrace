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



#include "scene.hpp"
#include "binding.h"
#include "fileformats/tiny_gltf_freeimage.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "shaders/compress.glsl"
#include "structures.h"
#include "tiny_gltf.h"
#include "tools.hpp"


using vkBU = vk::BufferUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;
using vkDS = vk::DescriptorSetLayoutBinding;
using vkDT = vk::DescriptorType;
using vkSS = vk::ShaderStageFlagBits;
using vkIU = vk::ImageUsageFlagBits;


namespace fs = std::filesystem;

void Scene::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device           = device;
  m_pAlloc           = allocator;
  m_queueFamilyIndex = familyIndex;
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
// Loading a GLTF Scene, allocate buffers and create descriptor set for all resources
//
bool Scene::load(const std::string& filename)
{
  destroy();
  nvh::GltfScene gltf;

  tinygltf::Model tmodel;
  if(loadGltfScene(filename, tmodel) == false)
    return false;

  m_stats = gltf.getStatistics(tmodel);


  // Extracting GLTF information to our format and adding, if missing, attributes such as tangent
  {
    LOGI("Convert to internal GLTF");
    MilliTimer timer;
    gltf.importMaterials(tmodel);
    gltf.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0
                                         | nvh::GltfAttributes::Tangent | nvh::GltfAttributes::Color_0);
    timer.print();
  }

  // Setting all cameras found in the scene, such that they appears in the camera GUI helper
  setCameraFromScene(filename, gltf);
  m_camera.nbLights = static_cast<int>(gltf.m_lights.size());

  // We are using a different index (1), to allow loading in a different queue/thread than the display (0) is using
  // Note: the GTC family queue is used because the nvvk::cmdGenerateMipmaps uses vkCmdBlitImage and this
  // command requires graphic queue and not only transfer.
  LOGI("Create Buffers\n");
  vk::Queue         queue = m_device.getQueue(m_queueFamilyIndex, 1);
  nvvk::CommandPool cmdBufGet(m_device, m_queueFamilyIndex, vk::CommandPoolCreateFlagBits::eTransient, queue);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();

  // Create camera buffer
  m_buffer[eCameraMat] = m_pAlloc->createBuffer(sizeof(SceneCamera), vkBU::eUniformBuffer | vkBU::eTransferDst, vkMP::eDeviceLocal);
  NAME_VK(m_buffer[eCameraMat].buffer);

  createMaterialBuffer(cmdBuf, gltf);
  createLightBuffer(cmdBuf, gltf);
  createTextureImages(cmdBuf, tmodel);
  createVertexBuffer(cmdBuf, gltf);
  createInstanceDataBuffer(cmdBuf, gltf);


  // Finalizing the command buffer - upload data to GPU
  LOGI(" <Finalize>");
  MilliTimer timer;
  cmdBufGet.submitAndWait(cmdBuf);
  m_pAlloc->finalizeAndReleaseStaging();
  timer.print();


  // Descriptor set for all elements
  createDescriptorSet(gltf);

  // Keeping minimal resources
  m_gltf.m_nodes      = gltf.m_nodes;
  m_gltf.m_primMeshes = gltf.m_primMeshes;
  m_gltf.m_materials  = gltf.m_materials;

  return true;
}

//--------------------------------------------------------------------------------------------------
//
//
bool Scene::loadGltfScene(const std::string& filename, tinygltf::Model& tmodel)
{
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;
  MilliTimer         timer;

  LOGI("Loading scene: %s", filename.c_str());
  bool        result;
  fs::path    fspath(filename);
  std::string extension = fspath.extension().string();
  m_sceneName           = fspath.stem().string();
  if(extension == ".gltf")
  {
    // Loading the scene using tinygltf, but don't load textures with it
    // because it is faster to use FreeImage
    tcontext.RemoveImageLoader();
    result = tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename);
    timer.print();
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
    timer.print();
  }

  if(result == false)
  {
    LOGE(error.c_str());
    assert(!"Error while loading scene");
    return false;
  }
  LOGW(warn.c_str());

  return true;
}

//--------------------------------------------------------------------------------------------------
// Information per instance/geometry (currently only material)
//
void Scene::createInstanceDataBuffer(vk::CommandBuffer cmdBuf, nvh::GltfScene& gltf)
{
  std::vector<InstanceData> instData;
  for(auto& primMesh : gltf.m_primMeshes)
  {
    instData.push_back({primMesh.materialIndex});
  }
  m_buffer[eInstData] = m_pAlloc->createBuffer(cmdBuf, instData, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NAME_VK(m_buffer[eInstData].buffer);
}

//--------------------------------------------------------------------------------------------------
// Creating a buffer per primitive mesh (BLAS) containing all Vertex (pos, nrm, .. )
// and a buffer of index.
//
// We are compressing the data, because it makes a huge difference in the raytracer when accessing the
// data.
//
// normal and tangent are compressed using "A Survey of Efficient Representations for Independent Unit Vectors"
// http://jcgt.org/published/0003/02/01/paper.pdf
// The handiness of the tangent is stored in the less significant bit of the V component of the tcoord.
// Color is encoded on 32bit
//
void Scene::createVertexBuffer(vk::CommandBuffer cmdBuf, const nvh::GltfScene& gltf)
{
  LOGI(" - Create %d Vertex Buffers", gltf.m_primMeshes.size());
  MilliTimer timer;

  std::vector<VertexAttributes> vertex;
  std::vector<uint32_t>         indices;


  std::unordered_map<std::string, nvvk::Buffer> m_cachePrimitive;

  uint32_t prim_idx{0};
  for(const nvh::GltfPrimMesh& primMesh : gltf.m_primMeshes)
  {

    // Create a key to find a primitive that is already uploaded
    std::stringstream o;
    {
      o << primMesh.vertexOffset << ":";
      o << primMesh.vertexCount;
    }
    std::string key           = o.str();
    bool        primProcessed = false;

    nvvk::Buffer v_buffer;
    auto         it = m_cachePrimitive.find(key);
    if(it == m_cachePrimitive.end())
    {

      vertex.resize(primMesh.vertexCount);
      for(size_t v_ctx = 0; v_ctx < primMesh.vertexCount; v_ctx++)
      {
        size_t           idx = primMesh.vertexOffset + v_ctx;
        VertexAttributes v;
        v.position = gltf.m_positions[idx];
        v.normal   = compress_unit_vec(gltf.m_normals[idx]);
        v.tangent  = compress_unit_vec(gltf.m_tangents[idx]);
        v.texcoord = gltf.m_texcoords0[idx];
        v.color    = packUnorm4x8(gltf.m_colors0[idx]);

        // Encode to the Less-Significant-Bit the handiness of the tangent
        // Not a significant change on the UV to make a visual difference
        //auto     uintBitsToFloat = [](uint32_t a) -> float { return *(float*)&(a); };
        //auto     floatBitsToUint = [](float a) -> uint32_t { return *(uint32_t*)&(a); };
        uint32_t value = floatBitsToUint(v.texcoord.y);
        if(gltf.m_tangents[idx].w > 0)
          value |= 1;  // set bit, H == +1
        else
          value &= ~1;  // clear bit, H == -1
        v.texcoord.y = uintBitsToFloat(value);

        vertex[v_ctx] = std::move(v);
      }
      v_buffer = m_pAlloc->createBuffer(cmdBuf, vertex, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
      NAME_IDX_VK(v_buffer.buffer, prim_idx);
      m_cachePrimitive[key] = v_buffer;
    }
    else
    {
      v_buffer = it->second;
    }


    indices.resize(primMesh.indexCount);
    for(size_t idx = 0; idx < primMesh.indexCount; idx++)
    {
      indices[idx] = gltf.m_indices[idx + primMesh.firstIndex];
    }

    //nvvk::Buffer v_buffer =
    //    m_pAlloc->createBuffer(cmdBuf, vertex, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    nvvk::Buffer i_buffer =
        m_pAlloc->createBuffer(cmdBuf, indices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    m_buffers[eVertex].push_back(v_buffer);
    //NAME_IDX_VK(v_buffer.buffer, prim_idx);

    m_buffers[eIndex].push_back(i_buffer);
    NAME_IDX_VK(i_buffer.buffer, prim_idx);

    prim_idx++;
  }
  timer.print();
}

//--------------------------------------------------------------------------------------------------
// Setting up the camera in the GUI from the camera found in the scene
// or, fit the camera to see the scene.
//
void Scene::setCameraFromScene(const std::string& filename, const nvh::GltfScene& gltf)
{
  ImGuiH::SetCameraJsonFile(fs::path(filename).stem().string());
  if(gltf.m_cameras.empty() == false)
  {
    auto& c = gltf.m_cameras[0];
    CameraManip.setCamera({c.eye, c.center, c.up, (float)rad2deg(c.cam.perspective.yfov)});
    ImGuiH::SetHomeCamera({c.eye, c.center, c.up, (float)rad2deg(c.cam.perspective.yfov)});

    for(auto& c : gltf.m_cameras)
    {
      ImGuiH::AddCamera({c.eye, c.center, c.up, (float)rad2deg(c.cam.perspective.yfov)});
    }
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    CameraManip.fit(gltf.m_dimensions.min, gltf.m_dimensions.max, true);
  }
}

//--------------------------------------------------------------------------------------------------
// Create a buffer of all lights
//
void Scene::createLightBuffer(vk::CommandBuffer cmdBuf, const nvh::GltfScene& gltf)
{
  std::vector<Light> all_lights;
  for(const auto& l_gltf : gltf.m_lights)
  {
    Light l;
    l.position     = l_gltf.worldMatrix * nvmath::vec4f(0, 0, 0, 1);
    l.direction    = l_gltf.worldMatrix * nvmath::vec4f(0, 0, -1, 0);
    l.color        = nvmath::vec3f(l_gltf.light.color[0], l_gltf.light.color[1], l_gltf.light.color[2]);
    l.innerConeCos = static_cast<float>(cos(l_gltf.light.spot.innerConeAngle));
    l.outerConeCos = static_cast<float>(cos(l_gltf.light.spot.outerConeAngle));
    l.range        = static_cast<float>(l_gltf.light.range);
    l.intensity    = static_cast<float>(l_gltf.light.intensity);
    if(l_gltf.light.type == "point")
      l.type = LightType_Point;
    else if(l_gltf.light.type == "directional")
      l.type = LightType_Directional;
    else if(l_gltf.light.type == "spot")
      l.type = LightType_Spot;
    all_lights.emplace_back(l);
  }

  if(all_lights.empty())  // Cannot be null
    all_lights.emplace_back(Light{});
  m_buffer[eLights] = m_pAlloc->createBuffer(cmdBuf, all_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NAME_VK(m_buffer[eLights].buffer);
}

//--------------------------------------------------------------------------------------------------
// Create a buffer of all materials
// Most parameters are supported, and GltfShadeMaterial is GLSL packed compliant
//
void Scene::createMaterialBuffer(vk::CommandBuffer cmdBuf, const nvh::GltfScene& gltf)
{
  LOGI(" - Create %d Material Buffer", gltf.m_materials.size());
  MilliTimer timer;

  std::vector<GltfShadeMaterial> shadeMaterials;
  for(auto& m : gltf.m_materials)
  {
    GltfShadeMaterial smat;
    smat.pbrBaseColorFactor           = m.baseColorFactor;
    smat.pbrBaseColorTexture          = m.baseColorTexture;
    smat.pbrMetallicFactor            = m.metallicFactor;
    smat.pbrRoughnessFactor           = m.roughnessFactor;
    smat.pbrMetallicRoughnessTexture  = m.metallicRoughnessTexture;
    smat.khrDiffuseFactor             = m.specularGlossiness.diffuseFactor;
    smat.khrSpecularFactor            = m.specularGlossiness.specularFactor;
    smat.khrDiffuseTexture            = m.specularGlossiness.diffuseTexture;
    smat.khrGlossinessFactor          = m.specularGlossiness.glossinessFactor;
    smat.khrSpecularGlossinessTexture = m.specularGlossiness.specularGlossinessTexture;
    smat.shadingModel                 = m.shadingModel;
    smat.emissiveTexture              = m.emissiveTexture;
    smat.emissiveFactor               = m.emissiveFactor;
    smat.alphaMode                    = m.alphaMode;
    smat.alphaCutoff                  = m.alphaCutoff;
    smat.doubleSided                  = m.doubleSided;
    smat.normalTexture                = m.normalTexture;
    smat.normalTextureScale           = m.normalTextureScale;
    smat.uvTransform                  = m.textureTransform.uvTransform;
    smat.unlit                        = m.unlit.active;
    smat.transmissionFactor           = m.transmission.factor;
    smat.transmissionTexture          = m.transmission.texture;
    smat.anisotropy                   = m.anisotropy.factor;
    smat.anisotropyDirection          = m.anisotropy.direction;
    smat.ior                          = m.ior.ior;
    smat.attenuationColor             = m.volume.attenuationColor;
    smat.thicknessFactor              = m.volume.thicknessFactor;
    smat.thicknessTexture             = m.volume.thicknessTexture;
    smat.attenuationDistance          = m.volume.attenuationDistance;
    smat.clearcoatFactor              = m.clearcoat.factor;
    smat.clearcoatRoughness           = m.clearcoat.roughnessFactor;
    smat.clearcoatTexture             = m.clearcoat.texture;
    smat.clearcoatRoughnessTexture    = m.clearcoat.roughnessTexture;

    shadeMaterials.emplace_back(smat);
  }
  m_buffer[eMaterial] = m_pAlloc->createBuffer(cmdBuf, shadeMaterials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NAME_VK(m_buffer[eMaterial].buffer);
  timer.print();
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocated resources
//
void Scene::destroy()
{

  for(auto& buffer : m_buffer)
  {
    m_pAlloc->destroy(buffer);
    buffer = {};
  }

  // This is to avoid deleting twice a buffer, the vector
  // of vertex buffer can be sharing buffers
  std::unordered_map<VkBuffer, nvvk::Buffer> map_bv;
  for(auto& buffers : m_buffers[eVertex])
    map_bv[buffers.buffer] = buffers;
  for(auto& bv : map_bv)
    m_pAlloc->destroy(bv.second);
  m_buffers[eVertex].clear();

  for(auto& buffers : m_buffers[eIndex])
  {
    m_pAlloc->destroy(buffers);
  }
  m_buffers[eIndex].clear();

  for(auto& i : m_images)
  {
    m_pAlloc->destroy(i.first);
    i = {};
  }
  m_images.clear();

  for(size_t i = 0; i < m_defaultTextures.size(); i++)
  {
    size_t last_index = m_defaultTextures[m_defaultTextures.size() - 1 - i];
    m_pAlloc->destroy(m_textures[last_index]);
    m_textures.erase(m_textures.begin() + last_index);
  }
  m_defaultTextures.clear();

  for(auto& t : m_textures)
  {
    vkDestroyImageView(m_device, t.descriptor.imageView, nullptr);
    t = {};
  }
  m_textures.clear();


  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);

  m_gltf          = {};
  m_stats         = {};
  m_descPool      = vk::DescriptorPool();
  m_descSetLayout = vk::DescriptorSetLayout();
  m_descSet       = vk::DescriptorSet();
}

//--------------------------------------------------------------------------------------------------
// Return the Vulkan sampler based on the glTF sampler information
//
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

  // Always allow LOD
  vk_sampler.maxLod = FLT_MAX;
  return vk_sampler;
}


//--------------------------------------------------------------------------------------------------
// Uploading all textures and images to the GPU
//
void Scene::createTextureImages(vk::CommandBuffer cmdBuf, tinygltf::Model& gltfModel)
{
  LOGI(" - Create %d Textures, %d Images", gltfModel.textures.size(), gltfModel.images.size());
  MilliTimer timer;

  vk::Format format = vk::Format::eB8G8R8A8Unorm;

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [this, cmdBuf]() {
    std::array<uint8_t, 4> white           = {255, 255, 255, 255};
    vk::ImageCreateInfo    imageCreateInfo = nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1});
    nvvk::Image            image           = m_pAlloc->createImage(cmdBuf, 4, white.data(), imageCreateInfo);
    m_images.push_back({image, imageCreateInfo});
    m_debug.setObjectName(m_images.back().first.image, "dummy");
  };

  // Make dummy texture/image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [this, cmdBuf]() {
    m_defaultTextures.push_back(m_textures.size());
    std::array<uint8_t, 4> white = {255, 255, 255, 255};
    m_textures.emplace_back(m_pAlloc->createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    // No images, add a default one.
    addDefaultTexture();
    timer.print();
    return;
  }

  // Creating all images
  m_images.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    size_t sourceImage = i;

    auto& gltfimage = gltfModel.images[sourceImage];
    if(gltfimage.width == -1 || gltfimage.height == -1 || gltfimage.image.empty())
    {
      // Image not present or incorrectly loaded (image.empty)
      addDefaultImage();
      continue;
    }

    void*        buffer     = &gltfimage.image[0];
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = vk::Extent2D(gltfimage.width, gltfimage.height);

    // Creating an image, the sampler and generating mipmaps
    vk::ImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);
    nvvk::Image         image           = m_pAlloc->createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    // nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    m_images.push_back({image, imageCreateInfo});

    NAME_IDX_VK(m_images[i].first.image, i);
  }

  // Creating the textures using the above images
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

    // Sampler
    vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
    if(gltfModel.textures[i].sampler > -1)
    {
      // Retrieve the texture sampler
      auto gltfSampler  = gltfModel.samplers[gltfModel.textures[i].sampler];
      samplerCreateInfo = gltfSamplerToVulkan(gltfSampler);
    }
    std::pair<nvvk::Image, vk::ImageCreateInfo>& image = m_images[sourceImage];
    vk::ImageViewCreateInfo ivInfo                     = nvvk::makeImageViewCreateInfo(image.first.image, image.second);
    m_textures.emplace_back(m_pAlloc->createTexture(image.first, ivInfo, samplerCreateInfo));

    NAME_IDX_VK(m_textures[i].image, i);
  }

  timer.print();
}

//--------------------------------------------------------------------------------------------------
// Creating the descriptor for the scene
// Vertex, Index and Textures are array of buffers or images
//
void Scene::createDescriptorSet(const nvh::GltfScene& gltf)
{
  auto flag       = vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR | vkSS::eCompute | vkSS::eFragment;
  auto nb_meshes  = static_cast<uint32_t>(gltf.m_primMeshes.size());
  auto nbTextures = static_cast<uint32_t>(m_textures.size());

  nvvk::DescriptorSetBindings bind;
  bind.addBinding(vkDS(B_CAMERA, vkDT::eUniformBuffer, 1, vkSS::eRaygenKHR | flag));
  bind.addBinding(vkDS(B_VERTEX, vkDT::eStorageBuffer, nb_meshes, flag));
  bind.addBinding(vkDS(B_INDICES, vkDT::eStorageBuffer, nb_meshes, flag));
  bind.addBinding(vkDS(B_MATERIALS, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures, flag));
  bind.addBinding(vkDS(B_INSTDATA, vkDT::eStorageBuffer, 1, flag));
  bind.addBinding(vkDS(B_LIGHTS, vkDT::eStorageBuffer, 1, flag));

  m_descPool = bind.createPool(m_device, 1);
  CREATE_NAMED_VK(m_descSetLayout, bind.createLayout(m_device));
  CREATE_NAMED_VK(m_descSet, nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

  std::array<vk::DescriptorBufferInfo, 5> dbi;
  dbi[eCameraMat] = vk::DescriptorBufferInfo{m_buffer[eCameraMat].buffer, 0, VK_WHOLE_SIZE};
  dbi[eMaterial]  = vk::DescriptorBufferInfo{m_buffer[eMaterial].buffer, 0, VK_WHOLE_SIZE};
  dbi[eInstData]  = vk::DescriptorBufferInfo{m_buffer[eInstData].buffer, 0, VK_WHOLE_SIZE};
  dbi[eLights]    = vk::DescriptorBufferInfo{m_buffer[eLights].buffer, 0, VK_WHOLE_SIZE};

  // array of buffers/images
  std::vector<vk::DescriptorBufferInfo> v_info;
  std::vector<vk::DescriptorBufferInfo> i_info;
  std::vector<vk::DescriptorImageInfo>  t_info;
  for(auto i = 0U; i < nb_meshes; i++)
  {
    v_info.push_back(vk::DescriptorBufferInfo{m_buffers[eVertex][i].buffer, 0, VK_WHOLE_SIZE});
    i_info.push_back(vk::DescriptorBufferInfo{m_buffers[eIndex][i].buffer, 0, VK_WHOLE_SIZE});
  }
  for(auto& texture : m_textures)
    t_info.emplace_back(texture.descriptor);

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_descSet, B_CAMERA, &dbi[eCameraMat]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_MATERIALS, &dbi[eMaterial]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_INSTDATA, &dbi[eInstData]));
  writes.emplace_back(bind.makeWrite(m_descSet, B_LIGHTS, &dbi[eLights]));
  writes.emplace_back(bind.makeWriteArray(m_descSet, B_VERTEX, v_info.data()));
  writes.emplace_back(bind.makeWriteArray(m_descSet, B_INDICES, i_info.data()));
  writes.emplace_back(bind.makeWriteArray(m_descSet, B_TEXTURES, t_info.data()));

  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Updating camera matrix
//
void Scene::updateCamera(const vk::CommandBuffer& cmdBuf, float aspectRatio)
{
  m_camera.view        = CameraManip.getMatrix();
  m_camera.proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.001f, 100000.0f);
  m_camera.viewInverse = nvmath::invert(m_camera.view);
  m_camera.projInverse = nvmath::invert(m_camera.proj);

  // Focal is the interest point
  nvmath::vec3f eye, center, up;
  CameraManip.getLookat(eye, center, up);
  m_camera.focalDist = nvmath::length(center - eye);

  // UBO on the device
  vk::Buffer deviceUBO = m_buffer[eCameraMat].buffer;

  // Ensure that the modified UBO is not visible to previous frames.
  vk::BufferMemoryBarrier beforeBarrier;
  beforeBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  beforeBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  beforeBarrier.setBuffer(deviceUBO);
  beforeBarrier.setSize(sizeof m_camera);
  cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                         vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eDeviceGroup, {}, {beforeBarrier}, {});

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  cmdBuf.updateBuffer<SceneCamera>(deviceUBO, 0, m_camera);

  // Making sure the updated UBO will be visible.
  vk::BufferMemoryBarrier afterBarrier;
  afterBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  afterBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  afterBarrier.setBuffer(deviceUBO);
  afterBarrier.setSize(sizeof m_camera);
  cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                         vk::DependencyFlagBits::eDeviceGroup, {}, {afterBarrier}, {});
}

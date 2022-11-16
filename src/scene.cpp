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
 * - Loading and storing the glTF scene
 * - Creates the buffers and descriptor set for the scene
 */


#include <sstream>

#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"

#include "shaders/host_device.h"
#include "scene.hpp"
#include "shaders/compress.glsl"
#include "tiny_gltf.h"
#include "tools.hpp"

#include "fileformats/tiny_gltf_freeimage.h"

namespace fs = std::filesystem;

void Scene::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const nvvk::Queue& queue, nvvk::ResourceAllocator* allocator)
{
  m_device = device;
  m_pAlloc = allocator;
  m_queue  = queue;
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
  nvvk::CommandPool cmdBufGet(m_device, m_queue.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_queue.queue);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  // Create camera buffer
  m_buffer[eCameraMat] = m_pAlloc->createBuffer(sizeof(SceneCamera), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
  m_gltf.m_dimensions = gltf.m_dimensions;

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
// Information per instance/geometry, the material it uses, and also the pointer to the vertex
// and index buffers
//
void Scene::createInstanceDataBuffer(VkCommandBuffer cmdBuf, nvh::GltfScene& gltf)
{
  std::vector<InstanceData> instData;
  uint32_t                  cnt{0};
  for(auto& primMesh : gltf.m_primMeshes)
  {
    InstanceData data;
    data.indexAddress  = nvvk::getBufferDeviceAddress(m_device, m_buffers[eIndex][cnt].buffer);
    data.vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_buffers[eVertex][cnt].buffer);
    data.materialIndex = primMesh.materialIndex;
    instData.emplace_back(data);
    cnt++;
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
void Scene::createVertexBuffer(VkCommandBuffer cmdBuf, const nvh::GltfScene& gltf)
{
  LOGI(" - Create %d Vertex Buffers", gltf.m_primMeshes.size());
  MilliTimer timer;

  std::vector<VertexAttributes> vertex{};
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
        VertexAttributes v{};
        v.position = gltf.m_positions[idx];
        v.normal   = compress_unit_vec(gltf.m_normals[idx]);
        v.tangent  = compress_unit_vec(nvmath::vec3f(gltf.m_tangents[idx]));  // See .w encoding below
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
      v_buffer = m_pAlloc->createBuffer(cmdBuf, vertex,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      NAME_IDX_VK(v_buffer.buffer, prim_idx);
      m_cachePrimitive[key] = v_buffer;
    }
    else
    {
      v_buffer = it->second;
    }

    // Buffer of indices
    indices.resize(primMesh.indexCount);
    for(size_t idx = 0; idx < primMesh.indexCount; idx++)
    {
      indices[idx] = gltf.m_indices[idx + primMesh.firstIndex];
    }

    nvvk::Buffer i_buffer = m_pAlloc->createBuffer(cmdBuf, indices,
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    m_buffers[eVertex].push_back(v_buffer);
    NAME_IDX_VK(v_buffer.buffer, prim_idx);

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
void Scene::createLightBuffer(VkCommandBuffer cmdBuf, const nvh::GltfScene& gltf)
{
  std::vector<Light> all_lights;
  for(const auto& l_gltf : gltf.m_lights)
  {
    Light l{};
    l.position  = nvmath::vec3f(l_gltf.worldMatrix * nvmath::vec4f(0, 0, 0, 1));
    l.direction = nvmath::vec3f(l_gltf.worldMatrix * nvmath::vec4f(0, 0, -1, 0));
    if(!l_gltf.light.color.empty())
      l.color = nvmath::vec3f(l_gltf.light.color[0], l_gltf.light.color[1], l_gltf.light.color[2]);
    else
      l.color = nvmath::vec3f(1, 1, 1);
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
// #TODO: compress the material, is it too large.
void Scene::createMaterialBuffer(VkCommandBuffer cmdBuf, const nvh::GltfScene& gltf)
{
  LOGI(" - Create %d Material Buffer", gltf.m_materials.size());
  MilliTimer timer;

  std::vector<GltfShadeMaterial> shadeMaterials;
  for(auto& m : gltf.m_materials)
  {
    GltfShadeMaterial smat{};
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
    smat.uvTransform                  = nvmath::mat4f(m.textureTransform.uvTransform);
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
    smat.sheen                        = packUnorm4x8(vec4(m.sheen.colorFactor, m.sheen.roughnessFactor));

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


  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  m_gltf          = {};
  m_stats         = {};
  m_descPool      = VkDescriptorPool();
  m_descSetLayout = VkDescriptorSetLayout();
  m_descSet       = VkDescriptorSet();
}

//--------------------------------------------------------------------------------------------------
// Return the Vulkan sampler based on the glTF sampler information
//
VkSamplerCreateInfo gltfSamplerToVulkan(tinygltf::Sampler& tsampler)
{
  VkSamplerCreateInfo vk_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  std::map<int, VkFilter> filters;
  filters[9728] = VK_FILTER_NEAREST;  // NEAREST
  filters[9729] = VK_FILTER_LINEAR;   // LINEAR
  filters[9984] = VK_FILTER_NEAREST;  // NEAREST_MIPMAP_NEAREST
  filters[9985] = VK_FILTER_LINEAR;   // LINEAR_MIPMAP_NEAREST
  filters[9986] = VK_FILTER_NEAREST;  // NEAREST_MIPMAP_LINEAR
  filters[9987] = VK_FILTER_LINEAR;   // LINEAR_MIPMAP_LINEAR

  std::map<int, VkSamplerMipmapMode> mipmap;
  mipmap[9728] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // NEAREST
  mipmap[9729] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // LINEAR
  mipmap[9984] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // NEAREST_MIPMAP_NEAREST
  mipmap[9985] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // LINEAR_MIPMAP_NEAREST
  mipmap[9986] = VK_SAMPLER_MIPMAP_MODE_LINEAR;   // NEAREST_MIPMAP_LINEAR
  mipmap[9987] = VK_SAMPLER_MIPMAP_MODE_LINEAR;   // LINEAR_MIPMAP_LINEAR

  std::map<int, VkSamplerAddressMode> addressMode;
  addressMode[33071] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  addressMode[33648] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  addressMode[10497] = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  vk_sampler.magFilter  = filters[tsampler.magFilter];
  vk_sampler.minFilter  = filters[tsampler.minFilter];
  vk_sampler.mipmapMode = mipmap[tsampler.minFilter];

  vk_sampler.addressModeU = addressMode[tsampler.wrapS];
  vk_sampler.addressModeV = addressMode[tsampler.wrapT];

  // Always allow LOD
  vk_sampler.maxLod = FLT_MAX;
  return vk_sampler;
}


//--------------------------------------------------------------------------------------------------
// Uploading all textures and images to the GPU
//
void Scene::createTextureImages(VkCommandBuffer cmdBuf, tinygltf::Model& gltfModel)
{
  LOGI(" - Create %d Textures, %d Images", gltfModel.textures.size(), gltfModel.images.size());
  MilliTimer timer;

  VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [this, cmdBuf]() {
    std::array<uint8_t, 4> white           = {255, 255, 255, 255};
    VkImageCreateInfo      imageCreateInfo = nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1});
    nvvk::Image            image           = m_pAlloc->createImage(cmdBuf, 4, white.data(), imageCreateInfo);
    m_images.emplace_back(image, imageCreateInfo);
    m_debug.setObjectName(m_images.back().first.image, "dummy");
  };

  // Make dummy texture/image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [this, cmdBuf]() {
    m_defaultTextures.push_back(m_textures.size());
    std::array<uint8_t, 4> white = {255, 255, 255, 255};
    VkSamplerCreateInfo    sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_textures.emplace_back(m_pAlloc->createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1}), sampler));
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
    auto         imgSize    = VkExtent2D{(uint32_t)gltfimage.width, (uint32_t)gltfimage.height};

    // Creating an image, the sampler and generating mipmaps
    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
    nvvk::Image       image           = m_pAlloc->createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    // nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    m_images.emplace_back(image, imageCreateInfo);

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
    VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
    samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if(gltfModel.textures[i].sampler > -1)
    {
      // Retrieve the texture sampler
      auto gltfSampler  = gltfModel.samplers[gltfModel.textures[i].sampler];
      samplerCreateInfo = gltfSamplerToVulkan(gltfSampler);
    }
    std::pair<nvvk::Image, VkImageCreateInfo>& image  = m_images[sourceImage];
    VkImageViewCreateInfo                      ivInfo = nvvk::makeImageViewCreateInfo(image.first.image, image.second);
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
  VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                            | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  auto nb_meshes  = static_cast<uint32_t>(gltf.m_primMeshes.size());
  auto nbTextures = static_cast<uint32_t>(m_textures.size());

  nvvk::DescriptorSetBindings bind;
  bind.addBinding({SceneBindings::eCamera, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, flag});
  bind.addBinding({SceneBindings::eMaterials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag});
  bind.addBinding({SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTextures, flag});
  bind.addBinding({SceneBindings::eInstData, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag});
  bind.addBinding({SceneBindings::eLights, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag});

  m_descPool = bind.createPool(m_device, 1);
  CREATE_NAMED_VK(m_descSetLayout, bind.createLayout(m_device));
  CREATE_NAMED_VK(m_descSet, nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

  std::array<VkDescriptorBufferInfo, 5> dbi;
  dbi[eCameraMat] = VkDescriptorBufferInfo{m_buffer[eCameraMat].buffer, 0, VK_WHOLE_SIZE};
  dbi[eMaterial]  = VkDescriptorBufferInfo{m_buffer[eMaterial].buffer, 0, VK_WHOLE_SIZE};
  dbi[eInstData]  = VkDescriptorBufferInfo{m_buffer[eInstData].buffer, 0, VK_WHOLE_SIZE};
  dbi[eLights]    = VkDescriptorBufferInfo{m_buffer[eLights].buffer, 0, VK_WHOLE_SIZE};

  // array of images
  std::vector<VkDescriptorImageInfo> t_info;
  for(auto& texture : m_textures)
    t_info.emplace_back(texture.descriptor);

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_descSet, SceneBindings::eCamera, &dbi[eCameraMat]));
  writes.emplace_back(bind.makeWrite(m_descSet, SceneBindings::eMaterials, &dbi[eMaterial]));
  writes.emplace_back(bind.makeWrite(m_descSet, SceneBindings::eInstData, &dbi[eInstData]));
  writes.emplace_back(bind.makeWrite(m_descSet, SceneBindings::eLights, &dbi[eLights]));
  writes.emplace_back(bind.makeWriteArray(m_descSet, SceneBindings::eTextures, t_info.data()));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Updating camera matrix
//
void Scene::updateCamera(const VkCommandBuffer& cmdBuf, float aspectRatio)
{
  const auto& view     = CameraManip.getMatrix();
  const auto  proj     = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.001f, 100000.0f);
  m_camera.viewInverse = nvmath::invert(view);
  m_camera.projInverse = nvmath::invert(proj);

  // Focal is the interest point
  nvmath::vec3f eye, center, up;
  CameraManip.getLookat(eye, center, up);
  m_camera.focalDist = nvmath::length(center - eye);

  // UBO on the device
  VkBuffer deviceUBO = m_buffer[eCameraMat].buffer;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.size          = sizeof(m_camera);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, deviceUBO, 0, sizeof(SceneCamera), &m_camera);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.size          = sizeof(m_camera);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &afterBarrier, 0, nullptr);
}

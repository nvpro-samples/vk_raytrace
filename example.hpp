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

#include <vulkan/vulkan.hpp>

#include "vkalloc.hpp"


#include <array>
#include <nvmath/nvmath.h>

#include "nvh/gltfscene.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "raypick.hpp"
#include "raytracer.hpp"
#include "skydome.hpp"
#include "tonemapper.hpp"


// Overloading the glTF scene to hold the vk::DescriptorImageInfo needed for Vulkan
struct gltfScene : nvh::gltf::Scene
{
  std::vector<vk::DescriptorImageInfo> m_textureDescriptors;

  void getMaterials(tinygltf::Model& gltfModel)
  {
    Scene::loadMaterials(gltfModel);
    m_textureDescriptors.resize(m_numTextures);
  }

  vk::DescriptorImageInfo& getDescriptor(nvh::gltf::TextureIDX idx) { return m_textureDescriptors[idx]; }
};

//--------------------------------------------------------------------------------------------------
// Loading a glTF scene, raytrace and tonemap result
//
class VkRtExample : public nvvk::AppBase
{
public:
  VkRtExample() = default;

  void initExample();
  void display();

  void setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex) override
  {
    AppBase::setup(instance, device, physicalDevice, graphicsQueueIndex);
    m_debug.setup(device);

#ifdef NVVK_ALLOC_DMA
    m_dmaAllocator.init(device, physicalDevice);
    m_alloc.init(device, physicalDevice, &m_dmaAllocator);
#elif defined(NVVK_ALLOC_DEDICATED)
    m_alloc.init(device, physicalDevice);
#endif

    m_raytracer.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
    m_rayPicker.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
    m_tonemapper.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
    m_skydome.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  }

  void destroy() override;
  void onResize(int w, int h) override;
  void createRenderPass() override;
  void onKeyboardChar(unsigned char key) override;
  void onKeyboard(int key, int scancode, int action, int mods) override;
  void setScene(const std::string& filename);
  void setEnvironmentHdr(const std::string& hdrFilename);

private:
  struct Light
  {
    nvmath::vec4f position{50.f, 50.f, 50.f, 1.f};
    nvmath::vec4f color{1.f, 1.f, 1.f, 1.f};
    //float         intensity{10.f};
    //float         _pad;
  };

  struct SceneUBO
  {
    nvmath::mat4f projection;
    nvmath::mat4f model;
    nvmath::vec4f cameraPosition{0.f, 0.f, 0.f};
    int           nbLights{0};
    int           _pad1{0};
    int           _pad2{0};
    int           _pad3{0};
    Light         lights[10];
  };

  // Each primitive/BLAS used for retrieving info
  struct PrimitiveSBO
  {
    uint32_t indexOffset;
    uint32_t vertexOffset;
    uint32_t materialIndex;
  };

  vk::GeometryNV primitiveToGeometry(const nvh::gltf::Primitive& prim);
  void           createDescriptorFinal();
  void           createDescriptorMaterial();
  void           createDescriptorRaytrace();
  void           updateDescriptor(const vk::DescriptorImageInfo& descriptor);
  void           createEmptyTexture();
  void           prepareUniformBuffers();
  void           createPipeline();
  void           updateCameraBuffer(const vk::CommandBuffer& cmdBuffer);
  void           drawUI();
  void           loadImages(tinygltf::Model& gltfModel);
  void           updateFrame();
  void           resetFrame();
  bool           uiLights();

  vk::RenderPass     m_renderPassUI;
  vk::PipelineLayout m_pipelineLayout;
  vk::Pipeline       m_pipeline;

  // Descriptors
  enum Dset
  {
    eFinal,     // For the tonemapper
    eMaterial,  // All materials
    eRaytrace,  // All info for raytracer
    Total
  };
  std::vector<nvvk::DescriptorSetBindings> m_descSetLayoutBind{Dset::Total};
  std::vector<vk::DescriptorSetLayout>     m_descSetLayout{Dset::Total};
  std::vector<vk::DescriptorPool>          m_descPool{Dset::Total};
  std::vector<vk::DescriptorSet>           m_descSet{Dset::Total};


  // GLTF scene model
  gltfScene                 m_gltfScene;         // The scene
  nvh::gltf::VertexData     m_vertices;          // All vertices
  std::vector<uint32_t>     m_indices;           // All indices
  SceneUBO                  m_sceneUbo;          // Camera, light and more
  nvvk::Texture               m_emptyTexture[2];   // black and white
  std::vector<PrimitiveSBO> m_primitiveOffsets;  // Primitive information: vertex, index offset + mat

  std::string m_filename;  // Scene filename
  std::string m_hdrFilename;

  int m_upVector = 1;  // Y up
  int m_frameNumber{0};

  nvvk::AxisVK m_axis;        // To display the axis in the lower left corner
  Raytracer    m_raytracer;   // The raytracer
  RayPicker    m_rayPicker;   // Picking under mouse using raytracer
  Tonemapper   m_tonemapper;  //
  SkydomePbr   m_skydome;     // The HDR environment

  // All buffers on the Device
  nvvk::Buffer m_sceneBuffer;
  nvvk::Buffer m_vertexBuffer;
  nvvk::Buffer m_normalBuffer;
  nvvk::Buffer m_uvBuffer;
  nvvk::Buffer m_indexBuffer;
  nvvk::Buffer m_matrixBuffer;
  nvvk::Buffer m_materialBuffer;
  nvvk::Buffer m_primitiveInfoBuffer;

  // All textures
  std::vector<nvvk::Texture> m_textures;

  // Memory allocator for buffers and images
#ifdef NVVK_ALLOC_DMA
  nvvk::DeviceMemoryAllocator   m_dmaAllocator;
#endif
  nvvk::Allocator m_alloc;

  nvvk::DebugUtil m_debug;
};

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

//--------------------------------------------------------------------------------------------------
// Loading a glTF scene, raytrace and tonemap result
//
class VkRtExample : public nvvk::AppBase
{
public:
  VkRtExample() = default;
  void destroy() override;
  void setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex) override;

  void loadScene(const std::string& filename);
  void loadEnvironmentHdr(const std::string& hdrFilename);

  void display();

  void createRenderPass() override;
  void createTonemapper();
  void createAxis();
  void createDescriptorFinal();
  void createFinalPipeline();

  void onKeyboardChar(unsigned char key) override;
  void onKeyboard(int key, int scancode, int action, int mods) override;
  void onFileDrop(const char* filename) override;
  void onResize(int w, int h) override;

private:
  struct Light
  {
    nvmath::vec4f position{50.f, 50.f, 50.f, 1.f};
    nvmath::vec4f color{1.f, 1.f, 1.f, 1.f};
  };

  struct SceneUBO
  {
    nvmath::mat4f projection;
    nvmath::mat4f model;
    nvmath::vec4f cameraPosition{0.f, 0.f, 0.f};
    int           debugMode{0};
    int           nbLights{0};
    Light         lights[1];
  };

  vk::GeometryNV primitiveToGeometry(const nvh::GltfPrimMesh& prim);
  void           createSceneDescriptors();
  void           createSceneBuffers();
  void           updateDescriptor(const vk::DescriptorImageInfo& descriptor);
  void           updateCameraBuffer(const vk::CommandBuffer& cmdBuffer);
  void           importImages(tinygltf::Model& gltfModel);
  void           updateFrame();
  void           resetFrame();
  void           drawUI();
  bool           uiLights();

  vk::RenderPass     m_renderPassUI;
  vk::PipelineLayout m_pipelineLayout;
  vk::Pipeline       m_pipeline;

  // Descriptors
  enum Dset
  {
    eFinal,  // For the tonemapper
    eScene,  // All scene data
    Total
  };
  std::vector<vk::DescriptorSetLayout>     m_descSetLayout{Dset::Total};
  std::vector<vk::DescriptorPool>          m_descPool{Dset::Total};
  std::vector<vk::DescriptorSet>           m_descSet{Dset::Total};
  std::vector<nvvk::DescriptorSetBindings> m_descSetLayoutBind{Dset::Total};


  // GLTF scene model
  nvh::GltfScene m_gltfScene;   // The scene
  nvh::GltfStats m_sceneStats;  // The scene stats

  SceneUBO m_sceneUbo;  // Camera, light and more

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
  nvvk::Buffer m_tangentBuffer;
  nvvk::Buffer m_indexBuffer;
  nvvk::Buffer m_materialBuffer;

  // All textures
  std::vector<nvvk::Texture> m_textures;

  // Memory allocator for buffers and images
#ifdef NVVK_ALLOC_DMA
  nvvk::MemAllocator m_memAllocator;
#endif
  nvvk::Allocator m_alloc;

  nvvk::DebugUtil m_debug;
};

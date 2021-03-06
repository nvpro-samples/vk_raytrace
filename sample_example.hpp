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


#pragma once
#include "hdr_sampling.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "renderer.h"

/*

 Structure of the application

    +--------------------------------------------+
    |             SampleExample                  |
    +--------+-----------------------------------+
    |  Pick  |    RtxPipeline   | other   ? ...  |
    +--------+---------+-------------------------+
    |       TLAS       |                         |
    +------------------+     Offscreen           |
    |      Scene       |                         |
    +------------------+-------------------------+

*/


// #define ALLOC_DMA  <--- This is in the CMakeLists.txt
#include "nvvk/resourceallocator_vk.hpp"
#if defined(ALLOC_DMA)
#include <nvvk/memallocator_dma_vk.hpp>
typedef nvvk::ResourceAllocatorDma Allocator;
#elif defined(ALLOC_VMA)
#include <nvvk/memallocator_vma_vk.hpp>
typedef nvvk::ResourceAllocatorVma Allocator;
#else
typedef nvvk::ResourceAllocatorDedicated Allocator;
#endif

#define CPP  // For sun_and_sky

#include "nvh/gltfscene.hpp"
#include "nvvk/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/raypicker_vk.hpp"

#include "accelstruct.hpp"
#include "offscreen.hpp"
#include "scene.hpp"
#include "shaders/sun_and_sky.h"
#include "structures.h"

#include "imgui_internal.h"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class SampleExample : public nvvk::AppBaseVk
{
public:
  enum RndMethod
  {
    eRtxPipeline,
    eRayQuery,
    eNone,
  };

  void setup(const VkInstance&       instance,
             const VkDevice&         device,
             const VkPhysicalDevice& physicalDevice,
             uint32_t                  gtcQueueIndexFamily,
             uint32_t                  computeQueueIndex,
             uint32_t                  transferQueueIndex);
  void createDescriptorSetLayout();
  void updateHdrDescriptors();
  void loadScene(const std::string& filename);
  void loadEnvironmentHdr(const std::string& hdrFilename);
  void createUniformBuffer();
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();

  void updateFrame();
  void resetFrame();

  bool isBusy() { return m_busy; }

  bool guiCamera();
  bool guiTonemapper();
  bool guiEnvironment();
  bool guiStatistics();
  bool guiProfiler(nvvk::ProfilerVK& profiler);
  bool guiGpuMeasures();
  void showBusyWindow();
  bool guiRayTracing();
  void titleBar();


  void menuBar();
  void onKeyboard(int key, int scancode, int action, int mods) override;

  void screenPicking();

  void onFileDrop(const char* filename) override;

  void loadAssets(const char* filename);

  void onMouseMotion(int x, int y) override;
  void onMouseButton(int button, int action, int mods) override;

  VkRenderPass  getOffscreenRenderPass() { return m_offscreen.getRenderPass(); }
  VkFramebuffer getOffscreenFrameBuffer() { return m_offscreen.getFrameBuffer(); }


  Scene              m_scene;
  AccelStructure     m_accelStruct;
  SunAndSky          m_sunAndSky;
  Offscreen          m_offscreen;
  HdrSampling        m_skydome;
  nvvk::AxisVK       m_axis;
  nvvk::RayPickerKHR m_picker;

  // All renderers
  std::array<Renderer*, eNone> m_pRender;
  RndMethod                    m_rndMethod{eNone};

  nvvk::Buffer m_sunAndSkyBuffer;

  // Graphic pipeline
  VkDescriptorPool          m_descPool;
  VkDescriptorSetLayout     m_descSetLayout;
  VkDescriptorSet           m_descSet;
  nvvk::DescriptorSetBindings m_bind;

  Allocator       m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil m_debug;  // Utility to name objects


  VkRect2D m_renderRegion;
  void       setRenderRegion(const VkRect2D& size);

  // #Post
  void createOffscreenRender();
  void drawPost(VkCommandBuffer cmdBuf);

  // #VKRay
  void render(RndMethod method, const VkCommandBuffer& cmdBuf, nvvk::ProfilerVK& profiler);


  RtxState    m_rtxState{};
  int         m_maxFrames{100000};
  bool        m_showAxis{true};
  bool        m_descaling{false};
  int         m_descalingLevel{1};
  bool        m_busy{false};
  std::string m_busyReasonText;
};

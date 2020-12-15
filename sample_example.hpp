/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "hdr_sampling.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "renderer.h"
#include <vulkan/vulkan.hpp>

/*

 Structure of the application

    +--------------------------------------------+
    |             SampleExample                  |
    +--------+-----------------------------------+
    |  Pick  |    RtxPipeline   | other   ? ...     |
    +--------+---------+-------------------------+
    |       TLAS       |                         |
    +------------------+     Offscreen           |
    |      Scene       |                         |
    +------------------+-------------------------+

*/


// #define NVVK_ALLOC_DMA  <--- This is in the CMakeLists.txt
#define CPP  // For sun_and_sky

#include "nvh/gltfscene.hpp"
#include "nvvk/allocator_vk.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"

#include "accelstruct.hpp"
#include "offscreen.hpp"
#include "raypick_KHR.hpp"
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
class SampleExample : public nvvk::AppBase
{
public:
  enum RndMethod
  {
    eRtxPipeline,
    eNone,
  };

  void setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueFamily, uint32_t computeQueueIndex);
  void createDescriptorSetLayout();
  void updateHdrDescriptors();
  void loadScene(const std::string& filename);
  void loadEnvironmentHdr(const std::string& hdrFilename);
  void createUniformBuffer();
  void updateUniformBuffer(const vk::CommandBuffer& cmdBuf);
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
  void onFileDrop(const char* filename) override;

  void loadAssets(const char* filename);

  void onMouseMotion(int x, int y) override;
  void onMouseButton(int button, int action, int mods) override;

  vk::RenderPass  getOffscreenRenderPass() { return m_offscreen.getRenderPass(); }
  vk::Framebuffer getOffscreenFrameBuffer() { return m_offscreen.getFrameBuffer(); }


  Scene          m_scene;
  AccelStructure m_accelStruct;
  SunAndSky      m_sunAndSky;
  Offscreen      m_offscreen;
  RayPickerKHR   m_picker;
  HdrSampling    m_skydome;
  nvvk::AxisVK   m_axis;

  // All renderers
  std::array<Renderer*, eNone> m_pRender;
  RndMethod                    m_rndMethod{eNone};

  nvvk::Buffer m_sunAndSkyBuffer;

  // Graphic pipeline
  vk::DescriptorPool          m_descPool;
  vk::DescriptorSetLayout     m_descSetLayout;
  vk::DescriptorSet           m_descSet;
  nvvk::DescriptorSetBindings m_bind;


  nvvk::MemAllocator m_memAllocator;
  nvvk::Allocator    m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil    m_debug;  // Utility to name objects

  // #Post
  void createOffscreenRender();
  void drawPost(vk::CommandBuffer cmdBuf);

  // #VKRay
  void render(RndMethod method, const vk::CommandBuffer& cmdBuf, nvvk::ProfilerVK& profiler);


  RtxState    m_state{0 /*frame*/, 5 /*depth*/, 1 /*sample*/, 1 /*firefly*/, 1 /*intensity*/, 0 /*debug mode*/};
  int         m_maxFrames{1000};
  bool        m_showAxis{true};
  bool        m_descaling{false};
  int         m_descalingLevel{1};
  bool        m_busy{false};
  std::string m_busyReasonText;
};

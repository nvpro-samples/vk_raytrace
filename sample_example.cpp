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

#define VMA_IMPLEMENTATION

#include <iomanip>
#include <iostream>
#include <sstream>

#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include "binding.h"
#include "imgui/imgui_helper.h"
#include "imgui/imgui_orient.h"
#include "rayquery.hpp"
#include "rtx_pipeline.hpp"
#include "sample_example.hpp"
#include "tools.hpp"


#include "fileformats/tiny_gltf_freeimage.h"

#include "nvml_monitor.hpp"

static NvmlMonitor g_nvml(100, 100);

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void SampleExample::setup(const VkInstance&       instance,
                          const VkDevice&         device,
                          const VkPhysicalDevice& physicalDevice,
                          uint32_t                gtcQueueIndexFamily,
                          uint32_t                computeQueueIndex,
                          uint32_t                transferQueueIndex)
{
  AppBaseVk::setup(instance, device, physicalDevice, gtcQueueIndexFamily);

  // Memory allocator for buffers and images
  m_alloc.init(instance, device, physicalDevice);

  m_debug.setup(m_device);

  m_sunAndSky = SunAndSky_default();

  // Compute queues can be use for acceleration structures
  m_picker.setup(m_device, physicalDevice, computeQueueIndex, &m_alloc);
  m_accelStruct.setup(m_device, physicalDevice, computeQueueIndex, &m_alloc);

  // Note: the GTC family queue is used because the nvvk::cmdGenerateMipmaps uses vkCmdBlitImage and this
  // command requires graphic queue and not only transfer.
  m_scene.setup(m_device, physicalDevice, gtcQueueIndexFamily, &m_alloc);

  // Transfer queues can be use for the creation of the following assets
  m_offscreen.setup(m_device, physicalDevice, transferQueueIndex, &m_alloc);
  m_skydome.setup(device, physicalDevice, transferQueueIndex, &m_alloc);

  // Create and setup all renderers
  m_pRender[eRtxPipeline] = new RtxPipeline;
  m_pRender[eRayQuery]    = new RayQuery;
  for(auto r : m_pRender)
  {
    r->setup(m_device, physicalDevice, transferQueueIndex, &m_alloc);
  }


  // Default RTX state (push_constant)
  m_rtxState.frame                 = 0;
  m_rtxState.maxDepth              = 10;
  m_rtxState.frame                 = 0;   // Current frame, start at 0
  m_rtxState.maxDepth              = 10;  // How deep the path is
  m_rtxState.maxSamples            = 1;   // How many samples to do per render
  m_rtxState.fireflyClampThreshold = 1;   // to cut fireflies
  m_rtxState.hdrMultiplier         = 1;   // To brightening the scene
  m_rtxState.debugging_mode        = 0;   //
  m_rtxState.pbrMode               = 0;   // 0-Disney, 1-glTF

  m_rtxState.minHeatmap = 0;
  m_rtxState.maxHeatmap = 65000;
}


//--------------------------------------------------------------------------------------------------
// Loading the scene file and setting up all buffers
//
void SampleExample::loadScene(const std::string& filename)
{
  m_scene.load(filename);
  m_accelStruct.create(m_scene.getScene(), m_scene.getBuffers(Scene::eVertex), m_scene.getBuffers(Scene::eIndex));

  m_picker.setTlas(m_accelStruct.getTlas());
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
//
//
void SampleExample::loadEnvironmentHdr(const std::string& hdrFilename)
{
  MilliTimer timer;
  LOGI("Loading HDR and converting %s\n", hdrFilename.c_str());
  m_skydome.loadEnvironment(hdrFilename);
  LOGI(" --> (%5.3f ms)\n", timer.elapse());

  //m_offscreen.m_tReinhard.avgLum = m_skydome.getAverage();
  m_rtxState.fireflyClampThreshold = m_skydome.getIntegral() * 4.f;  //magic
}


//--------------------------------------------------------------------------------------------------
// Loading asset in a separate thread
// - Used by file drop and menu operation
// Marking the session as busy, to avoid calling rendering while loading assets
//
void SampleExample::loadAssets(const char* filename)
{
  std::string sfile = filename;

  // Need to stop current rendering
  m_busy = true;
  vkDeviceWaitIdle(m_device);

  std::thread([&, sfile]() {
    LOGI("Loading: %s\n", sfile.c_str());

    // Supporting only GLTF and HDR files
    namespace fs          = std::filesystem;
    std::string extension = fs::path(sfile).extension().string();
    if(extension == ".gltf" || extension == ".glb")
    {
      m_busyReasonText = "Loading scene ";

      // Loading scene and creating acceleration structure
      loadScene(sfile);
      // Loading the scene might have loaded new textures, which is changing the number of elements
      // in the DescriptorSetLayout. Therefore, the PipelineLayout will be out-of-date and need
      // to be re-created. If they are re-created, the pipeline also need to be re-created.
      m_pRender[m_rndMethod]->create(
          m_size, {m_accelStruct.getDescLayout(), m_offscreen.getDescLayout(), m_scene.getDescLayout(), m_descSetLayout}, &m_scene);
    }

    if(extension == ".hdr")  //|| extension == ".exr")
    {
      m_busyReasonText = "Loading HDR ";
      loadEnvironmentHdr(sfile);
      updateHdrDescriptors();
    }


    // Re-starting the frame count to 0
    SampleExample::resetFrame();
    m_busy = false;
  }).detach();
}


//--------------------------------------------------------------------------------------------------
// Called at each frame to update the environment (sun&sky)
//
void SampleExample::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  const float aspectRatio = m_renderRegion.extent.width / static_cast<float>(m_renderRegion.extent.height);

  m_scene.updateCamera(cmdBuf, aspectRatio);
  vkCmdUpdateBuffer(cmdBuf, m_sunAndSkyBuffer.buffer, 0, sizeof(SunAndSky), &m_sunAndSky);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame otherwise, increments frame.
//
void SampleExample::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         fov = 0;

  auto& m = CameraManip.getMatrix();
  auto  f = CameraManip.getFov();
  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || f != fov)
  {
    resetFrame();
    refCamMatrix = m;
    fov          = f;
  }

  if(m_rtxState.frame < m_maxFrames)
    m_rtxState.frame++;
}

void SampleExample::resetFrame()
{
  m_rtxState.frame = -1;
}

//--------------------------------------------------------------------------------------------------
// Descriptors for the Sun&Sky buffer
//
void SampleExample::createDescriptorSetLayout()
{
  VkShaderStageFlags flags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                             | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;


  m_bind.addBinding({B_SUNANDSKY, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_MISS_BIT_KHR | flags});
  m_bind.addBinding({B_HDR, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags});          // HDR image
  m_bind.addBinding({B_IMPORT_SMPL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags});  // importance sampling


  m_descPool = m_bind.createPool(m_device, 1);
  CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
  CREATE_NAMED_VK(m_descSet, nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

  // Using the environment
  std::vector<VkWriteDescriptorSet> writes;
  VkDescriptorBufferInfo            sunskyDesc{m_sunAndSkyBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_bind.makeWrite(m_descSet, B_SUNANDSKY, &sunskyDesc));
  writes.emplace_back(m_bind.makeWrite(m_descSet, B_HDR, &m_skydome.m_textures.txtHdr.descriptor));
  writes.emplace_back(m_bind.makeWrite(m_descSet, B_IMPORT_SMPL, &m_skydome.m_textures.accelImpSmpl.descriptor));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void SampleExample::updateHdrDescriptors()
{
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_bind.makeWrite(m_descSet, B_HDR, &m_skydome.m_textures.txtHdr.descriptor));
  writes.emplace_back(m_bind.makeWrite(m_descSet, B_IMPORT_SMPL, &m_skydome.m_textures.accelImpSmpl.descriptor));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void SampleExample::createUniformBuffer()
{
  m_sunAndSkyBuffer = m_alloc.createBuffer(sizeof(SunAndSky), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  NAME_VK(m_sunAndSkyBuffer.buffer);
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void SampleExample::destroyResources()
{
  // Resources
  m_alloc.destroy(m_sunAndSkyBuffer);

  // Descriptors
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  // Other
  m_picker.destroy();
  m_scene.destroy();
  m_accelStruct.destroy();
  m_offscreen.destroy();
  m_skydome.destroy();
  m_axis.deinit();

  // All renderers
  for(auto p : m_pRender)
  {
    p->destroy();
    p = nullptr;
  }

  // Memory
  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void SampleExample::onResize(int /*w*/, int /*h*/)
{
  m_offscreen.update(m_size);
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// The size of the rendering area is smaller than the viewport
// This is the space left in the center view.
void SampleExample::setRenderRegion(const VkRect2D& size)
{
  if(memcmp(&m_renderRegion, &size, sizeof(VkRect2D)) != 0)
    resetFrame();
  m_renderRegion = size;
}

//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////

void SampleExample::createOffscreenRender()
{
  m_offscreen.create(m_size, m_renderPass);
  m_axis.init(m_device, m_renderPass, 0, 50.0f);
}


void SampleExample::drawPost(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  m_offscreen.m_tonemapper.zoom           = m_descaling ? 1.0f / m_descalingLevel : 1.0f;
  auto size                               = nvmath::vec2f(m_size.width, m_size.height);
  auto area                               = nvmath::vec2f(m_renderRegion.extent.width, m_renderRegion.extent.height);
  m_offscreen.m_tonemapper.renderingRatio = size / area;

  VkViewport viewport{static_cast<float>(m_renderRegion.offset.x),
                      static_cast<float>(m_renderRegion.offset.y),
                      static_cast<float>(m_size.width),
                      static_cast<float>(m_size.height),
                      0.0f,
                      1.0f};
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

  VkRect2D scissor{{0, 0}, {m_renderRegion.extent.width, m_renderRegion.extent.height}};
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);


  m_offscreen.run(cmdBuf);
  if(m_showAxis)
    m_axis.display(cmdBuf, CameraManip.getMatrix(), m_size);
}

//////////////////////////////////////////////////////////////////////////
// Ray tracing
//////////////////////////////////////////////////////////////////////////

void SampleExample::render(RndMethod method, const VkCommandBuffer& cmdBuf, nvvk::ProfilerVK& profiler)
{
  LABEL_SCOPE_VK(cmdBuf);
  g_nvml.refresh();

  // We are done rendering
  if(m_rtxState.frame >= m_maxFrames)
    return;

  // #TODO - add more rendering engines
  if(method != m_rndMethod)
  {
    LOGI("Switching renderer, from %d to %d \n", m_rndMethod, method);
    vkDeviceWaitIdle(m_device);  // cannot destroy while in use
    if(m_rndMethod != eNone)
      m_pRender[m_rndMethod]->destroy();
    m_rndMethod = method;

    m_pRender[m_rndMethod]->create(
        m_size, {m_accelStruct.getDescLayout(), m_offscreen.getDescLayout(), m_scene.getDescLayout(), m_descSetLayout}, &m_scene);
  }

  // Handling de-scaling by reducing the size to render
  VkExtent2D render_size = m_renderRegion.extent;
  if(m_descaling)
    render_size = VkExtent2D{render_size.width / m_descalingLevel, render_size.height / m_descalingLevel};

  m_rtxState.size = {render_size.width, render_size.height};
  // State is the push constant structure
  m_pRender[m_rndMethod]->m_state = m_rtxState;
  // Running the renderer
  m_pRender[m_rndMethod]->run(cmdBuf, render_size, profiler,
                              {m_accelStruct.getDescSet(), m_offscreen.getDescSet(), m_scene.getDescSet(), m_descSet});
}


//////////////////////////////////////////////////////////////////////////
/// GUI
//////////////////////////////////////////////////////////////////////////
using GuiH = ImGuiH::Control;


//--------------------------------------------------------------------------------------------------
//
//
void SampleExample::titleBar()
{
  static float dirtyTimer = 0.0f;

  dirtyTimer += ImGui::GetIO().DeltaTime;
  if(dirtyTimer > 1)
  {
    std::stringstream o;
    o << "VK glTF Viewer";
    o << " | " << m_scene.getSceneName();                                              // Scene name
    o << " | " << m_renderRegion.extent.width << "x" << m_renderRegion.extent.height;  // resolution
    o << " | " << static_cast<int>(ImGui::GetIO().Framerate)                           // FPS / ms
      << " FPS / " << std::setprecision(3) << 1000.F / ImGui::GetIO().Framerate << "ms";
    if(g_nvml.isValid())  // Graphic card, driver
    {
      const auto& i = g_nvml.getInfo(0);
      o << " | " << i.name;
      o << " | " << g_nvml.getSysInfo().driverVersion;
    }
    if(m_rndMethod != eNone && m_pRender[m_rndMethod] != nullptr)
      o << " | " << m_pRender[m_rndMethod]->name();
    glfwSetWindowTitle(m_window, o.str().c_str());
    dirtyTimer = 0;
  }
}

void SampleExample::menuBar()
{
  auto openFilename = [](const char* filter) {
#ifdef _WIN32
    char         filename[MAX_PATH];
    OPENFILENAME ofn;
    ZeroMemory(&filename, sizeof(filename));
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner   = nullptr;  // If you have a window to center over, put its HANDLE here
    ofn.lpstrFilter = filter;
    ofn.lpstrFile   = filename;
    ofn.nMaxFile    = MAX_PATH;
    ofn.lpstrTitle  = "Select a File";
    ofn.Flags       = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

    if(GetOpenFileNameA(&ofn))
    {
      return std::string(filename);
    }
#endif

    return std::string("");
  };


  // Menu Bar
  if(ImGui::BeginMainMenuBar())
  {
    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Open GLTF Scene"))
        loadAssets(openFilename("GLTF Files\0*.gltf;*.glb\0\0").c_str());
      if(ImGui::MenuItem("Open HDR Environment"))
        loadAssets(openFilename("HDR Files\0*.hdr\0\0").c_str());
      ImGui::Separator();
      if(ImGui::MenuItem("Quit", "ESC"))
        glfwSetWindowShouldClose(m_window, 1);
      ImGui::EndMenu();
    }

    if(ImGui::BeginMenu("Tools"))
    {
      ImGui::MenuItem("Settings", "F10", &m_show_gui);
      ImGui::MenuItem("Axis", nullptr, &m_showAxis);
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }
}

//--------------------------------------------------------------------------------------------------
//
//
bool SampleExample::guiCamera()
{
  bool changed{false};
  changed |= ImGuiH::CameraWidget();
  auto& cam = m_scene.getCamera();
  changed |= GuiH::Slider("Aperture", "", &cam.aperture, nullptr, ImGuiH::Control::Flags::Normal, 0.0f, 0.5f);

  return changed;
}

bool SampleExample::guiRayTracing()
{
  auto Normal = ImGuiH::Control::Flags::Normal;
  bool changed{false};

  changed |= GuiH::Slider("Max Ray Depth", "", &m_rtxState.maxDepth, nullptr, Normal, 1, 10);
  changed |= GuiH::Slider("Samples Per Frame", "", &m_rtxState.maxSamples, nullptr, Normal, 1, 10);
  changed |= GuiH::Slider("Max Iteration ", "", &m_maxFrames, nullptr, Normal, 1, 1000);
  changed |= GuiH::Slider("De-scaling ",
                          "Reduce resolution while navigating.\n"
                          "Speeding up rendering while camera moves.\n"
                          "Value of 1, will not de-scale",
                          &m_descalingLevel, nullptr, Normal, 1, 8);

  changed |= GuiH::Selection("Pbr Mode", "PBR material model", &m_rtxState.pbrMode, nullptr, Normal, {"Disney", "Gltf"});

  static bool bAnyHit = true;
  if(GuiH::Checkbox("Enable AnyHit",
                    "AnyHit is used for double sided, cutout opacity, but can be slower when all objects are opaque", &bAnyHit, nullptr))
  {
    auto rtx = dynamic_cast<RtxPipeline*>(m_pRender[m_rndMethod]);
    vkDeviceWaitIdle(m_device);  // cannot run while changing this
    rtx->useAnyHit(bAnyHit);
    changed = true;
  }

  GuiH::Group<bool>("Debugging", false, [&] {
    changed |= GuiH::Selection("Debug Mode", "Display unique values of material", &m_rtxState.debugging_mode, nullptr, Normal,
                               {
                                   "No Debug",
                                   "BaseColor",
                                   "Normal",
                                   "Metallic",
                                   "Emissive",
                                   "Alpha",
                                   "Roughness",
                                   "TexCoord",
                                   "Tangent",
                                   "Radiance",
                                   "Weight",
                                   "RayDir",
                                   "HeatMap",
                               });

    if(m_rtxState.debugging_mode == eHeatmap)
    {
      changed |= GuiH::Drag("Min Heat map", "Minimum timing value, below this value it will be blue",
                            &m_rtxState.minHeatmap, nullptr, Normal, 0, 1'000'000, 100);
      changed |= GuiH::Drag("Max Heat map", "Maximum timing value, above this value it will be red",
                            &m_rtxState.maxHeatmap, nullptr, Normal, 0, 1'000'000, 100);
    }
    return changed;
  });

  GuiH::Info("Frame", "", std::to_string(m_rtxState.frame), GuiH::Flags::Disabled);
  return changed;
}


bool SampleExample::guiTonemapper()
{
  static Offscreen::Tonemapper default_tm;  // default values
  auto&                        tm = m_offscreen.m_tonemapper;
  bool                         changed{false};

  changed |= GuiH::Slider("Exposure", "Scene Exposure", &tm.avgLum, &default_tm.avgLum, GuiH::Flags::Normal, 0.001f, 5.00f);
  changed |= GuiH::Slider("Brightness", "", &tm.brightness, &default_tm.brightness, GuiH::Flags::Normal, 0.0f, 2.0f);
  changed |= GuiH::Slider("Contrast", "", &tm.contrast, &default_tm.contrast, GuiH::Flags::Normal, 0.0f, 2.0f);
  changed |= GuiH::Slider("Saturation", "", &tm.saturation, &default_tm.saturation, GuiH::Flags::Normal, 0.0f, 5.0f);
  changed |= GuiH::Slider("Vignette", "", &tm.vignette, &default_tm.vignette, GuiH::Flags::Normal, 0.0f, 2.0f);

  return false;  // no need to restart the renderer
}

bool SampleExample::guiEnvironment()
{
  static SunAndSky dss = SunAndSky_default();  // default values
  bool             changed{false};

  changed |= ImGui::Checkbox("Use Sun & Sky", (bool*)&m_sunAndSky.in_use);
  changed |= GuiH::Slider("Exposure", "Intensity of the environment", &m_rtxState.hdrMultiplier, nullptr,
                          GuiH::Flags::Normal, 0.f, 5.f);

  // Adjusting the up with the camera
  nvmath::vec3f eye, center, up;
  CameraManip.getLookat(eye, center, up);
  m_sunAndSky.y_is_up = (up.y == 1);

  if(m_sunAndSky.in_use)
  {
    GuiH::Group<bool>("Sun", true, [&] {
      changed |= GuiH::Custom("Direction", "Sun Direction", [&] {
        float indent = ImGui::GetCursorPos().x;
        changed |= ImGui::DirectionGizmo("", &m_sunAndSky.sun_direction.x, true);
        ImGui::NewLine();
        ImGui::SameLine(indent);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        changed |= ImGui::InputFloat3("##IG", &m_sunAndSky.sun_direction.x);
        return changed;
      });
      changed |= GuiH::Slider("Disk Scale", "", &m_sunAndSky.sun_disk_scale, &dss.sun_disk_scale, GuiH::Flags::Normal, 0.f, 100.f);
      changed |= GuiH::Slider("Glow Intensity", "", &m_sunAndSky.sun_glow_intensity, &dss.sun_glow_intensity,
                              GuiH::Flags::Normal, 0.f, 5.f);
      changed |= GuiH::Slider("Disk Intensity", "", &m_sunAndSky.sun_disk_intensity, &dss.sun_disk_intensity,
                              GuiH::Flags::Normal, 0.f, 5.f);
      changed |= GuiH::Color("Night Color", "", &m_sunAndSky.night_color.x, &dss.night_color.x, GuiH::Flags::Normal);
      return changed;
    });

    GuiH::Group<bool>("Ground", true, [&] {
      changed |=
          GuiH::Slider("Horizon Height", "", &m_sunAndSky.horizon_height, &dss.horizon_height, GuiH::Flags::Normal, -1.f, 1.f);
      changed |= GuiH::Slider("Horizon Blur", "", &m_sunAndSky.horizon_blur, &dss.horizon_blur, GuiH::Flags::Normal, 0.f, 1.f);
      changed |= GuiH::Color("Ground Color", "", &m_sunAndSky.ground_color.x, &dss.ground_color.x, GuiH::Flags::Normal);
      changed |= GuiH::Slider("Haze", "", &m_sunAndSky.haze, &dss.haze, GuiH::Flags::Normal, 0.f, 15.f);
      return changed;
    });

    GuiH::Group<bool>("Other", false, [&] {
      changed |= GuiH::Drag("Multiplier", "", &m_sunAndSky.multiplier, &dss.multiplier, GuiH::Flags::Normal, 0.f,
                            std::numeric_limits<float>::max(), 2, "%5.5f");
      changed |= GuiH::Slider("Saturation", "", &m_sunAndSky.saturation, &dss.saturation, GuiH::Flags::Normal, 0.f, 1.f);
      changed |= GuiH::Slider("Red Blue Shift", "", &m_sunAndSky.redblueshift, &dss.redblueshift, GuiH::Flags::Normal, -1.f, 1.f);
      changed |= GuiH::Color("RGB Conversion", "", &m_sunAndSky.rgb_unit_conversion.x, &dss.rgb_unit_conversion.x,
                             GuiH::Flags::Normal);

      nvmath::vec3f eye, center, up;
      CameraManip.getLookat(eye, center, up);
      m_sunAndSky.y_is_up = up.y == 1;
      changed |= GuiH::Checkbox("Y is Up", "", (bool*)&m_sunAndSky.y_is_up, nullptr, GuiH::Flags::Disabled);
      return changed;
    });
  }

  return changed;
}

bool SampleExample::guiStatistics()
{
  ImGuiStyle& style    = ImGui::GetStyle();
  auto        pushItem = style.ItemSpacing;
  style.ItemSpacing.y  = -4;  // making the lines more dense

  auto& stats = m_scene.getStat();

  if(stats.nbCameras > 0)
    GuiH::Info("Cameras", "", FormatNumbers(stats.nbCameras));
  if(stats.nbImages > 0)
    GuiH::Info("Images", "", FormatNumbers(stats.nbImages) + " (" + FormatNumbers(stats.imageMem) + ")");
  if(stats.nbTextures > 0)
    GuiH::Info("Textures", "", FormatNumbers(stats.nbTextures));
  if(stats.nbMaterials > 0)
    GuiH::Info("Material", "", FormatNumbers(stats.nbMaterials));
  if(stats.nbSamplers > 0)
    GuiH::Info("Samplers", "", FormatNumbers(stats.nbSamplers));
  if(stats.nbNodes > 0)
    GuiH::Info("Nodes", "", FormatNumbers(stats.nbNodes));
  if(stats.nbMeshes > 0)
    GuiH::Info("Meshes", "", FormatNumbers(stats.nbMeshes));
  if(stats.nbLights > 0)
    GuiH::Info("Lights", "", FormatNumbers(stats.nbLights));
  if(stats.nbTriangles > 0)
    GuiH::Info("Triangles", "", FormatNumbers(stats.nbTriangles));
  if(stats.nbUniqueTriangles > 0)
    GuiH::Info("Unique Tri", "", FormatNumbers(stats.nbUniqueTriangles));
  GuiH::Info("Resolution", "", std::to_string(m_size.width) + "x" + std::to_string(m_size.height));

  style.ItemSpacing = pushItem;

  return false;
}

bool SampleExample::guiProfiler(nvvk::ProfilerVK& profiler)
{
  struct Info
  {
    vec2  statRender{0.0f, 0.0f};
    vec2  statTone{0.0f, 0.0f};
    float frameTime{0.0f};
  };
  static Info display;
  static Info collect;

  // Collecting data
  static float dirtyCnt = 0.0f;
  {
    dirtyCnt++;
    nvh::Profiler::TimerInfo info;
    profiler.getTimerInfo("Render", info);
    collect.statRender.x += float(info.gpu.average / 1000.0f);
    collect.statRender.y += float(info.cpu.average / 1000.0f);
    profiler.getTimerInfo("Tonemap", info);
    collect.statTone.x += float(info.gpu.average / 1000.0f);
    collect.statTone.y += float(info.cpu.average / 1000.0f);
    collect.frameTime += 1000.0f / ImGui::GetIO().Framerate;
  }

  // Averaging display of the data every 0.5 seconds
  static float dirtyTimer = 1.0f;
  dirtyTimer += ImGui::GetIO().DeltaTime;
  if(dirtyTimer >= 0.5f)
  {
    display.statRender = collect.statRender / dirtyCnt;
    display.statTone   = collect.statTone / dirtyCnt;
    display.frameTime  = collect.frameTime / dirtyCnt;
    dirtyTimer         = 0;
    dirtyCnt           = 0;
    collect            = Info{};
  }

  ImGui::Text("Frame     [ms]: %2.3f", display.frameTime);
  ImGui::Text("Render GPU/CPU [ms]: %2.3f  /  %2.3f", display.statRender.x, display.statRender.y);
  ImGui::Text("Tone+UI GPU/CPU [ms]: %2.3f  /  %2.3f", display.statTone.x, display.statTone.y);
  ImGui::ProgressBar(display.statRender.x / display.frameTime);


  return false;
}


bool SampleExample::guiGpuMeasures()
{
  if(g_nvml.isValid() == false)
    ImGui::Text("NVML wasn't loaded");

  auto memoryNumbers = [](float n, int precision = 3) -> std::string {
    std::vector<std::string> t{" KB", " MB", " GB", " TB"};
    int                      level{0};
    while(n > 1024)
    {
      n = n / 1024;
      level++;
    }
    assert(level < 3);
    std::stringstream o;
    o << std::setprecision(precision) << std::fixed << n << t[level];

    return o.str();
  };

  uint32_t offset = g_nvml.getOffset();

  for(uint32_t g = 0; g < g_nvml.nbGpu(); g++)  // Number of gpu
  {
    const auto& i = g_nvml.getInfo(g);
    const auto& m = g_nvml.getMeasures(g);

    std::stringstream o;
    o << "Driver: " << i.driver_model << "\n"                                                                //
      << "Memory: " << memoryNumbers(m.memory[offset]) << "/" << memoryNumbers(float(i.max_mem), 0) << "\n"  //
      << "Load: " << m.load[offset];

    float                mem = m.memory[offset] / float(i.max_mem) * 100.f;
    std::array<char, 64> desc;
    sprintf(desc.data(), "%s: \n- Load: %2.0f%s \n- Mem: %2.0f%s", i.name.c_str(), m.load[offset], "%%", mem, "%%");
    ImGuiH::Control::Custom(desc.data(), o.str().c_str(), [&]() {
      ImGui::ImPlotMulti datas[2];
      datas[0].plot_type     = static_cast<ImGuiPlotType>(ImGuiPlotType_Area);
      datas[0].name          = "Load";
      datas[0].color         = ImColor(0.07f, 0.9f, 0.06f, 1.0f);
      datas[0].thickness     = 1.5;
      datas[0].data          = m.load.data();
      datas[0].values_count  = (int)m.load.size();
      datas[0].values_offset = offset + 1;
      datas[0].scale_min     = 0;
      datas[0].scale_max     = 100;

      datas[1].plot_type     = ImGuiPlotType_Histogram;
      datas[1].name          = "Mem";
      datas[1].color         = ImColor(0.06f, 0.6f, 0.97f, 0.8f);
      datas[1].thickness     = 2.0;
      datas[1].data          = m.memory.data();
      datas[1].values_count  = (int)m.memory.size();
      datas[1].values_offset = offset + 1;
      datas[1].scale_min     = 0;
      datas[1].scale_max     = float(i.max_mem);


      std::string overlay = std::to_string((int)m.load[offset]) + " %";
      ImGui::PlotMultiEx("##NoName", 2, datas, overlay.c_str(), ImVec2(0, 100));

      return false;
    });

    ImGuiH::Control::Custom("CPU", "", [&]() {
      ImGui::ImPlotMulti datas[1];
      datas[0].plot_type     = ImGuiPlotType_Lines;
      datas[0].name          = "CPU";
      datas[0].color         = ImColor(0.96f, 0.96f, 0.07f, 1.0f);
      datas[0].thickness     = 1.0;
      datas[0].data          = g_nvml.getSysInfo().cpu.data();
      datas[0].values_count  = (int)g_nvml.getSysInfo().cpu.size();
      datas[0].values_offset = offset + 1;
      datas[0].scale_min     = 0;
      datas[0].scale_max     = 100;

      std::string overlay = std::to_string((int)m.load[offset]) + " %";
      ImGui::PlotMultiEx("##NoName", 1, datas, nullptr, ImVec2(0, 0));

      return false;
    });
  }


  return false;
}


//--------------------------------------------------------------------------------------------------
// Display a static window when loading assets
//
void SampleExample::showBusyWindow()
{
  static int   nb_dots   = 0;
  static float deltaTime = 0;
  bool         show      = true;
  size_t       width     = 270;
  size_t       height    = 60;

  deltaTime += ImGui::GetIO().DeltaTime;
  if(deltaTime > .25)
  {
    deltaTime = 0;
    nb_dots   = ++nb_dots % 10;
  }

  ImGui::SetNextWindowSize(ImVec2(float(width), float(height)));
  ImGui::SetNextWindowPos(ImVec2(float(m_size.width - width) * 0.5f, float(m_size.height - height) * 0.5f));

  ImGui::SetNextWindowBgAlpha(0.75f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
  if(ImGui::Begin("##notitle", &show,
                  ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                      | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMouseInputs))
  {
    ImVec2 available = ImGui::GetContentRegionAvail();

    ImVec2 text_size = ImGui::CalcTextSize(m_busyReasonText.c_str(), nullptr, false, available.x);

    ImVec2 pos = ImGui::GetCursorPos();
    pos.x += (available.x - text_size.x) * 0.5f;
    pos.y += (available.y - text_size.y) * 0.5f;

    ImGui::SetCursorPos(pos);
    ImGui::TextWrapped((m_busyReasonText + std::string(nb_dots, '.')).c_str());
  }
  ImGui::PopStyleVar();
  ImGui::End();
}


//////////////////////////////////////////////////////////////////////////
// Keyboard / Drag and Drop
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
// - Home key: fit all, the camera will move to see the entire scene bounding box
// - Space: Trigger ray picking and set the interest point at the intersection
//          also return all information under the cursor
//
void SampleExample::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvk::AppBaseVk::onKeyboard(key, scancode, action, mods);

  if(action == GLFW_RELEASE)
    return;

  if(key == GLFW_KEY_HOME)
  {
    // Set the camera as to see the model
    fitCamera(m_scene.getScene().m_dimensions.min, m_scene.getScene().m_dimensions.max, false);
  }
  else if(key == GLFW_KEY_SPACE)
  {
    screenPicking();
  }
  else if(key == GLFW_KEY_R)
  {
    resetFrame();
  }
}

void SampleExample::screenPicking()
{
  double x, y;
  glfwGetCursorPos(m_window, &x, &y);

  // Set the camera as to see the model
  nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = sc.createCommandBuffer();

  const float aspectRatio = m_renderRegion.extent.width / static_cast<float>(m_renderRegion.extent.height);
  auto        view        = CameraManip.getMatrix();
  auto        proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);

  nvvk::RayPickerKHR::PickInfo pickInfo;
  pickInfo.pickX          = float(x - m_renderRegion.offset.x) / float(m_renderRegion.extent.width);
  pickInfo.pickY          = float(y - m_renderRegion.offset.y) / float(m_renderRegion.extent.height);
  pickInfo.modelViewInv   = nvmath::invert(view);
  pickInfo.perspectiveInv = nvmath::invert(proj);


  m_picker.run(cmdBuf, pickInfo);
  sc.submitAndWait(cmdBuf);

  nvvk::RayPickerKHR::PickResult pr = m_picker.getResult();

  if(pr.instanceID == ~0)
  {
    LOGI("Nothing Hit\n");
    return;
  }

  nvmath::vec3 worldPos = pr.worldRayOrigin + pr.worldRayDirection * pr.hitT;
  // Set the interest position
  nvmath::vec3f eye, center, up;
  CameraManip.getLookat(eye, center, up);
  CameraManip.setLookat(eye, worldPos, up, false);


  auto& prim = m_scene.getScene().m_primMeshes[pr.instanceCustomIndex];
  LOGI("Hit(%d): %s\n", pr.instanceCustomIndex, prim.name.c_str());
  LOGI(" - PrimId(%d)\n", pr.primitiveID);
}

void SampleExample::onFileDrop(const char* filename)
{
  loadAssets(filename);
}

//--------------------------------------------------------------------------------------------------
// Window callback when the mouse move
// - Handling ImGui and a default camera
//
void SampleExample::onMouseMotion(int x, int y)
{
  AppBaseVk::onMouseMotion(x, y);

  if(ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard)
    return;

  if(m_inputs.lmb || m_inputs.rmb || m_inputs.mmb)
  {
    m_descaling = true;
  }
}

void SampleExample::onMouseButton(int button, int action, int mods)
{
  AppBaseVk::onMouseButton(button, action, mods);
  if((m_inputs.lmb || m_inputs.rmb || m_inputs.mmb) == false && action == GLFW_RELEASE && m_descaling == true)
  {
    m_descaling = false;
    resetFrame();
  }

  auto& IO = ImGui::GetIO();
  if(IO.MouseDownWasDoubleClick[0])
  {
    screenPicking();
  }
}

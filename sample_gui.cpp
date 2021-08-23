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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 *  This implements all graphical user interface of SampleExample.
 */


#include <bitset>  // std::bitset
#include <iomanip>
#include <sstream>

#include "nvmath/nvmath.h"
#include "nvmath/nvmath_glsltypes.h"
using namespace nvmath;

#include "imgui_helper.h"
#include "imgui_orient.h"
#include "rtx_pipeline.hpp"
#include "sample_example.hpp"
#include "sample_gui.hpp"
#include "tools.hpp"

#include "nvml_monitor.hpp"
#ifdef _WIN32
#include <commdlg.h>
#endif  // _WIN32

using GuiH = ImGuiH::Control;

#if defined(NVP_SUPPORTS_NVML)
extern NvmlMonitor g_nvml;  // GPU load and memory
#endif

//--------------------------------------------------------------------------------------------------
// Main rendering function for all
//
void SampleGUI::render(nvvk::ProfilerVK& profiler)
{
  // Show UI panel window.
  float panelAlpha = 1.0f;
  if(_se->showGui())
  {
    ImGuiH::Control::style.ctrlPerc = 0.55f;
    ImGuiH::Panel::Begin(ImGuiH::Panel::Side::Right, panelAlpha);

    using Gui = ImGuiH::Control;
    bool changed{false};

    if(ImGui::CollapsingHeader("Camera" /*, ImGuiTreeNodeFlags_DefaultOpen*/))
      changed |= guiCamera();
    if(ImGui::CollapsingHeader("Ray Tracing" /*, ImGuiTreeNodeFlags_DefaultOpen*/))
      changed |= guiRayTracing();
    if(ImGui::CollapsingHeader("Tonemapper" /*, ImGuiTreeNodeFlags_DefaultOpen*/))
      changed |= guiTonemapper();
    if(ImGui::CollapsingHeader("Environment" /*, ImGuiTreeNodeFlags_DefaultOpen*/))
      changed |= guiEnvironment();
    if(ImGui::CollapsingHeader("Stats"))
    {
      Gui::Group<bool>("Scene Info", false, [&] { return guiStatistics(); });
      Gui::Group<bool>("Profiler", false, [&] { return guiProfiler(profiler); });
      Gui::Group<bool>("Plot", false, [&] { return guiGpuMeasures(); });
    }
    ImGui::TextWrapped("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                       ImGui::GetIO().Framerate);

    if(changed)
    {
      _se->resetFrame();
    }

    ImGui::End();  // ImGui::Panel::end()
  }

  // Rendering region is different if the side panel is visible
  if(panelAlpha >= 1.0f && _se->showGui())
  {
    ImVec2 pos, size;
    ImGuiH::Panel::CentralDimension(pos, size);
    _se->setRenderRegion(VkRect2D{VkOffset2D{static_cast<int32_t>(pos.x), static_cast<int32_t>(pos.y)},
                                  VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)}});
  }
  else
  {
    _se->setRenderRegion(VkRect2D{{}, _se->getSize()});
  }
}

//--------------------------------------------------------------------------------------------------
//
//
bool SampleGUI::guiCamera()
{
  bool changed{false};
  changed |= ImGuiH::CameraWidget();
  auto& cam = _se->m_scene.getCamera();
  changed |= GuiH::Slider("Aperture", "", &cam.aperture, nullptr, ImGuiH::Control::Flags::Normal, 0.0f, 0.5f);

  return changed;
}

//--------------------------------------------------------------------------------------------------
//
//
bool SampleGUI::guiRayTracing()
{
  auto  Normal = ImGuiH::Control::Flags::Normal;
  bool  changed{false};
  auto& rtxState(_se->m_rtxState);

  changed |= GuiH::Slider("Max Ray Depth", "", &rtxState.maxDepth, nullptr, Normal, 1, 10);
  changed |= GuiH::Slider("Samples Per Frame", "", &rtxState.maxSamples, nullptr, Normal, 1, 10);
  changed |= GuiH::Slider("Max Iteration ", "", &_se->m_maxFrames, nullptr, Normal, 1, 1000);
  changed |= GuiH::Slider("De-scaling ",
                          "Reduce resolution while navigating.\n"
                          "Speeding up rendering while camera moves.\n"
                          "Value of 1, will not de-scale",
                          &_se->m_descalingLevel, nullptr, Normal, 1, 8);

  changed |= GuiH::Selection("Pbr Mode", "PBR material model", &rtxState.pbrMode, nullptr, Normal, {"Disney", "Gltf"});

  static bool bAnyHit = true;
  if(GuiH::Checkbox("Enable AnyHit",
                    "AnyHit is used for double sided, cutout opacity, but can be slower when all objects are opaque", &bAnyHit, nullptr))
  {
    auto rtx = dynamic_cast<RtxPipeline*>(_se->m_pRender[_se->m_rndMethod]);
    vkDeviceWaitIdle(_se->m_device);  // cannot run while changing this
    rtx->useAnyHit(bAnyHit);
    changed = true;
  }

  GuiH::Group<bool>("Debugging", false, [&] {
    changed |= GuiH::Selection("Debug Mode", "Display unique values of material", &rtxState.debugging_mode, nullptr, Normal,
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

    if(rtxState.debugging_mode == eHeatmap)
    {
      changed |= GuiH::Drag("Min Heat map", "Minimum timing value, below this value it will be blue",
                            &rtxState.minHeatmap, nullptr, Normal, 0, 1'000'000, 100);
      changed |= GuiH::Drag("Max Heat map", "Maximum timing value, above this value it will be red",
                            &rtxState.maxHeatmap, nullptr, Normal, 0, 1'000'000, 100);
    }
    return changed;
  });

  SampleExample::RndMethod method = _se->renderMethod;
  if(GuiH::Selection<int>("Rendering Pipeline", "Choose the type of rendering", (int*)&method, nullptr,
                          GuiH::Control::Flags::Normal, {"Rtx", "Compute"}))
  {
    _se->createRender(method);
    _se->renderMethod = method;
    changed           = true;
  }

  GuiH::Info("Frame", "", std::to_string(rtxState.frame), GuiH::Flags::Disabled);
  return changed;
}


bool SampleGUI::guiTonemapper()
{
  static Tonemapper default_tm{1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, {1.f, 1.f}, 0, .5f, .5f};
  auto&             tm = _se->m_offscreen.m_tonemapper;
  bool              changed{false};
  std::bitset<8>    b(tm.autoExposure);

  bool autoExposure = b.test(0);

  changed |= GuiH::Checkbox("Auto Exposure", "Adjust exposure", (bool*)&autoExposure);
  changed |= GuiH::Slider("Exposure", "Scene Exposure", &tm.avgLum, &default_tm.avgLum, GuiH::Flags::Normal, 0.001f, 5.00f);
  changed |= GuiH::Slider("Brightness", "", &tm.brightness, &default_tm.brightness, GuiH::Flags::Normal, 0.0f, 2.0f);
  changed |= GuiH::Slider("Contrast", "", &tm.contrast, &default_tm.contrast, GuiH::Flags::Normal, 0.0f, 2.0f);
  changed |= GuiH::Slider("Saturation", "", &tm.saturation, &default_tm.saturation, GuiH::Flags::Normal, 0.0f, 5.0f);
  changed |= GuiH::Slider("Vignette", "", &tm.vignette, &default_tm.vignette, GuiH::Flags::Normal, 0.0f, 2.0f);


  if(autoExposure)
  {
    bool localExposure = b.test(1);
    GuiH::Group<bool>("Auto Settings", true, [&] {
      changed |= GuiH::Checkbox("Local", "", &localExposure);
      changed |= GuiH::Slider("Burning White", "", &tm.Ywhite, &default_tm.Ywhite, GuiH::Flags::Normal, 0.0f, 1.0f);
      changed |= GuiH::Slider("Brightness", "", &tm.key, &default_tm.key, GuiH::Flags::Normal, 0.0f, 1.0f);
      b.set(1, localExposure);
      return changed;
    });
  }
  b.set(0, autoExposure);
  tm.autoExposure = b.to_ulong();

  return false;  // no need to restart the renderer
}

//--------------------------------------------------------------------------------------------------
//
//
bool SampleGUI::guiEnvironment()
{
  static SunAndSky dss = SunAndSky_default();  // default values
  bool             changed{false};
  auto&            sunAndSky(_se->m_sunAndSky);

  changed |= ImGui::Checkbox("Use Sun & Sky", (bool*)&sunAndSky.in_use);
  changed |= GuiH::Slider("Exposure", "Intensity of the environment", &_se->m_rtxState.hdrMultiplier, nullptr,
                          GuiH::Flags::Normal, 0.f, 5.f);

  // Adjusting the up with the camera
  nvmath::vec3f eye, center, up;
  CameraManip.getLookat(eye, center, up);
  sunAndSky.y_is_up = (up.y == 1);

  if(sunAndSky.in_use)
  {
    GuiH::Group<bool>("Sun", true, [&] {
      changed |= GuiH::Custom("Direction", "Sun Direction", [&] {
        float indent = ImGui::GetCursorPos().x;
        changed |= ImGui::DirectionGizmo("", &sunAndSky.sun_direction.x, true);
        ImGui::NewLine();
        ImGui::SameLine(indent);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        changed |= ImGui::InputFloat3("##IG", &sunAndSky.sun_direction.x);
        return changed;
      });
      changed |= GuiH::Slider("Disk Scale", "", &sunAndSky.sun_disk_scale, &dss.sun_disk_scale, GuiH::Flags::Normal, 0.f, 100.f);
      changed |= GuiH::Slider("Glow Intensity", "", &sunAndSky.sun_glow_intensity, &dss.sun_glow_intensity,
                              GuiH::Flags::Normal, 0.f, 5.f);
      changed |= GuiH::Slider("Disk Intensity", "", &sunAndSky.sun_disk_intensity, &dss.sun_disk_intensity,
                              GuiH::Flags::Normal, 0.f, 5.f);
      changed |= GuiH::Color("Night Color", "", &sunAndSky.night_color.x, &dss.night_color.x, GuiH::Flags::Normal);
      return changed;
    });

    GuiH::Group<bool>("Ground", true, [&] {
      changed |= GuiH::Slider("Horizon Height", "", &sunAndSky.horizon_height, &dss.horizon_height, GuiH::Flags::Normal, -1.f, 1.f);
      changed |= GuiH::Slider("Horizon Blur", "", &sunAndSky.horizon_blur, &dss.horizon_blur, GuiH::Flags::Normal, 0.f, 1.f);
      changed |= GuiH::Color("Ground Color", "", &sunAndSky.ground_color.x, &dss.ground_color.x, GuiH::Flags::Normal);
      changed |= GuiH::Slider("Haze", "", &sunAndSky.haze, &dss.haze, GuiH::Flags::Normal, 0.f, 15.f);
      return changed;
    });

    GuiH::Group<bool>("Other", false, [&] {
      changed |= GuiH::Drag("Multiplier", "", &sunAndSky.multiplier, &dss.multiplier, GuiH::Flags::Normal, 0.f,
                            std::numeric_limits<float>::max(), 2, "%5.5f");
      changed |= GuiH::Slider("Saturation", "", &sunAndSky.saturation, &dss.saturation, GuiH::Flags::Normal, 0.f, 1.f);
      changed |= GuiH::Slider("Red Blue Shift", "", &sunAndSky.redblueshift, &dss.redblueshift, GuiH::Flags::Normal, -1.f, 1.f);
      changed |= GuiH::Color("RGB Conversion", "", &sunAndSky.rgb_unit_conversion.x, &dss.rgb_unit_conversion.x, GuiH::Flags::Normal);

      nvmath::vec3f eye, center, up;
      CameraManip.getLookat(eye, center, up);
      sunAndSky.y_is_up = up.y == 1;
      changed |= GuiH::Checkbox("Y is Up", "", (bool*)&sunAndSky.y_is_up, nullptr, GuiH::Flags::Disabled);
      return changed;
    });
  }

  return changed;
}

//--------------------------------------------------------------------------------------------------
//
//
bool SampleGUI::guiStatistics()
{
  ImGuiStyle& style    = ImGui::GetStyle();
  auto        pushItem = style.ItemSpacing;
  style.ItemSpacing.y  = -4;  // making the lines more dense

  auto& stats = _se->m_scene.getStat();

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
  GuiH::Info("Resolution", "", std::to_string(_se->m_size.width) + "x" + std::to_string(_se->m_size.height));

  style.ItemSpacing = pushItem;

  return false;
}

//--------------------------------------------------------------------------------------------------
//
//
bool SampleGUI::guiProfiler(nvvk::ProfilerVK& profiler)
{
  struct Info
  {
    vec2  statRender{0.0f, 0.0f};
    vec2  statTone{0.0f, 0.0f};
    float frameTime{0.0f};
  };
  static Info  display;
  static Info  collect;
  static float mipmapGen{0.f};

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

    if(_se->m_offscreen.m_tonemapper.autoExposure == 1)
    {
      profiler.getTimerInfo("Mipmap", info);
      mipmapGen = float(info.gpu.average / 1000.0f);
      //LOGI("Mipmap Generation: %.2fms\n", info.gpu.average / 1000.0f);
    }
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
  if(_se->m_offscreen.m_tonemapper.autoExposure == 1)
    ImGui::Text("Mipmap Gen: %2.3fms", mipmapGen);
  ImGui::ProgressBar(display.statRender.x / display.frameTime);


  return false;
}

//--------------------------------------------------------------------------------------------------
//
//
bool SampleGUI::guiGpuMeasures()
{
#if defined(NVP_SUPPORTS_NVML)
  if(g_nvml.isValid() == false)
    ImGui::Text("NVML wasn't loaded");

  auto memoryNumbers = [](float n) {  // Memory numbers from nvml are in KB
    static const std::vector<char*> t{" KB", " MB", " GB", " TB"};
    static char                     s[16];
    int                             level{0};
    while(n > 1000)
    {
      n = n / 1000;
      level++;
    }
    sprintf(s, "%.3f %s", n, t[level]);
    return s;
  };

  uint32_t offset = g_nvml.getOffset();

  for(uint32_t g = 0; g < g_nvml.nbGpu(); g++)  // Number of gpu
  {
    const auto& i = g_nvml.getInfo(g);
    const auto& m = g_nvml.getMeasures(g);

    float                mem = m.memory[offset] / float(i.max_mem) * 100.f;
    std::array<char, 64> desc;
    sprintf(desc.data(), "%s: \n- Load: %2.0f%s \n- Mem: %2.0f%s", i.name.c_str(), m.load[offset], "%", mem, "%");
    ImGui::Text("%s \n- Load: %2.0f%s \n- Mem: %2.0f%s %s", i.name.c_str(), m.load[offset], "%", mem, "%",
                memoryNumbers(m.memory[offset]));
    {
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
      ImGui::PlotMultiEx("##NoName", 2, datas, overlay.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 150));
    }


    ImGui::Text("CPU");
    {
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
    }
  }
#else
  ImGui::Text("NVML wasn't loaded");
#endif
  return false;
}


//--------------------------------------------------------------------------------------------------
// This is displaying information in the titlebar
//
void SampleGUI::titleBar()
{
  static float dirtyTimer = 0.0f;

  dirtyTimer += ImGui::GetIO().DeltaTime;
  if(dirtyTimer > 1)
  {
    std::stringstream o;
    o << "VK glTF Viewer";
    o << " | " << _se->m_scene.getSceneName();                                                   // Scene name
    o << " | " << _se->m_renderRegion.extent.width << "x" << _se->m_renderRegion.extent.height;  // resolution
    o << " | " << static_cast<int>(ImGui::GetIO().Framerate)                                     // FPS / ms
      << " FPS / " << std::setprecision(3) << 1000.F / ImGui::GetIO().Framerate << "ms";
#if defined(NVP_SUPPORTS_NVML)
    if(g_nvml.isValid())  // Graphic card, driver
    {
      const auto& i = g_nvml.getInfo(0);
      o << " | " << i.name;
      o << " | " << g_nvml.getSysInfo().driverVersion;
    }
#endif
    if(_se->m_rndMethod != SampleExample::eNone && _se->m_pRender[_se->m_rndMethod] != nullptr)
      o << " | " << _se->m_pRender[_se->m_rndMethod]->name();
    glfwSetWindowTitle(_se->m_window, o.str().c_str());
    dirtyTimer = 0;
  }
}

//--------------------------------------------------------------------------------------------------
//
//
void SampleGUI::menuBar()
{
  auto openFilename = [](const char* filter) {
#ifdef _WIN32
    char         filename[MAX_PATH] = {0};
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
        _se->loadAssets(openFilename("GLTF Files\0*.gltf;*.glb\0\0").c_str());
      if(ImGui::MenuItem("Open HDR Environment"))
        _se->loadAssets(openFilename("HDR Files\0*.hdr\0\0").c_str());
      ImGui::Separator();
      if(ImGui::MenuItem("Quit", "ESC"))
        glfwSetWindowShouldClose(_se->m_window, 1);
      ImGui::EndMenu();
    }

    if(ImGui::BeginMenu("Tools"))
    {
      ImGui::MenuItem("Settings", "F10", &_se->m_show_gui);
      ImGui::MenuItem("Axis", nullptr, &_se->m_showAxis);
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }
}


//--------------------------------------------------------------------------------------------------
// Display a static window when loading assets
//
void SampleGUI::showBusyWindow()
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
  ImGui::SetNextWindowPos(ImVec2(float(_se->m_size.width - width) * 0.5f, float(_se->m_size.height - height) * 0.5f));

  ImGui::SetNextWindowBgAlpha(0.75f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
  if(ImGui::Begin("##notitle", &show,
                  ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                      | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMouseInputs))
  {
    ImVec2 available = ImGui::GetContentRegionAvail();

    ImVec2 text_size = ImGui::CalcTextSize(_se->m_busyReasonText.c_str(), nullptr, false, available.x);

    ImVec2 pos = ImGui::GetCursorPos();
    pos.x += (available.x - text_size.x) * 0.5f;
    pos.y += (available.y - text_size.y) * 0.5f;

    ImGui::SetCursorPos(pos);
    ImGui::TextWrapped((_se->m_busyReasonText + std::string(nb_dots, '.')).c_str());
  }
  ImGui::PopStyleVar();
  ImGui::End();
}

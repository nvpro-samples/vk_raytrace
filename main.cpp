/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


#include <array>
#include <filesystem>
#include <iostream>
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "imgui.h"
#include "imgui_helper.h"
#include "imgui_impl_glfw.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/inputparser.h"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "sample_example.hpp"


//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;

// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  InputParser parser(argc, argv);
  std::string sceneFile   = parser.getString("-f", "media/robot-toon.gltf");
  std::string hdrFilename = parser.getString("-e", R"(media/std_env.hdr)");


  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat({2.0, 2.0, -5.0}, {-1.0, 2.0, -1.0}, {0.000, 1.000, 0.000});

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(argv[0], PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      PROJECT_ABSDIRECTORY,
      PROJECT_ABSDIRECTORY "..",
      NVPSystem::exePath(),
      NVPSystem::exePath() + "..",
      NVPSystem::exePath() + std::string(PROJECT_NAME),
  };


  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo(true);
  contextInfo.setVersion(1, 2);
  contextInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef WIN32
  contextInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
  contextInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
  contextInfo.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeature;
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);
  vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature;
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);
  vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures;
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

  contextInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);


  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.ignoreDebugMessage(0x99fb7dfd);  // dstAccelerationStructure
  vkctx.ignoreDebugMessage(0x45e8716f);  // dstAccelerationStructure
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);


  // Create example
  SampleExample sample;

  // Window need to be opened to get the surface on which to draw
  const vk::SurfaceKHR surface = sample.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  sample.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex, vkctx.m_queueC.familyIndex);
  sample.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  sample.createDepthBuffer();
  sample.createRenderPass();
  sample.createFrameBuffers();

  // Setup Imgui

  sample.initGUI(0);  // Using sub-pass 0
  sample.createOffscreenRender();

  // Creation of the example
  sample.loadEnvironmentHdr(nvh::findFile(hdrFilename, defaultSearchPaths, true));
  sample.m_busy = true;
  std::thread([&] {
    sample.m_busyReasonText = "Loading Scene";
    sample.loadScene(nvh::findFile(sceneFile, defaultSearchPaths, true));
    sample.createUniformBuffer();
    sample.createDescriptorSetLayout();
    sample.m_busy = false;
  }).detach();


  nvmath::vec4f clearColor(1);


  SampleExample::RndMethod renderMethod = SampleExample::eRtxCore;


  sample.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);


  nvvk::ProfilerVK profiler;
  std::string      profilerStats;

  profiler.init(vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  profiler.setLabelUsage(true);  // depends on VK_EXT_debug_utils

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(sample.isMinimized())
      continue;

    profiler.beginFrame();

    // Start rendering the scene
    sample.prepareFrame();
    sample.updateFrame();

    // Start command buffer of this frame
    auto                     curFrame = sample.getCurFrame();
    const vk::CommandBuffer& cmdBuf   = sample.getCommandBuffers()[curFrame];

    cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    sample.titleBar();
    sample.menuBar();

    // Show UI window.
    if(sample.showGui())
    {
      ImGuiH::Control::style.ctrlPerc = 0.55f;
      ImGuiH::Panel::Begin(ImGuiH::Panel::Side::Right);

      using Gui = ImGuiH::Control;
      bool changed{false};
      //      changed |= Gui::Selection<int>("Rendering Mode\n", "Choose the type of rendering", (int*)&renderMethod, nullptr,
      //                                     Gui::Flags::Normal, {"RtCore"});
      if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
        changed |= sample.guiCamera();
      if(ImGui::CollapsingHeader("Ray Tracing", ImGuiTreeNodeFlags_DefaultOpen))
        changed |= sample.guiRayTracing();
      if(ImGui::CollapsingHeader("Tonemapper", ImGuiTreeNodeFlags_DefaultOpen))
        changed |= sample.guiTonemapper();
      if(ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen))
        changed |= sample.guiEnvironment();
      if(ImGui::CollapsingHeader("Stats"))
      {
        changed |= sample.guiStatistics();
        Gui::Group<bool>("Profiler", false, [&] { return sample.guiProfiler(profiler); });
        sample.guiGpuMeasures();
      }
      ImGui::TextWrapped("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                         ImGui::GetIO().Framerate);

      // generic print to string

      if(changed)
        sample.resetFrame();
      ImGui::End();  // ImGui::Panel::end()
    }


    // Clearing screen
    vk::ClearValue clearValues[2];
    clearValues[0].setColor(std::array<float, 4>({clearColor[0], clearColor[1], clearColor[2], clearColor[3]}));
    clearValues[1].setDepthStencil({1.0f, 0});

    // Offscreen render pass
    if(sample.isBusy() == false)
    {
      auto sec = profiler.timeRecurring("Render", cmdBuf);
      // Updating camera buffer
      sample.updateUniformBuffer(cmdBuf);

      // Rendering Scene
      sample.render(renderMethod, cmdBuf, profiler);
    }
    else
      sample.showBusyWindow();

    // Rendering pass: tone mapper, UI
    {
      auto sec = profiler.timeRecurring("Tonemap", cmdBuf);

      vk::RenderPassBeginInfo postRenderPassBeginInfo;
      postRenderPassBeginInfo.setClearValueCount(2);
      postRenderPassBeginInfo.setPClearValues(clearValues);
      postRenderPassBeginInfo.setRenderPass(sample.getRenderPass());
      postRenderPassBeginInfo.setFramebuffer(sample.getFramebuffers()[curFrame]);
      postRenderPassBeginInfo.setRenderArea({{}, sample.getSize()});

      cmdBuf.beginRenderPass(postRenderPassBeginInfo, vk::SubpassContents::eInline);
      // Rendering tonemapper
      sample.drawPost(cmdBuf);

      ImGui::Render();
      ImGui::RenderDrawDataVK(cmdBuf, ImGui::GetDrawData());
      cmdBuf.endRenderPass();
    }

    profiler.endFrame();

    // Submit for display
    cmdBuf.end();
    sample.submitFrame();

    CameraManip.updateAnim();
  }

  // Cleanup
  sample.getDevice().waitIdle();
  sample.destroyResources();
  sample.destroy();
  profiler.deinit();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

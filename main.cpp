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


#include <thread>

#include "backends/imgui_impl_glfw.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/inputparser.h"
#include "nvvk/context_vk.hpp"
#include "sample_example.hpp"


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
  std::string sceneFile   = parser.getString("-f", "robot_toon/robot-toon.gltf");
  std::string hdrFilename = parser.getString("-e", "std_env.hdr");

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(glfwInit() == GLFW_FALSE)
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat({2.0, 2.0, -5.0}, {-1.0, 2.0, -1.0}, {0.000, 1.000, 0.000});

  // Setup Vulkan
  if(glfwVulkanSupported() == GLFW_FALSE)
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // Setup logging file
  //  nvprintSetLogFileName(PROJECT_NAME "_log.txt")

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_NAME,
      NVPSystem::exePath() + R"(media)",
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY,
  };

  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeature);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);


  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);  // Find all compatible devices
  assert(!compatibleDevices.empty());
  vkctx.initDevice(compatibleDevices[0], contextInfo);  // Use first compatible device

  //
  SampleExample sample;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = sample.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);
  sample.setupGlfwCallbacks(window);

  // Create example
  sample.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex,
               vkctx.m_queueC.familyIndex, vkctx.m_queueT.familyIndex);
  sample.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  sample.createDepthBuffer();
  sample.createRenderPass();
  sample.createFrameBuffers();

  // Setup Imgui
  sample.initGUI();
  sample.createOffscreenRender();
  ImGui_ImplGlfw_InitForVulkan(window, true);


  // Creation of the example - loading scene in separate thread
  sample.loadEnvironmentHdr(nvh::findFile(hdrFilename, defaultSearchPaths, true));
  sample.m_busy = true;
  std::thread([&] {
    sample.m_busyReasonText = "Loading Scene";
    sample.loadScene(nvh::findFile(sceneFile, defaultSearchPaths, true));
    sample.createUniformBuffer();
    sample.createDescriptorSetLayout();
    sample.m_busy = false;
  }).detach();


  // It is possible to have various back-ends
  SampleExample::RndMethod renderMethod = SampleExample::eRtxPipeline;

  // Profiler measure the execution time on the GPU
  nvvk::ProfilerVK profiler;
  std::string      profilerStats;
  profiler.init(vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  profiler.setLabelUsage(true);  // depends on VK_EXT_debug_utils

  // Main loop
  while(glfwWindowShouldClose(window) == GLFW_FALSE)
  {
    glfwPollEvents();
    if(sample.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();


    // Start rendering the scene
    profiler.beginFrame();
    sample.prepareFrame();
    sample.updateFrame();

    // Start command buffer of this frame
    auto                   curFrame = sample.getCurFrame();
    const VkCommandBuffer& cmdBuf   = sample.getCommandBuffers()[curFrame];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // UI
    sample.titleBar();
    sample.menuBar();

    // Show UI panel window.
    float panelAlpha = 1.0f;
    if(sample.showGui())
    {
      ImGuiH::Control::style.ctrlPerc = 0.55f;
      ImGuiH::Panel::Begin(ImGuiH::Panel::Side::Right, panelAlpha);

      using Gui = ImGuiH::Control;
      bool changed{false};
      changed |= Gui::Selection<int>("Rendering Mode\n", "Choose the type of rendering", (int*)&renderMethod, nullptr,
                                     Gui::Flags::Normal, {"RtxPipeline", "RayQuery"});
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
        Gui::Group<bool>("Scene Info", false, [&] { return sample.guiStatistics(); });
        Gui::Group<bool>("Profiler", false, [&] { return sample.guiProfiler(profiler); });
        Gui::Group<bool>("Plot", false, [&] { return sample.guiGpuMeasures(); });
      }
      ImGui::TextWrapped("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                         ImGui::GetIO().Framerate);

      if(changed)
      {
        sample.resetFrame();
      }

      ImGui::End();  // ImGui::Panel::end()
    }

    // Rendering region is different if the side panel is visible
    if(panelAlpha >= 1.0f && sample.showGui())
    {
      ImVec2 pos, size;
      ImGuiH::Panel::CentralDimension(pos, size);
      sample.setRenderRegion(VkRect2D{VkOffset2D{static_cast<int32_t>(pos.x), static_cast<int32_t>(pos.y)},
                                      VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)}});
    }
    else
    {
      sample.setRenderRegion(VkRect2D{{}, sample.getSize()});
    }

    // Clearing screen
    std::array<VkClearValue, 2> clearValues;
    clearValues[0].color        = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    // Offscreen render pass
    if(sample.isBusy() == false)
    {
      auto sec = profiler.timeRecurring("Render", cmdBuf);

      sample.updateUniformBuffer(cmdBuf);             // Updating camera buffer
      sample.render(renderMethod, cmdBuf, profiler);  // Rendering Scene
    }
    else
    {
      sample.showBusyWindow();  // Busy while loading scene
    }

    // Rendering pass: tone mapper, UI
    {
      auto sec = profiler.timeRecurring("Tonemap", cmdBuf);

      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = 2;
      postRenderPassBeginInfo.pClearValues    = clearValues.data();
      postRenderPassBeginInfo.renderPass      = sample.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = sample.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{}, sample.getSize()};

      vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      // Rendering tonemapper
      sample.drawPost(cmdBuf);

      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
      vkCmdEndRenderPass(cmdBuf);
    }

    profiler.endFrame();

    // Submit for display
    vkEndCommandBuffer(cmdBuf);
    sample.submitFrame();

    CameraManip.updateAnim();
  }

  // Cleanup
  vkDeviceWaitIdle(sample.getDevice());
  sample.destroyResources();
  sample.destroy();
  profiler.deinit();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

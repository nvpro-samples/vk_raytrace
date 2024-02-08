/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#define IMGUI_DEFINE_MATH_OPERATORS
#include <implot.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/inputparser.h"
#include "nvpsystem.hpp"
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
  nvvk::ContextCreateInfo contextInfo(true);
  contextInfo.setVersion(1, 3);                       // Using Vulkan 1.3
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
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, true, &rayQueryFeatures);  // Optional extension
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

  // Extra queues for parallel load/build
  contextInfo.addRequestedQueue(contextInfo.defaultQueueGCT, 1, 1.0f);  // Loading scene - mipmap generation

// #define ENABLE_GPU_PRINTF //   Enabling printf in shaders
// #extension GL_EXT_debug_printf
// debugPrintfEXT("");
#ifdef ENABLE_GPU_PRINTF
  contextInfo.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
  std::vector<VkValidationFeatureEnableEXT>  enables{VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  std::vector<VkValidationFeatureDisableEXT> disables{};
  VkValidationFeaturesEXT                    features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  features.enabledValidationFeatureCount  = static_cast<uint32_t>(enables.size());
  features.pEnabledValidationFeatures     = enables.data();
  features.disabledValidationFeatureCount = static_cast<uint32_t>(disables.size());
  features.pDisabledValidationFeatures    = disables.data();
  contextInfo.instanceCreateInfoExt       = &features;
#endif  // ENABLE_GPU_PRINTF


  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);  // Find all compatible devices
  assert(!compatibleDevices.empty());
  vkctx.initDevice(compatibleDevices[0], contextInfo);  // Use first compatible device


  //
  SampleExample sample;
  sample.supportRayQuery(vkctx.hasDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME));

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = sample.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);
  sample.setupGlfwCallbacks(window);

  // Collecting all the Queues the sample will need.
  // - 3 default queues are created, but need extra for load/generate mip-maps
  // - GCT0 for graphic (main for rendering)
  // - GTC1 for loading in parallel and generating mip-maps
  // - Compute for creating acceleration structures
  // - Transfer for loading HDR images, creating offscreen pipeline
  auto                     qGCT1 = vkctx.createQueue(contextInfo.defaultQueueGCT, "GCT1", 1.0f);
  std::vector<nvvk::Queue> queues;
  queues.push_back({vkctx.m_queueGCT.queue, vkctx.m_queueGCT.familyIndex, vkctx.m_queueGCT.queueIndex});
  queues.push_back({qGCT1.queue, qGCT1.familyIndex, qGCT1.queueIndex});
  queues.push_back({vkctx.m_queueC.queue, vkctx.m_queueC.familyIndex, vkctx.m_queueC.queueIndex});
  queues.push_back({vkctx.m_queueT.queue, vkctx.m_queueT.familyIndex, vkctx.m_queueT.queueIndex});

  // Create example
  sample.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, queues);
  sample.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  sample.createDepthBuffer();
  sample.createRenderPass();
  sample.createFrameBuffers();

  // Setup Imgui
  sample.initGUI();
  ImPlot::CreateContext();
  sample.createOffscreenRender();
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGui::GetIO().MouseDoubleClickTime    = 0.2f;  // Default: 0.3
  ImGui::GetIO().MouseDoubleClickMaxDist = 2.0f;  // Default: 6.0

  // Creation of the example - loading scene in separate thread
  sample.loadEnvironmentHdr(nvh::findFile(hdrFilename, defaultSearchPaths, true));
  sample.m_busy = true;
  std::thread([&] {
    sample.m_busyReasonText = "Loading Scene";
    sample.loadScene(nvh::findFile(sceneFile, defaultSearchPaths, true));
    sample.createUniformBuffer();
    sample.createDescriptorSetLayout();
    sample.createRender(SampleExample::eRtxPipeline);
    sample.resetFrame();
    sample.m_busy = false;
  }).detach();


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
    profiler.beginFrame();  // GPU performance timer
    sample.prepareFrame();  // Waits for a framebuffer to be available
    sample.updateFrame();   // Increment/update rendering frame count

    // Start command buffer of this frame
    auto                   curFrame = sample.getCurFrame();
    const VkCommandBuffer& cmdBuf   = sample.getCommandBuffers()[curFrame];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    sample.renderGui(profiler);          // UI
    sample.updateUniformBuffer(cmdBuf);  // Updating UBOs

    // Rendering Scene (ray tracing)
    sample.renderScene(cmdBuf, profiler);

    // Rendering pass in swapchain framebuffer + tone mapper, UI
    {
      auto sec = profiler.timeRecurring("Tonemap", cmdBuf);

      std::array<VkClearValue, 2> clearValues;
      clearValues[0].color        = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[1].depthStencil = {1.0f, 0};

      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = 2;
      postRenderPassBeginInfo.pClearValues    = clearValues.data();
      postRenderPassBeginInfo.renderPass      = sample.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = sample.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{}, sample.getSize()};

      vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      // Draw the rendering result + tonemapper
      sample.drawPost(cmdBuf);

      // Render the UI
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
  glfwDestroyWindow(window);
  sample.destroyResources();
  sample.destroy();
  ImPlot::DestroyContext();
  profiler.deinit();
  vkctx.deinit();

  glfwTerminate();

  return 0;
}

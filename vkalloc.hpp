/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <vulkan/vulkan.hpp>

//#define NVVK_ALLOC_DEDICATED
#define NVVK_ALLOC_DMA
//#define NVVK_ALLOC_VMA

#include <nvvk/allocator_vk.hpp>

using vkDT = vk::DescriptorType;
using vkDS = vk::DescriptorSetLayoutBinding;
using vkSS = vk::ShaderStageFlagBits;
using vkCB = vk::CommandBufferUsageFlagBits;
using vkBU = vk::BufferUsageFlagBits;
using vkIU = vk::ImageUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;


// Utility to time the execution of something resetting the timer
// on each elapse call
// Usage:
// {
//   MilliTimer timer;
//   ... stuff ...
//   double time_elapse = timer.elapse();
// }
#include <chrono>

struct MilliTimer
{
  MilliTimer() { reset(); }
  void   reset() { startTime = std::chrono::high_resolution_clock::now(); }
  double elapse()
  {
    auto now  = std::chrono::high_resolution_clock::now();
    auto t    = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count() / 1000.0;
    startTime = now;
    return t;
  }
  std::chrono::high_resolution_clock::time_point startTime;
};

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

#pragma once


/**
  # class RayPickerKHR

  This is a utility to get hit information under a screen coordinate. 

  The information returned is: 
    - origin and direction in world space
    - hitT, the distance of the hit along the ray direction
    - primitiveID, instanceID and instanceCustomIndex
    - the barycentric coordinates in the triangle

  Setting up:
    - call setup() once with the Vulkan device, and allocator
    - call initialize with the TLAS previously build
  
  Getting results, for example, on mouse down:
  - fill the PickInfo structure
  - call run()
  - call getResult() to get all the information above


  Example to set the camera interest point 
    
    RayPickerKHR::PickResult pr = m_picker.getResult();
    if(pr.instanceID != ~0) // Hit something
    {
      nvmath::vec3 worldPos = pr.worldRayOrigin + pr.worldRayDirection * pr.hitT;
      nvmath::vec3f eye, center, up;
      CameraManip.getLookat(eye, center, up);
      CameraManip.setLookat(eye, worldPos, up, false); // Nice with CameraManip.updateAnim();
    }

  **/

#include "nvmath/nvmath.h"
#include "nvvk/allocator_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "vulkan/vulkan.hpp"


struct RayPickerKHR
{
public:
  struct PickInfo
  {
    mat4  modelViewInv;    // inverse model view matrix
    mat4  perspectiveInv;  // inverse perspective matrix
    float pickX{0};        // normalized X position
    float pickY{0};        // normalized Y position
  } m_pickInfo;

  struct PickResult
  {
    nvmath::vec4f worldRayOrigin{0, 0, 0, 0};
    nvmath::vec4f worldRayDirection{0, 0, 0, 0};
    float         hitT{0};
    int           primitiveID{0};
    int           instanceID{~0};
    int           instanceCustomIndex{0};
    nvmath::vec3f baryCoord{0, 0, 0};
  };

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
  {
    m_physicalDevice = physicalDevice;
    m_device         = device;
    m_queueIndex     = queueIndex;
    m_debug.setup(device);
    m_alloc = allocator;
  }

  void run(const vk::CommandBuffer& cmdBuf, const PickInfo& pickInfo)
  {
    m_pickInfo = pickInfo;

    cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, {m_descSet}, {});
    cmdBuf.pushConstants<PickInfo>(m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, m_pickInfo);
    cmdBuf.dispatch(1, 1, 1);  // one pixel
    // Wait for result
    vk::BufferMemoryBarrier bmb{vk::AccessFlagBits::eMemoryWrite,
                                vk::AccessFlagBits::eMemoryRead,
                                VK_QUEUE_FAMILY_IGNORED,
                                VK_QUEUE_FAMILY_IGNORED,
                                m_pickResult.buffer,
                                0,
                                VK_WHOLE_SIZE};
    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe,
                           vk::DependencyFlagBits::eDeviceGroup, 0, nullptr, 1, &bmb, 0, nullptr);
  }

  PickResult getResult()
  {
    PickResult pr;
    void*      mapped = m_alloc->map(m_pickResult);
    memcpy(&pr, mapped, sizeof(PickResult));
    m_alloc->unmap(m_pickResult);
    return pr;
  }

  void destroy()
  {
    m_alloc->destroy(m_pickResult);
    m_alloc->destroy(m_sbtBuffer);
    m_device.destroy(m_descSetLayout);
    m_device.destroy(m_descPool);
    m_device.destroy(m_pipelineLayout);
    m_device.destroy(m_pipeline);
    m_pickResult     = nvvk::Buffer();
    m_descSetLayout  = vk::DescriptorSetLayout();
    m_descSet        = vk::DescriptorSet();
    m_pipelineLayout = vk::PipelineLayout();
    m_pipeline       = vk::Pipeline();
  }

  // tlas : top acceleration structure
  void initialize(const vk::AccelerationStructureKHR& tlas)
  {
    m_tlas = tlas;
    createOutputResult();
    createDescriptorSet();
    createPipeline();
  }

private:
  nvvk::Buffer m_pickResult;
  nvvk::Buffer m_sbtBuffer;

  nvvk::DescriptorSetBindings  m_binding;
  vk::DescriptorPool           m_descPool;
  vk::DescriptorSetLayout      m_descSetLayout;
  vk::DescriptorSet            m_descSet;
  vk::PipelineLayout           m_pipelineLayout;
  vk::Pipeline                 m_pipeline;
  vk::AccelerationStructureKHR m_tlas;
  vk::PhysicalDevice           m_physicalDevice;
  vk::Device                   m_device;
  uint32_t                     m_queueIndex;
  nvvk::Allocator*             m_alloc{nullptr};
  nvvk::DebugUtil              m_debug;


  void createOutputResult()
  {
    m_alloc->destroy(m_pickResult);
    nvvk::CommandPool sCmd(m_device, m_queueIndex);
    vk::CommandBuffer cmdBuf = sCmd.createCommandBuffer();
    PickResult        presult{};
    m_pickResult = m_alloc->createBuffer(cmdBuf, sizeof(PickResult), &presult,
                                         vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                         vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);
    sCmd.submitAndWait(cmdBuf);
    m_alloc->finalizeAndReleaseStaging();
    NAME_VK(m_pickResult.buffer);
  }

  void createDescriptorSet()
  {
    m_device.destroy(m_descSetLayout);
    m_device.destroy(m_descPool);

    m_binding.clear();
    m_binding.addBinding(0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eCompute);
    m_binding.addBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);

    m_descPool      = m_binding.createPool(m_device);
    m_descSetLayout = m_binding.createLayout(m_device);
    m_descSet       = m_device.allocateDescriptorSets({m_descPool, 1, &m_descSetLayout})[0];

    vk::WriteDescriptorSetAccelerationStructureKHR descAsInfo{1, &m_tlas};

    vk::DescriptorBufferInfo            pickDesc{m_pickResult.buffer, 0, VK_WHOLE_SIZE};
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_binding.makeWrite(m_descSet, 0, &descAsInfo));
    writes.emplace_back(m_binding.makeWrite(m_descSet, 1, &pickDesc));
    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void createPipeline()
  {
    m_device.destroy(m_pipelineLayout);
    m_device.destroy(m_pipeline);

    vk::PushConstantRange        pushConstant{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PickInfo)};
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    pipelineLayoutCreateInfo.setSetLayoutCount(1);
    pipelineLayoutCreateInfo.setPSetLayouts(&m_descSetLayout);
    pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
    pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
    CREATE_NAMED_VK(m_pipelineLayout, m_device.createPipelineLayout(pipelineLayoutCreateInfo));

    vk::ComputePipelineCreateInfo computePipelineCreateInfo{{}, {}, m_pipelineLayout};
    computePipelineCreateInfo.stage = nvvk::createShaderStageInfo(m_device, getSpirV(), VK_SHADER_STAGE_COMPUTE_BIT);
    CREATE_NAMED_VK(m_pipeline, m_device.createComputePipeline({}, computePipelineCreateInfo).value);
    m_device.destroy(computePipelineCreateInfo.stage.module);
  }


  const std::vector<uint32_t> getSpirV()
  {  // glslangValidator.exe --target-env vulkan1.2 --variable-name pick
    //const uint32_t pick[] =
    return {0x07230203, 0x00010500, 0x0008000a, 0x00000089, 0x00000000, 0x00020011, 0x00000001, 0x00020011, 0x00001178,
            0x0006000a, 0x5f565053, 0x5f52484b, 0x5f796172, 0x72657571, 0x00000079, 0x0006000b, 0x00000001, 0x4c534c47,
            0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000005, 0x00000004,
            0x6e69616d, 0x00000000, 0x0000000e, 0x00000047, 0x0000005f, 0x00060010, 0x00000004, 0x00000011, 0x00000001,
            0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001cc, 0x00060004, 0x455f4c47, 0x725f5458, 0x715f7961,
            0x79726575, 0x00000000, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00050005, 0x00000009, 0x65786970,
            0x6e65436c, 0x00726574, 0x00050005, 0x0000000c, 0x736e6f43, 0x746e6174, 0x00000073, 0x00070006, 0x0000000c,
            0x00000000, 0x65646f6d, 0x6569566c, 0x766e4977, 0x00000000, 0x00070006, 0x0000000c, 0x00000001, 0x73726570,
            0x74636570, 0x49657669, 0x0000766e, 0x00050006, 0x0000000c, 0x00000002, 0x6b636970, 0x00000058, 0x00050006,
            0x0000000c, 0x00000003, 0x6b636970, 0x00000059, 0x00030005, 0x0000000e, 0x00000000, 0x00030005, 0x00000018,
            0x00000064, 0x00040005, 0x00000020, 0x6769726f, 0x00006e69, 0x00040005, 0x00000028, 0x67726174, 0x00007465,
            0x00050005, 0x00000036, 0x65726964, 0x6f697463, 0x0000006e, 0x00050005, 0x00000044, 0x51796172, 0x79726575,
            0x00000000, 0x00050005, 0x00000047, 0x4c706f74, 0x6c657665, 0x00005341, 0x00030005, 0x00000058, 0x00746968,
            0x00050005, 0x0000005c, 0x6b636950, 0x75736552, 0x0000746c, 0x00070006, 0x0000005c, 0x00000000, 0x6c726f77,
            0x79615264, 0x6769724f, 0x00006e69, 0x00080006, 0x0000005c, 0x00000001, 0x6c726f77, 0x79615264, 0x65726944,
            0x6f697463, 0x0000006e, 0x00050006, 0x0000005c, 0x00000002, 0x54746968, 0x00000000, 0x00060006, 0x0000005c,
            0x00000003, 0x6d697270, 0x76697469, 0x00444965, 0x00060006, 0x0000005c, 0x00000004, 0x74736e69, 0x65636e61,
            0x00004449, 0x00080006, 0x0000005c, 0x00000005, 0x74736e69, 0x65636e61, 0x74737543, 0x6e496d6f, 0x00786564,
            0x00060006, 0x0000005c, 0x00000006, 0x79726162, 0x726f6f43, 0x00000064, 0x00050005, 0x0000005d, 0x7365725f,
            0x50746c75, 0x006b6369, 0x00060006, 0x0000005d, 0x00000000, 0x75736572, 0x6950746c, 0x00006b63, 0x00030005,
            0x0000005f, 0x00000000, 0x00040005, 0x00000079, 0x79726162, 0x00000000, 0x00040048, 0x0000000c, 0x00000000,
            0x00000005, 0x00050048, 0x0000000c, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000c, 0x00000000,
            0x00000007, 0x00000010, 0x00040048, 0x0000000c, 0x00000001, 0x00000005, 0x00050048, 0x0000000c, 0x00000001,
            0x00000023, 0x00000040, 0x00050048, 0x0000000c, 0x00000001, 0x00000007, 0x00000010, 0x00050048, 0x0000000c,
            0x00000002, 0x00000023, 0x00000080, 0x00050048, 0x0000000c, 0x00000003, 0x00000023, 0x00000084, 0x00030047,
            0x0000000c, 0x00000002, 0x00040047, 0x00000047, 0x00000022, 0x00000000, 0x00040047, 0x00000047, 0x00000021,
            0x00000000, 0x00050048, 0x0000005c, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000005c, 0x00000001,
            0x00000023, 0x00000010, 0x00050048, 0x0000005c, 0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x0000005c,
            0x00000003, 0x00000023, 0x00000024, 0x00050048, 0x0000005c, 0x00000004, 0x00000023, 0x00000028, 0x00050048,
            0x0000005c, 0x00000005, 0x00000023, 0x0000002c, 0x00050048, 0x0000005c, 0x00000006, 0x00000023, 0x00000030,
            0x00050048, 0x0000005d, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000005d, 0x00000002, 0x00040047,
            0x0000005f, 0x00000022, 0x00000000, 0x00040047, 0x0000005f, 0x00000021, 0x00000001, 0x00020013, 0x00000002,
            0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006,
            0x00000002, 0x00040020, 0x00000008, 0x00000007, 0x00000007, 0x00040017, 0x0000000a, 0x00000006, 0x00000004,
            0x00040018, 0x0000000b, 0x0000000a, 0x00000004, 0x0006001e, 0x0000000c, 0x0000000b, 0x0000000b, 0x00000006,
            0x00000006, 0x00040020, 0x0000000d, 0x00000009, 0x0000000c, 0x0004003b, 0x0000000d, 0x0000000e, 0x00000009,
            0x00040015, 0x0000000f, 0x00000020, 0x00000001, 0x0004002b, 0x0000000f, 0x00000010, 0x00000002, 0x00040020,
            0x00000011, 0x00000009, 0x00000006, 0x0004002b, 0x0000000f, 0x00000014, 0x00000003, 0x0004002b, 0x00000006,
            0x0000001a, 0x40000000, 0x0004002b, 0x00000006, 0x0000001c, 0x3f800000, 0x00040020, 0x0000001f, 0x00000007,
            0x0000000a, 0x0004002b, 0x0000000f, 0x00000021, 0x00000000, 0x00040020, 0x00000022, 0x00000009, 0x0000000b,
            0x0004002b, 0x00000006, 0x00000025, 0x00000000, 0x0007002c, 0x0000000a, 0x00000026, 0x00000025, 0x00000025,
            0x00000025, 0x0000001c, 0x0004002b, 0x0000000f, 0x00000029, 0x00000001, 0x00040015, 0x0000002c, 0x00000020,
            0x00000000, 0x0004002b, 0x0000002c, 0x0000002d, 0x00000000, 0x00040020, 0x0000002e, 0x00000007, 0x00000006,
            0x0004002b, 0x0000002c, 0x00000031, 0x00000001, 0x00040017, 0x00000039, 0x00000006, 0x00000003, 0x00021178,
            0x00000042, 0x00040020, 0x00000043, 0x00000007, 0x00000042, 0x000214dd, 0x00000045, 0x00040020, 0x00000046,
            0x00000000, 0x00000045, 0x0004003b, 0x00000046, 0x00000047, 0x00000000, 0x0004002b, 0x0000002c, 0x00000049,
            0x000000ff, 0x0004002b, 0x00000006, 0x0000004c, 0x3727c5ac, 0x0004002b, 0x00000006, 0x0000004f, 0x749dc5ae,
            0x00020014, 0x00000055, 0x00040020, 0x00000057, 0x00000007, 0x00000055, 0x00030029, 0x00000055, 0x00000059,
            0x0009001e, 0x0000005c, 0x0000000a, 0x0000000a, 0x00000006, 0x0000000f, 0x0000000f, 0x0000000f, 0x00000039,
            0x0003001e, 0x0000005d, 0x0000005c, 0x00040020, 0x0000005e, 0x0000000c, 0x0000005d, 0x0004003b, 0x0000005e,
            0x0000005f, 0x0000000c, 0x00040020, 0x00000061, 0x0000000c, 0x0000000a, 0x00040020, 0x00000066, 0x0000000c,
            0x00000006, 0x00040020, 0x00000069, 0x0000000c, 0x0000000f, 0x0004002b, 0x0000000f, 0x0000006b, 0x00000004,
            0x00040020, 0x0000006d, 0x00000007, 0x0000000f, 0x0004002b, 0x0000000f, 0x00000073, 0xffffffff, 0x0004002b,
            0x0000000f, 0x00000076, 0x00000005, 0x0004002b, 0x0000000f, 0x0000007b, 0x00000006, 0x00040020, 0x00000087,
            0x0000000c, 0x00000039, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
            0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003b, 0x00000008, 0x00000018, 0x00000007, 0x0004003b,
            0x0000001f, 0x00000020, 0x00000007, 0x0004003b, 0x0000001f, 0x00000028, 0x00000007, 0x0004003b, 0x0000001f,
            0x00000036, 0x00000007, 0x0004003b, 0x00000043, 0x00000044, 0x00000007, 0x0004003b, 0x00000057, 0x00000058,
            0x00000007, 0x0004003b, 0x0000006d, 0x0000006e, 0x00000007, 0x0004003b, 0x00000008, 0x00000079, 0x00000007,
            0x00050041, 0x00000011, 0x00000012, 0x0000000e, 0x00000010, 0x0004003d, 0x00000006, 0x00000013, 0x00000012,
            0x00050041, 0x00000011, 0x00000015, 0x0000000e, 0x00000014, 0x0004003d, 0x00000006, 0x00000016, 0x00000015,
            0x00050050, 0x00000007, 0x00000017, 0x00000013, 0x00000016, 0x0003003e, 0x00000009, 0x00000017, 0x0004003d,
            0x00000007, 0x00000019, 0x00000009, 0x0005008e, 0x00000007, 0x0000001b, 0x00000019, 0x0000001a, 0x00050050,
            0x00000007, 0x0000001d, 0x0000001c, 0x0000001c, 0x00050083, 0x00000007, 0x0000001e, 0x0000001b, 0x0000001d,
            0x0003003e, 0x00000018, 0x0000001e, 0x00050041, 0x00000022, 0x00000023, 0x0000000e, 0x00000021, 0x0004003d,
            0x0000000b, 0x00000024, 0x00000023, 0x00050091, 0x0000000a, 0x00000027, 0x00000024, 0x00000026, 0x0003003e,
            0x00000020, 0x00000027, 0x00050041, 0x00000022, 0x0000002a, 0x0000000e, 0x00000029, 0x0004003d, 0x0000000b,
            0x0000002b, 0x0000002a, 0x00050041, 0x0000002e, 0x0000002f, 0x00000018, 0x0000002d, 0x0004003d, 0x00000006,
            0x00000030, 0x0000002f, 0x00050041, 0x0000002e, 0x00000032, 0x00000018, 0x00000031, 0x0004003d, 0x00000006,
            0x00000033, 0x00000032, 0x00070050, 0x0000000a, 0x00000034, 0x00000030, 0x00000033, 0x0000001c, 0x0000001c,
            0x00050091, 0x0000000a, 0x00000035, 0x0000002b, 0x00000034, 0x0003003e, 0x00000028, 0x00000035, 0x00050041,
            0x00000022, 0x00000037, 0x0000000e, 0x00000021, 0x0004003d, 0x0000000b, 0x00000038, 0x00000037, 0x0004003d,
            0x0000000a, 0x0000003a, 0x00000028, 0x0008004f, 0x00000039, 0x0000003b, 0x0000003a, 0x0000003a, 0x00000000,
            0x00000001, 0x00000002, 0x0006000c, 0x00000039, 0x0000003c, 0x00000001, 0x00000045, 0x0000003b, 0x00050051,
            0x00000006, 0x0000003d, 0x0000003c, 0x00000000, 0x00050051, 0x00000006, 0x0000003e, 0x0000003c, 0x00000001,
            0x00050051, 0x00000006, 0x0000003f, 0x0000003c, 0x00000002, 0x00070050, 0x0000000a, 0x00000040, 0x0000003d,
            0x0000003e, 0x0000003f, 0x00000025, 0x00050091, 0x0000000a, 0x00000041, 0x00000038, 0x00000040, 0x0003003e,
            0x00000036, 0x00000041, 0x0004003d, 0x00000045, 0x00000048, 0x00000047, 0x0004003d, 0x0000000a, 0x0000004a,
            0x00000020, 0x0008004f, 0x00000039, 0x0000004b, 0x0000004a, 0x0000004a, 0x00000000, 0x00000001, 0x00000002,
            0x0004003d, 0x0000000a, 0x0000004d, 0x00000036, 0x0008004f, 0x00000039, 0x0000004e, 0x0000004d, 0x0000004d,
            0x00000000, 0x00000001, 0x00000002, 0x00091179, 0x00000044, 0x00000048, 0x0000002d, 0x00000049, 0x0000004b,
            0x0000004c, 0x0000004e, 0x0000004f, 0x000200f9, 0x00000050, 0x000200f8, 0x00000050, 0x000400f6, 0x00000052,
            0x00000053, 0x00000000, 0x000200f9, 0x00000054, 0x000200f8, 0x00000054, 0x0004117d, 0x00000055, 0x00000056,
            0x00000044, 0x000400fa, 0x00000056, 0x00000051, 0x00000052, 0x000200f8, 0x00000051, 0x0002117c, 0x00000044,
            0x000200f9, 0x00000053, 0x000200f8, 0x00000053, 0x000200f9, 0x00000050, 0x000200f8, 0x00000052, 0x0005117f,
            0x0000002c, 0x0000005a, 0x00000044, 0x00000029, 0x000500ab, 0x00000055, 0x0000005b, 0x0000005a, 0x0000002d,
            0x0003003e, 0x00000058, 0x0000005b, 0x0004003d, 0x0000000a, 0x00000060, 0x00000020, 0x00060041, 0x00000061,
            0x00000062, 0x0000005f, 0x00000021, 0x00000021, 0x0003003e, 0x00000062, 0x00000060, 0x0004003d, 0x0000000a,
            0x00000063, 0x00000036, 0x00060041, 0x00000061, 0x00000064, 0x0000005f, 0x00000021, 0x00000029, 0x0003003e,
            0x00000064, 0x00000063, 0x00051782, 0x00000006, 0x00000065, 0x00000044, 0x00000029, 0x00060041, 0x00000066,
            0x00000067, 0x0000005f, 0x00000021, 0x00000010, 0x0003003e, 0x00000067, 0x00000065, 0x00051787, 0x0000000f,
            0x00000068, 0x00000044, 0x00000029, 0x00060041, 0x00000069, 0x0000006a, 0x0000005f, 0x00000021, 0x00000014,
            0x0003003e, 0x0000006a, 0x00000068, 0x0004003d, 0x00000055, 0x0000006c, 0x00000058, 0x000300f7, 0x00000070,
            0x00000000, 0x000400fa, 0x0000006c, 0x0000006f, 0x00000072, 0x000200f8, 0x0000006f, 0x00051784, 0x0000000f,
            0x00000071, 0x00000044, 0x00000029, 0x0003003e, 0x0000006e, 0x00000071, 0x000200f9, 0x00000070, 0x000200f8,
            0x00000072, 0x0003003e, 0x0000006e, 0x00000073, 0x000200f9, 0x00000070, 0x000200f8, 0x00000070, 0x0004003d,
            0x0000000f, 0x00000074, 0x0000006e, 0x00060041, 0x00000069, 0x00000075, 0x0000005f, 0x00000021, 0x0000006b,
            0x0003003e, 0x00000075, 0x00000074, 0x00051783, 0x0000000f, 0x00000077, 0x00000044, 0x00000029, 0x00060041,
            0x00000069, 0x00000078, 0x0000005f, 0x00000021, 0x00000076, 0x0003003e, 0x00000078, 0x00000077, 0x00051788,
            0x00000007, 0x0000007a, 0x00000044, 0x00000029, 0x0003003e, 0x00000079, 0x0000007a, 0x00050041, 0x0000002e,
            0x0000007c, 0x00000079, 0x0000002d, 0x0004003d, 0x00000006, 0x0000007d, 0x0000007c, 0x00050083, 0x00000006,
            0x0000007e, 0x0000001c, 0x0000007d, 0x00050041, 0x0000002e, 0x0000007f, 0x00000079, 0x00000031, 0x0004003d,
            0x00000006, 0x00000080, 0x0000007f, 0x00050083, 0x00000006, 0x00000081, 0x0000007e, 0x00000080, 0x00050041,
            0x0000002e, 0x00000082, 0x00000079, 0x0000002d, 0x0004003d, 0x00000006, 0x00000083, 0x00000082, 0x00050041,
            0x0000002e, 0x00000084, 0x00000079, 0x00000031, 0x0004003d, 0x00000006, 0x00000085, 0x00000084, 0x00060050,
            0x00000039, 0x00000086, 0x00000081, 0x00000083, 0x00000085, 0x00060041, 0x00000087, 0x00000088, 0x0000005f,
            0x00000021, 0x0000007b, 0x0003003e, 0x00000088, 0x00000086, 0x000100fd, 0x00010038};
  }

  std::string getGlsl()
  {
    return R"(
#version 460
#extension GL_EXT_ray_query : require

// clang-format off
struct PickResult
{
  vec4  worldRayOrigin;
  vec4  worldRayDirection;
  float hitT;
  int   primitiveID;
  int   instanceID;
  int   instanceCustomIndex;
  vec3  baryCoord;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1) buffer _resultPick { PickResult resultPick; };
layout(push_constant) uniform Constants
{
  mat4  modelViewInv;
  mat4  perspectiveInv;
  float pickX;  // normalized
  float pickY;
};

void main()
{
  const vec2 pixelCenter = vec2(pickX, pickY);
  vec2       d           = pixelCenter * 2.0 - 1.0;
  vec4 origin            = modelViewInv * vec4(0, 0, 0, 1);
  vec4 target            = perspectiveInv * vec4(d.x, d.y, 1, 1);
  vec4 direction         = modelViewInv * vec4(normalize(target.xyz), 0);

  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery, topLevelAS, 0, 0xff, origin.xyz, 0.00001, direction.xyz, 1e32);
  while(rayQueryProceedEXT(rayQuery)) {rayQueryConfirmIntersectionEXT(rayQuery); }

  bool hit = (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
  resultPick.worldRayOrigin      = origin;
  resultPick.worldRayDirection   = direction;
  resultPick.hitT                = rayQueryGetIntersectionTEXT(rayQuery, true);
  resultPick.primitiveID         = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
  resultPick.instanceID          = hit ? rayQueryGetIntersectionInstanceIdEXT(rayQuery, true) : ~0;
  resultPick.instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
  vec2 bary                      = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
  resultPick.baryCoord           = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
}
// clang-format on
)";
  }
};

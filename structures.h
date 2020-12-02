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


/*
  Various structure used by CPP and GLSL 
*/


#ifndef STRUCTURES_H
#define STRUCTURES_H

// Payload structure
struct RayPayload
{
  vec4 origin;     // Ray origin(xyz), pixelIndex
  vec4 direction;  // Ray direction(xyz), seed
  vec4 state;      // baryCENTICS, CustomInstanceID, PrimitiveId, InstanceId
  vec4 weight;     //
};


// Camera of the scene
struct CameraMatrices
{
  mat4 view;
  mat4 proj;
  mat4 viewInverse;
  mat4 projInverse;
};

#ifdef __cplusplus
enum DebugMode
{
#else
const uint
#endif  // __cplusplus
  eNoDebug   = 0,
  eBaseColor = 1,
  eNormal    = 2,
  eMetallic  = 3,
  eAO        = 4,
  eEmissive  = 5,
  eAlpha     = 6,
  eRoughness = 7,
  eTextcoord = 8,
  eTangent   = 9
#ifdef __cplusplus
}
#endif  // __cplusplus
;

// Use with PushConstant
struct RtState
{
  int   frame;                         // Current frame, start at 0
  int   maxDepth;                      // How deep the path is
  int   maxSamples;                    // How many samples to do per render
  float fireflyClampThreshold;         // to cut fireflies
  float environment_intensity_factor;  // To brightening the scene
  int   debugging_mode;
};

// Structure used for retrieving the primitive information in the closest hit
// The gl_InstanceCustomIndexNV
struct RtPrimitiveLookup
{
  uint indexOffset;
  uint vertexOffset;
  int  materialIndex;
};

struct InstanceMatrices
{
  mat4 object2World;
  mat4 world2Object;
};

#endif

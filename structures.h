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


// GLTF material
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2
struct GltfShadeMaterial
{

  vec4 pbrBaseColorFactor;

  int   pbrBaseColorTexture;
  float pbrMetallicFactor;
  float pbrRoughnessFactor;
  int   pbrMetallicRoughnessTexture;

  // KHR_materials_pbrSpecularGlossiness
  vec4 khrDiffuseFactor;
  vec3 khrSpecularFactor;
  int  khrDiffuseTexture;

  int   shadingModel;  // 0: metallic-roughness, 1: specular-glossiness
  float khrGlossinessFactor;
  int   khrSpecularGlossinessTexture;
  int   emissiveTexture;

  vec3 emissiveFactor;
  int  alphaMode;

  float alphaCutoff;
  int   doubleSided;
  int   normalTexture;
  float normalTextureScale;

  mat4 uvTransform;
  int  unlit;

  float anisotropy;
  vec3  anisotropyDirection;
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
  eEmissive  = 4,
  eAlpha     = 5,
  eRoughness = 6,
  eTextcoord = 7,
  eTangent   = 8,
  eRadiance  = 9,
  eWeight    = 10,
  eRayDir    = 11


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

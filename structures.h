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


// Camera of the scene
struct SceneCamera
{
  mat4  view;
  mat4  proj;
  mat4  viewInverse;
  mat4  projInverse;
  float focalDist;
  float aperture;
  // Extra
  int nbLights;
};


// GLTF material
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2
struct GltfShadeMaterial
{
  // 0
  vec4 pbrBaseColorFactor;
  // 4
  int   pbrBaseColorTexture;
  float pbrMetallicFactor;
  float pbrRoughnessFactor;
  int   pbrMetallicRoughnessTexture;
  // 8
  vec4 khrDiffuseFactor;  // KHR_materials_pbrSpecularGlossiness
  vec3 khrSpecularFactor;
  int  khrDiffuseTexture;
  // 16
  int   shadingModel;  // 0: metallic-roughness, 1: specular-glossiness
  float khrGlossinessFactor;
  int   khrSpecularGlossinessTexture;
  int   emissiveTexture;
  // 20
  vec3 emissiveFactor;
  int  alphaMode;
  // 24
  float alphaCutoff;
  int   doubleSided;
  int   normalTexture;
  float normalTextureScale;
  // 28
  mat4 uvTransform;
  // 32
  int unlit;

  float transmissionFactor;
  int   transmissionTexture;

  float ior;
  // 36
  vec3  anisotropyDirection;
  float anisotropy;
  // 40
  vec3  attenuationColor;
  float thicknessFactor;  // 44
  int   thicknessTexture;
  float attenuationDistance;
  // --
  float clearcoatFactor;
  float clearcoatRoughness;
  // 48
  int clearcoatTexture;
  int clearcoatRoughnessTexture;
  int _pad0;
  int _pad1;
  // 52
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
struct RtxState
{
  int   frame;                  // Current frame, start at 0
  int   maxDepth;               // How deep the path is
  int   maxSamples;             // How many samples to do per render
  float fireflyClampThreshold;  // to cut fireflies

  float hdrMultiplier;   // To brightening the scene
  int   debugging_mode;  //
  int   pbrMode;         // 0-Disney, 1-Gltf
  int   _pad0;

  ivec2 size;  // rendering size
  int   _pad1;
  int   _pad2;
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

// KHR_lights_punctual extension.
// see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual
struct Light
{
  vec3  direction;
  float range;

  vec3  color;
  float intensity;

  vec3  position;
  float innerConeCos;

  float outerConeCos;
  int   type;

  vec2 padding;
};

const int LightType_Directional = 0;
const int LightType_Point       = 1;
const int LightType_Spot        = 2;
#endif

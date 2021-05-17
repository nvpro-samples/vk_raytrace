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

struct VertexAttributes
{
  vec3 position;
  uint normal;    // compressed using oct
  vec2 texcoord;  // Tangent handiness, stored in LSB of .y
  uint tangent;   // compressed using oct
  uint color;     // RGBA
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
  eTexcoord  = 7,
  eTangent   = 8,
  eRadiance  = 9,
  eWeight    = 10,
  eRayDir    = 11,
  eHeatmap   = 12


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
  int   minHeatmap;
  int   maxHeatmap;
};

// Structure used for retrieving the primitive information in the closest hit
// The gl_InstanceCustomIndexNV
struct InstanceData
{
  int materialIndex;
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

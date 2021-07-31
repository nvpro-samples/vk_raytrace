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


//-------------------------------------------------------------------------------------------------
// This file as all constant, global values and structures not shared with CPP

#ifndef GLOBALS_GLSL
#define GLOBALS_GLSL 1

#define PI 3.14159265358979323
#define TWO_PI 6.28318530717958648
#define INFINITY 1e32
#define EPS 0.0001

//precision highp int;
precision highp float;

const float M_PI        = 3.14159265358979323846;   // pi
const float M_TWO_PI    = 6.28318530717958648;      // 2*pi
const float M_PI_2      = 1.57079632679489661923;   // pi/2
const float M_PI_4      = 0.785398163397448309616;  // pi/4
const float M_1_OVER_PI = 0.318309886183790671538;  // 1/pi
const float M_2_OVER_PI = 0.636619772367581343076;  // 2/pi


#define RngStateType uint // Random type

//-----------------------------------------------------------------------
struct Ray
{
  vec3 origin;
  vec3 direction;
};


struct PtPayload
{
  uint   seed;
  float  hitT;
  int    primitiveID;
  int    instanceID;
  int    instanceCustomIndex;
  vec2   baryCoord;
  mat4x3 objectToWorld;
  mat4x3 worldToObject;
};

struct ShadowHitPayload
{
  RngStateType seed;
  bool         isHit;
};

// This material is the shading material after applying textures and any
// other operation. This structure is filled in gltfmaterial.glsl
struct Material
{
  vec3  albedo;
  float specular;
  vec3  emission;
  float anisotropy;
  float metallic;
  float roughness;
  float subsurface;
  float specularTint;
  float sheen;
  float sheenTint;
  float clearcoat;
  float clearcoatRoughness;
  float transmission;
  float ior;
  vec3  attenuationColor;
  float attenuationDistance;

  //vec3  texIDs;
  // Roughness calculated from anisotropic
  float ax;
  float ay;
  // ----
  vec3  f0;
  float alpha;
  bool  unlit;
  bool  thinwalled;
};

// From shading state, this is the structure pass to the eval functions
struct State
{
  int   depth;
  float eta;

  vec3 position;
  vec3 normal;
  vec3 ffnormal;
  vec3 tangent;
  vec3 bitangent;
  vec2 texCoord;

  bool isEmitter;
  bool specularBounce;
  bool isSubsurface;

  uint     matID;
  Material mat;
};


//-----------------------------------------------------------------------
struct BsdfSampleRec
{
  vec3  L;
  vec3  f;
  float pdf;
};

//-----------------------------------------------------------------------
struct LightSampleRec
{
  vec3  surfacePos;
  vec3  normal;
  vec3  emission;
  float pdf;
};


#endif  // GLOBALS_GLSL

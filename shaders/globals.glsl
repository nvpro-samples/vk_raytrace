/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef GLOBALS_GLSL
#define GLOBALS_GLSL 1

#define PI 3.14159265358979323
#define TWO_PI 6.28318530717958648
#define INFINITY 1e32
#define EPS 0.0001

const highp float M_PI        = 3.14159265358979323846;   // pi
const highp float M_TWO_PI    = 6.28318530717958648;      // 2*pi
const highp float M_PI_2      = 1.57079632679489661923;   // pi/2
const highp float M_PI_4      = 0.785398163397448309616;  // pi/4
const highp float M_1_OVER_PI = 0.318309886183790671538;  // 1/pi
const highp float M_2_OVER_PI = 0.636619772367581343076;  // 2/pi

#define REFL 0
#define REFR 1
#define SUBS 2

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
  uint seed;
  bool isHit;
};

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

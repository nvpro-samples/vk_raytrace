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



#ifndef PBR_SAMPLING_GLSL
#define PBR_SAMPLING_GLSL 1

#include "random.glsl"
#include "sampling.glsl"


float clampedDot(vec3 x, vec3 y)
{
  return clamp(dot(x, y), 0.0, 1.0);
}


// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH)
{
  return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

float F_Schlick(float f0, float f90, float VdotH)
{
  return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
float V_GGX(float NdotL, float NdotV, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;

  float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
  float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

  float GGX = GGXV + GGXL;
  if(GGX > 0.0)
  {
    return 0.5 / GGX;
  }
  return 0.0;
}

// Anisotropic GGX visibility function, with height correlation.
// T: Tanget, B: Bi-tanget
float V_GGX_anisotropic(float NdotL, float NdotV, float BdotV, float TdotV, float TdotL, float BdotL, float anisotropy, float at, float ab)
{
  float GGXV = NdotL * length(vec3(at * TdotV, ab * BdotV, NdotV));
  float GGXL = NdotV * length(vec3(at * TdotL, ab * BdotL, NdotL));
  float v    = 0.5 / (GGXV + GGXL);
  return clamp(v, 0.0, 1.0);
}

// https://github.com/google/filament/blob/master/shaders/src/brdf.fs#L136
// https://github.com/google/filament/blob/master/libs/ibl/src/CubemapIBL.cpp#L179
// Note: Google call it V_Ashikhmin and V_Neubelt
float V_Ashikhmin(float NdotL, float NdotV)
{
  return clamp(1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV)), 0.0, 1.0);
}

// https://github.com/google/filament/blob/master/shaders/src/brdf.fs#L131
float V_Kelemen(float LdotH)
{
  // Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
  return 0.25 / (LdotH * LdotH);
}


// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float D_GGX(float NdotH, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;
  float f                = (NdotH * NdotH) * (alphaRoughnessSq - 1.0) + 1.0;
  return alphaRoughnessSq / (M_PI * f * f);
}

// Anisotropic GGX NDF with a single anisotropy parameter controlling the normal orientation.
// See https://google.github.io/filament/Filament.html#materialsystem/anisotropicmodel
// T: Tanget, B: Bi-tanget
float D_GGX_anisotropic(float NdotH, float TdotH, float BdotH, float anisotropy, float at, float ab)
{
  float a2 = at * ab;
  vec3  f  = vec3(ab * TdotH, at * BdotH, a2 * NdotH);
  float w2 = a2 / dot(f, f);
  return a2 * w2 * w2 / M_PI;
}


//Sheen implementation-------------------------------------------------------------------------------------
// See  https://github.com/sebavan/glTF/tree/KHR_materials_sheen/extensions/2.0/Khronos/KHR_materials_sheen

// Estevez and Kulla http://www.aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf
float D_Charlie(float sheenRoughness, float NdotH)
{
  sheenRoughness = max(sheenRoughness, 0.000001);  //clamp (0,1]
  float alphaG   = sheenRoughness * sheenRoughness;
  float invR     = 1.0 / alphaG;
  float cos2h    = NdotH * NdotH;
  float sin2h    = 1.0 - cos2h;
  return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * M_PI);
}

//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
vec3 BRDF_lambertian(vec3 f0, vec3 f90, vec3 diffuseColor, float VdotH, float metallic)
{
  // see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
  // return (1.0 - F_Schlick(f0, f90, VdotH)) * (diffuseColor / M_PI);

  return (1.0 - metallic) * (diffuseColor / M_PI);
}

//  https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
vec3 BRDF_specularGGX(vec3 f0, vec3 f90, float alphaRoughness, float VdotH, float NdotL, float NdotV, float NdotH)
{
  vec3  F = F_Schlick(f0, f90, VdotH);
  float V = V_GGX(NdotL, NdotV, alphaRoughness);
  float D = D_GGX(NdotH, max(0.001, alphaRoughness));

  return F * V * D;
}


vec3 BRDF_specularAnisotropicGGX(vec3  f0,
                                 vec3  f90,
                                 float alphaRoughness,
                                 float VdotH,
                                 float NdotL,
                                 float NdotV,
                                 float NdotH,
                                 float BdotV,
                                 float TdotV,
                                 float TdotL,
                                 float BdotL,
                                 float TdotH,
                                 float BdotH,
                                 float anisotropy)
{
  // Roughness along tangent and bitangent.
  // Christopher Kulla and Alejandro Conty. 2017. Revisiting Physically Based Shading at Imageworks
  float at = max(alphaRoughness * (1.0 + anisotropy), 0.00001);
  float ab = max(alphaRoughness * (1.0 - anisotropy), 0.00001);

  vec3  F = F_Schlick(f0, f90, VdotH);
  float V = V_GGX_anisotropic(NdotL, NdotV, BdotV, TdotV, TdotL, BdotL, anisotropy, at, ab);
  float D = D_GGX_anisotropic(NdotH, TdotH, BdotH, anisotropy, at, ab);

  return F * V * D;
}

// f_sheen
vec3 BRDF_specularSheen(vec3 sheenColor, float sheenIntensity, float sheenRoughness, float NdotL, float NdotV, float NdotH)
{
  float sheenDistribution = D_Charlie(sheenRoughness, NdotH);
  float sheenVisibility   = V_Ashikhmin(NdotL, NdotV);
  return sheenColor * sheenIntensity * sheenDistribution * sheenVisibility;
}


#endif  // PBR_SAMPLING_GLSL

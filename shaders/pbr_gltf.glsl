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
// This fils has all the Gltf sampling and evaluation methods


#ifndef PBR_GLTF_GLSL
#define PBR_GLTF_GLSL 1

#include "random.glsl"
#include "env_sampling.glsl"


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


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 GgxSampling(float specularAlpha, float r1, float r2)
{
  float phi = r1 * 2.0 * M_PI;

  float cosTheta = sqrt((1.0 - r2) / (1.0 + (specularAlpha * specularAlpha - 1.0) * r2));
  float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
  float sinPhi   = sin(phi);
  float cosPhi   = cos(phi);

  return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalDiffuseGltf(State state, vec3 f0, vec3 f90, vec3 V, vec3 N, vec3 L, vec3 H, out float pdf)
{
  pdf         = 0;
  float NdotV = dot(N, V);
  float NdotL = dot(N, L);

  if(NdotL < 0.0 || NdotV < 0.0)
    return vec3(0.0);

  NdotL = clamp(NdotL, 0.001, 1.0);
  NdotV = clamp(abs(NdotV), 0.001, 1.0);

  float VdotH = dot(V, H);

  pdf = NdotL * M_1_OVER_PI;
  return BRDF_lambertian(f0, f90, state.mat.albedo.xyz, VdotH, state.mat.metallic);
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalAnisotropicSpecularGltf(State state, vec3 f0, vec3 f90, vec3 V, vec3 N, vec3 L, vec3 H, out float pdf)
{
  pdf         = 0;
  float NdotL = dot(N, L);

  if(NdotL < 0.0)
    return vec3(0.0);

  vec3 T = state.tangent;
  vec3 B = state.bitangent;

  float TdotV = clamp(dot(T, V), 0, 1);
  float BdotV = clamp(dot(B, V), 0, 1);
  float TdotL = dot(T, L);
  float BdotL = dot(B, L);
  float TdotH = dot(T, H);
  float BdotH = dot(B, H);
  float NdotH = dot(N, H);
  float NdotV = dot(N, V);
  float VdotH = dot(V, H);
  float LdotH = dot(L, H);

  NdotL = clamp(NdotL, 0.001, 1.0);
  NdotV = clamp(abs(NdotV), 0.001, 1.0);


  float at = max(state.mat.roughness * (1.0 + state.mat.anisotropy), 0.001);
  float ab = max(state.mat.roughness * (1.0 - state.mat.anisotropy), 0.001);
  //pdf      = D_GGX_anisotropic(NdotH, TdotH, BdotH, state.mat.anisotropy, at, ab) * NdotH / (4.0 * VdotH);
  pdf = D_GGX_anisotropic(NdotH, TdotH, BdotH, state.mat.anisotropy, at, ab) / (4.0 * LdotH);

  return BRDF_specularAnisotropicGGX(f0, f90, state.mat.roughness, VdotH, NdotL, NdotV, NdotH, BdotV, TdotV, TdotL,
                                     BdotL, TdotH, BdotH, state.mat.anisotropy);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalSpecularGltf(State state, vec3 f0, vec3 f90, vec3 V, vec3 N, vec3 L, vec3 H, out float pdf)
{
  if(state.mat.anisotropy > 0)
    return EvalAnisotropicSpecularGltf(state, f0, f90, V, N, L, H, pdf);

  pdf         = 0;
  float NdotL = dot(N, L);

  if(NdotL < 0.0)
    return vec3(0.0);

  float NdotV = dot(N, V);
  float NdotH = clamp(dot(N, H), 0, 1);
  float LdotH = clamp(dot(L, H), 0, 1);
  float VdotH = clamp(dot(V, H), 0, 1);

  NdotL = clamp(NdotL, 0.001, 1.0);
  NdotV = clamp(abs(NdotV), 0.001, 1.0);


  pdf = D_GGX(NdotH, state.mat.roughness) * NdotH / (4.0 * LdotH);
  return BRDF_specularGGX(f0, f90, state.mat.roughness, VdotH, NdotL, NdotV, NdotH);
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalClearcoatGltf(State state, vec3 V, vec3 N, vec3 L, vec3 H, out float pdf)
{
  pdf         = 0;
  float NdotL = dot(N, L);

  if(NdotL < 0.0)
    return vec3(0.0);

  float NdotH = dot(N, H);
  float NdotV = dot(N, V);
  float VdotH = dot(V, H);
  float LdotH = dot(L, H);

  NdotL = clamp(NdotL, 0.001, 1.0);
  NdotV = clamp(abs(NdotV), 0.001, 1.0);


  float clearcoat        = state.mat.clearcoat;
  float clearcoatFresnel = F_Schlick(0.04, 1, VdotH);
  float clearcoatAlpha   = state.mat.clearcoatRoughness * state.mat.clearcoatRoughness;
  float G                = V_GGX(NdotL, NdotV, clearcoatAlpha);
  float D                = D_GGX(NdotH, max(0.001, clearcoatAlpha));
  pdf                    = D * NdotH / (4.0 * LdotH);

  return vec3(clearcoatFresnel * D * G * clearcoat);
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalDielectricReflectionGltf(State state, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
  float NdotL = dot(N, L);
  if(NdotL < 0.0)
    return vec3(0.0);

  float NdotH = dot(N, H);
  float NdotV = dot(N, V);
  float VdotH = dot(V, H);

  float F = F_Schlick(state.eta, 1, dot(V, H));
  float D = D_GGX(NdotH, state.mat.roughness);
  float G = V_GGX(NdotL, NdotV, state.mat.roughness);

  pdf = D * NdotH * F / (4.0 * VdotH);

  return state.mat.albedo * F * D * G;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalDielectricRefractionGltf(State state, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
  pdf = abs(dot(N, L));
  return state.mat.albedo;


  //float NdotL = dot(N, L);
  //float NdotH = dot(N, H);
  //float NdotV = dot(N, V);
  //float VdotH = dot(V, H);
  //float LdotH = dot(L, H);

  //float F = F_Schlick(state.eta, 1, dot(V, H));
  //float D = D_GGX(NdotH, state.mat.roughness);
  //float G = V_GGX(NdotL, NdotV, state.mat.roughness);


  //float denomSqrt = LdotH * state.eta + VdotH;
  //pdf             = D * NdotH * (1.0 - F) * abs(LdotH) / (denomSqrt * denomSqrt);

  //return state.mat.albedo * (1.0 - F) * D * G * abs(VdotH) * abs(LdotH) * 4.0 * state.eta * state.eta / (denomSqrt * denomSqrt);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 PbrEval(in State state, vec3 V, vec3 N, vec3 L, inout float pdf)
{
  vec3 H;

  if(dot(N, L) < 0.0)
    H = normalize(L * (1.0 / state.eta) + V);
  else
    H = normalize(L + V);

  if(dot(N, H) < 0.0)
    H = -H;

  float diffuseRatio     = 0.5 * (1.0 - state.mat.metallic);
  float primarySpecRatio = 1.0 / (1.0 + state.mat.clearcoat);
  float transWeight      = (1.0 - state.mat.metallic) * state.mat.transmission;

  vec3  brdf    = vec3(0.0);
  vec3  bsdf    = vec3(0.0);
  float brdfPdf = 0.0;
  float bsdfPdf = 0.0;

  // BSDF
  if(transWeight > 0.0)
  {
    bsdf = EvalDielectricRefractionGltf(state, V, N, L, H, bsdfPdf);

    //// Transmission
    //if(dot(N, L) < 0.0)
    //{
    //  bsdf = EvalDielectricRefractionGltf(state, V, N, L, H, bsdfPdf);
    //}
    //else  // Reflection
    //{
    //  bsdf = EvalDielectricReflectionGltf(state, V, N, L, H, bsdfPdf);
    //}
  }

  if(transWeight < 1.0 && dot(N, L) > 0)
  {
    float pdf;

    float diffuseRatio     = 0.5 * (1.0 - state.mat.metallic);
    float specularRatio    = 1.0 - diffuseRatio;
    float primarySpecRatio = 1.0 / (1.0 + state.mat.clearcoat);

    // Compute reflectance.
    // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
    vec3  specularCol = state.mat.f0;
    float reflectance = max(max(specularCol.r, specularCol.g), specularCol.b);
    vec3  f0          = specularCol.rgb;
    vec3  f90         = vec3(clamp(reflectance * 50.0, 0.0, 1.0));

    // Calculation of analytical lighting contribution
    // Diffuse
    brdf += EvalDiffuseGltf(state, f0, f90, V, N, L, H, pdf);
    brdfPdf += pdf * diffuseRatio;

    // Clearcoat
    brdf += EvalClearcoatGltf(state, V, N, L, H, pdf);
    brdfPdf += pdf * (1.0 - primarySpecRatio) * specularRatio;

    // Specular
    brdf += EvalSpecularGltf(state, f0, f90, V, N, L, H, pdf);
    brdfPdf += pdf * primarySpecRatio * specularRatio;  //*(1 - clearcoat * clearcoatFresnel)
  }

  pdf = mix(brdfPdf, bsdfPdf, transWeight);

  return mix(brdf, bsdf, transWeight);
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 PbrSample(in State state, vec3 V, vec3 N, inout vec3 L, inout float pdf, inout RngStateType seed)
{
  pdf       = 0.0;
  vec3 brdf = vec3(0.0);

  float probability   = rand(seed);
  float diffuseRatio  = 0.5 * (1.0 - state.mat.metallic);
  float specularRatio = 1.0 - diffuseRatio;
  float transWeight   = (1.0 - state.mat.metallic) * state.mat.transmission;

  float r1 = rand(seed);
  float r2 = rand(seed);

  if(rand(seed) < transWeight)
  {
    // See http://viclw17.github.io/2018/08/05/raytracing-dielectric-materials/
    float eta = state.eta;

    float n1          = 1.0;
    float n2          = state.mat.ior;
    float R0          = (n1 - n2) / (n1 + n2);
    vec3  H           = GgxSampling(state.mat.roughness, r1, r2);
    H                 = state.tangent * H.x + state.bitangent * H.y + N * H.z;
    float VdotH       = dot(V, H);
    float F           = F_Schlick(R0 * R0, 1.0, VdotH);           // Reflection
    float discriminat = 1.0 - eta * eta * (1.0 - VdotH * VdotH);  // (Total internal reflection)

    if(state.mat.thinwalled)
    {
      // If inside surface, don't reflect
      if(dot(state.ffnormal, state.normal) < 0.0)
      {
        F           = 0;
        discriminat = 0;
      }
      eta = 1.00;  // go through
    }


    // Reflection/Total internal reflection
    if(discriminat < 0.0 || rand(seed) < F)
    {
      L = normalize(reflect(-V, H));
    }
    else
    {
      // Find the pure refractive ray
      L = normalize(refract(-V, H, eta));

      // Cought rays perpendicular to surface, and simply continue
      if(isnan(L.x) || isnan(L.y) || isnan(L.z))
      {
        L = -V;
      }
    }


    // Transmission
    brdf = EvalDielectricRefractionGltf(state, V, N, L, H, pdf);
  }
  else
  {
    // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
    vec3  specularCol = state.mat.f0;
    float reflectance = max(max(specularCol.r, specularCol.g), specularCol.b);
    vec3  f0          = specularCol.rgb;
    vec3  f90         = vec3(clamp(reflectance * 50.0, 0.0, 1.0));

    vec3 T = state.tangent;
    vec3 B = state.bitangent;

    if(probability < diffuseRatio)  // sample diffuse
    {
      L = CosineSampleHemisphere(r1, r2);
      L = L.x * T + L.y * B + L.z * N;

      vec3 H = normalize(L + V);

      brdf = EvalDiffuseGltf(state, f0, f90, V, N, L, H, pdf);
      pdf *= (1.0 - state.mat.subsurface) * diffuseRatio;
    }
    else
    {
      float primarySpecRatio = 1.0 / (1.0 + state.mat.clearcoat);
      float roughness;
      if(rand(seed) < primarySpecRatio)
        roughness = state.mat.roughness;
      else
        roughness = state.mat.clearcoatRoughness;

      vec3 H = GgxSampling(roughness, r1, r2);
      H      = T * H.x + B * H.y + N * H.z;
      L      = reflect(-V, H);


      // Sample primary specular lobe
      if(rand(seed) < primarySpecRatio)
      {
        // Specular
        brdf = EvalSpecularGltf(state, f0, f90, V, N, L, H, pdf);
        pdf *= primarySpecRatio * specularRatio;  //*(1 - clearcoat * clearcoatFresnel)
      }
      else
      {
        // Clearcoat
        brdf = EvalClearcoatGltf(state, V, N, L, H, pdf);
        pdf *= (1.0 - primarySpecRatio) * specularRatio;
      }
    }

    brdf *= (1.0 - transWeight);
    pdf *= (1.0 - transWeight);
  }

  return brdf;
}

#endif  // PBR_GLTF_GLSL

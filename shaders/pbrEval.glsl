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



#ifndef PBR_EVAL_GLSL
#define PBR_EVAL_GLSL 1

#include "pbrSampling.glsl"


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
vec3 PbrSample(in State state, vec3 V, vec3 N, inout vec3 L, inout float pdf, inout uint seed)
{
  pdf       = 0.0;
  vec3 brdf = vec3(0.0);

  float probability   = rnd(seed);
  float diffuseRatio  = 0.5 * (1.0 - state.mat.metallic);
  float specularRatio = 1.0 - diffuseRatio;
  float transWeight   = (1.0 - state.mat.metallic) * state.mat.transmission;

  float r1 = rnd(seed);
  float r2 = rnd(seed);

  if(rnd(seed) < transWeight)
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
    if(discriminat < 0.0 || rnd(seed) < F)
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
      if(rnd(seed) < primarySpecRatio)
        roughness = state.mat.roughness;
      else
        roughness = state.mat.clearcoatRoughness;

      vec3 H = GgxSampling(roughness, r1, r2);
      H      = T * H.x + B * H.y + N * H.z;
      L      = reflect(-V, H);


      // Sample primary specular lobe
      if(rnd(seed) < primarySpecRatio)
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

#endif  // BRDF_GLSL

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
vec3 BsdfSample(in State state, in vec3 V, in vec3 N, inout uint seed)
{
  vec3 dir;

  float probability  = rnd(seed);
  float diffuseRatio = 0.5 * (1.0 - state.mat.metallic);
  float transWeight  = (1.0 - state.mat.metallic) * state.mat.transmission;

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
      eta = 1.0;  // go through
    }


    // Reflection/Total internal reflection
    if(discriminat < 0.0 || rnd(seed) < F)
    {
      return normalize(reflect(-V, H));
    }
    else
    {
      // Find the pure refractive ray
      vec3 dir = normalize(refract(-V, H, eta));

      // Cought rays perpendicular to surface, and simply continue
      if(isnan(dir.x) || isnan(dir.y) || isnan(dir.z))
      {
        dir = -V;
      }

      return dir;
    }
  }


  vec3 T, B;
  CreateCoordinateSystem(N, T, B);

  if(probability < diffuseRatio)  // sample diffuse
  {
    dir = CosineSampleHemisphere(r1, r2);
    dir = dir.x * T + dir.y * B + dir.z * N;
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
    dir    = reflect(-V, H);
  }

  return dir;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 PbrEval(in State state, vec3 V, vec3 N, vec3 L, inout float pdf)
{
  pdf         = 0;
  float NdotL = dot(N, L);
  float NdotV = dot(N, V);

  float transWeight = (1.0 - state.mat.metallic) * state.mat.transmission;
  if(transWeight > 0)
  {
    pdf = abs(NdotL);
    return state.mat.albedo;
  }

  if(NdotL <= 0.0 || NdotV <= 0.0)
    return vec3(0.0);

  NdotL = clamp(NdotL, 0.001, 1.0);
  NdotV = clamp(abs(NdotV), 0.001, 1.0);

  vec3  H     = normalize(L + V);
  float NdotH = clamp(dot(N, H), 0, 1);
  float LdotH = clamp(dot(L, H), 0, 1);
  float VdotH = clamp(dot(V, H), 0, 1);

  // specular
  vec3 specularCol = state.mat.f0;

  float alphaRoughness   = state.mat.roughness;
  float diffuseRatio     = 0.5 * (1.0 - state.mat.metallic);
  float specularRatio    = 1.0 - diffuseRatio;
  float primarySpecRatio = 1.0 / (1.0 + state.mat.clearcoat);


  // Compute reflectance.
  // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
  float reflectance = max(max(specularCol.r, specularCol.g), specularCol.b);
  vec3  f0          = specularCol.rgb;
  vec3  f90         = vec3(clamp(reflectance * 50.0, 0.0, 1.0));


  // Calculation of analytical lighting contribution
  // Diffuse
  vec3  diffuseContrib = BRDF_lambertian(f0, f90, state.mat.albedo.xyz, VdotH, state.mat.metallic);
  float pdfDiff        = NdotL * M_1_OVER_PI;


  // Specular
  vec3  specContrib;
  float pdfSpec;

  if(state.mat.anisotropy == 0.0)
  {
    specContrib = BRDF_specularGGX(f0, f90, state.mat.roughness, VdotH, NdotL, NdotV, NdotH);
    pdfSpec     = D_GGX(NdotH, alphaRoughness) * NdotH;
  }
  else
  {
    vec3 T = state.tangent;
    vec3 B = state.bitangent;

    float TdotV = clamp(dot(T, V), 0, 1);
    float BdotV = clamp(dot(B, V), 0, 1);
    float TdotL = dot(T, L);
    float BdotL = dot(B, L);
    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    specContrib = BRDF_specularAnisotropicGGX(f0, f90, state.mat.roughness, VdotH, NdotL, NdotV, NdotH, BdotV, TdotV,
                                              TdotL, BdotL, TdotH, BdotH, state.mat.anisotropy);

    float at = max(alphaRoughness * (1.0 + state.mat.anisotropy), 0.001);
    float ab = max(alphaRoughness * (1.0 - state.mat.anisotropy), 0.001);
    pdfSpec  = D_GGX_anisotropic(NdotH, TdotH, BdotH, state.mat.anisotropy, at, ab);
  }
  pdfSpec = primarySpecRatio * pdfSpec / (4.0 * LdotH);

  // Clearcoat
  float clearcoat        = state.mat.clearcoat;
  float clearcoatFresnel = F_Schlick(0.04, 1, VdotH);
  float clearcoatAlpha   = state.mat.clearcoatRoughness * state.mat.clearcoatRoughness;
  float G                = V_GGX(NdotL, NdotV, clearcoatAlpha);
  float D                = D_GGX(NdotH, max(0.001, clearcoatAlpha));
  float f_clearcoat      = clearcoatFresnel * D * G;
  float pdfClearcoat     = (1.0 - primarySpecRatio) * D * NdotH / (4.0 * LdotH);

  // Weight PDF
  pdf = (diffuseRatio * pdfDiff) + (specularRatio * pdfSpec) + (pdfClearcoat * specularRatio);

  return diffuseContrib + specContrib * (1 - clearcoat * clearcoatFresnel) + f_clearcoat * clearcoat;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 PbrSample(in State state, vec3 V, vec3 N, inout vec3 L, inout float pdf, inout uint seed)
{
  L = BsdfSample(state, V, N, seed);

  return PbrEval(state, V, N, L, pdf);
}

#endif  // BRDF_GLSL

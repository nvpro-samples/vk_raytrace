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


#ifndef BRDF_GLSL
#define BRDF_GLSL 1

#include "random.glsl"
#include "sampling.glsl"


// clang-format off
struct Ray { vec3 origin; vec3 direction; };
struct Material { vec4 albedo; float metallic; float roughness; vec3 f0; float anisotropy;};
// clang-format on


//-----------------------------------------------------------------------
float powerHeuristic(float a, float b)
{
  float t = a * a;
  return t / (b * b + t);
}


// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH)
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
float BsdfPdf(in Ray ray, in vec3 N, in vec3 T, in vec3 B, in Material mat, in vec3 bsdfDir)
{
  vec3 V = -ray.direction;
  vec3 L = bsdfDir;

  float alphaRoughness = max(0.001, mat.roughness);

  float diffuseRatio  = 0.5 * (1.0 - mat.metallic);
  float specularRatio = 1.0 - diffuseRatio;

  vec3  H     = normalize(L + V);
  float NdotH = abs(dot(N, H));
  float LdotH = clamp(abs(dot(L, H)), 0.001, 1);
  float NdotL = abs(dot(N, L));

  float pdfDs;

  if(mat.anisotropy > 0)
  {
    float TdotH = dot(T, H);
    float BdotH = dot(B, H);

    float at = max(alphaRoughness * (1.0 + mat.anisotropy), 0.001);
    float ab = max(alphaRoughness * (1.0 - mat.anisotropy), 0.001);
    pdfDs    = D_GGX_anisotropic(NdotH, TdotH, BdotH, mat.anisotropy, at, ab);
  }
  else
  {
    pdfDs = D_GGX(NdotH, alphaRoughness) * NdotH;
  }

  float pdfSpec = pdfDs / (4.0 * LdotH);
  float pdfDiff = NdotL * M_1_PI;

  // Weight PDF
  return diffuseRatio * pdfDiff + specularRatio * pdfSpec;
}


//-----------------------------------------------------------------------
vec3 BsdfSample(in Ray ray, in vec3 N, in vec3 T, in vec3 B, in Material mat, vec3 rndVal)
{
  vec3 V = -ray.direction;

  vec3 dir;

  float probability  = rndVal.x;
  float diffuseRatio = 0.5 * (1.0 - mat.metallic);

  float r1 = rndVal.y;
  float r2 = rndVal.z;

  vec3 tangent_u;  // = T;
  vec3 tangent_v;  // = B;
  CreateCoordinateSystem(N, tangent_u, tangent_v);

  if(probability < diffuseRatio)  // sample diffuse
  {
    dir = CosineSampleHemisphere(r1, r2);
    dir = dir.x * tangent_u + dir.y * tangent_v + dir.z * N;
  }
  else
  {
    float specularAlpha = max(0.001, mat.roughness);

    float phi = r1 * 2.0 * M_PI;

    float cosTheta = sqrt((1.0 - r2) / (1.0 + (specularAlpha * specularAlpha - 1.0) * r2));
    float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
    float sinPhi   = sin(phi);
    float cosPhi   = cos(phi);

    vec3 halfVec = vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
    halfVec      = tangent_u * halfVec.x + tangent_v * halfVec.y + N * halfVec.z;

    dir = 2.0 * dot(V, halfVec) * halfVec - V;
  }
  return dir;
}


//-----------------------------------------------------------------------
vec3 BsdfEval(in Ray ray, in vec3 N, in vec3 T, in vec3 B, in Material mat, in vec3 bsdfDir)
{
  vec3 V = -ray.direction;
  vec3 L = bsdfDir;

  float NdotL = clamp(dot(N, L), 0.001, 1.0);
  float NdotV = clamp(abs(dot(N, V)), 0.001, 1.0);

  if(NdotL <= 0.0 || NdotV <= 0.0)
    return vec3(0.0);

  vec3  H     = normalize(L + V);
  float NdotH = clamp(dot(N, H), 0, 1);
  float LdotH = clamp(dot(L, H), 0, 1);
  float VdotH = clamp(dot(V, H), 0, 1);

  // specular
  vec3 specularCol = mat.f0;

  // Compute reflectance.
  // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
  float reflectance = max(max(specularCol.r, specularCol.g), specularCol.b);
  vec3  f0          = specularCol.rgb;
  vec3  f90         = vec3(clamp(reflectance * 50.0, 0.0, 1.0));


  // Calculation of analytical lighting contribution
  vec3 diffuseContrib = BRDF_lambertian(f0, f90, mat.albedo.xyz, VdotH, mat.metallic);
  vec3 specContrib;
  if(mat.anisotropy == 0.0)
  {
    specContrib = BRDF_specularGGX(f0, f90, mat.roughness, VdotH, NdotL, NdotV, NdotH);
  }
  else
  {

    float TdotV = clamp(dot(T, V), 0, 1);
    float BdotV = clamp(dot(B, V), 0, 1);
    float TdotL = dot(T, L);
    float BdotL = dot(B, L);
    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    specContrib = BRDF_specularAnisotropicGGX(f0, f90, mat.roughness, VdotH, NdotL, NdotV, NdotH, BdotV, TdotV, TdotL,
                                              BdotL, TdotH, BdotH, mat.anisotropy);
  }

  return diffuseContrib + specContrib;
}


//-------------------------------------------------------------------------------------------------
// Specular-Glossiness converter
// See: // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/js/three.pbrUtilities.js#L34
//-------------------------------------------------------------------------------------------------


const float c_MinReflectance = 0.04;

float getPerceivedBrightness(vec3 vector)
{
  return sqrt(0.299 * vector.r * vector.r + 0.587 * vector.g * vector.g + 0.114 * vector.b * vector.b);
}


float solveMetallic(vec3 diffuse, vec3 specular, float oneMinusSpecularStrength)
{
  float specularBrightness = getPerceivedBrightness(specular);

  if(specularBrightness < c_MinReflectance)
  {
    return 0.0;
  }

  float diffuseBrightness = getPerceivedBrightness(diffuse);

  float a = c_MinReflectance;
  float b = diffuseBrightness * oneMinusSpecularStrength / (1.0 - c_MinReflectance) + specularBrightness - 2.0 * c_MinReflectance;
  float c = c_MinReflectance - specularBrightness;
  float D = max(b * b - 4.0 * a * c, 0);

  return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}


#endif  // BRDF_GLSL

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
// This file has all the Disney evaluation and sampling methods.
//

#ifndef PBR_DISNEY_GLSL
#define PBR_DISNEY_GLSL


#include "globals.glsl"
#include "random.glsl"

/*
 * MIT License
 *
 * Copyright(c) 2019-2021 Asif Ali
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this softwareand associated documentation files(the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions :
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* References:
 * [1] https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
 * [2] https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
 * [3] https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
 * [4] https://github.com/mmacklin/tinsel/blob/master/src/disney.h
 * [5] http://simon-kallweit.me/rendercompo2015/report/
 * [6] http://shihchinw.github.io/2015/07/implementing-disney-principled-brdf-in-arnold.html
 * [7] https://github.com/mmp/pbrt-v4/blob/0ec29d1ec8754bddd9d667f0e80c4ff025c900ce/src/pbrt/bxdfs.cpp#L76-L286
 * [8] https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
 */


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 ImportanceSampleGTR1(float rgh, float r1, float r2)
{
  float a  = max(0.001, rgh);
  float a2 = a * a;

  float phi = r1 * TWO_PI;

  float cosTheta = sqrt((1.0 - pow(a2, 1.0 - r1)) / (1.0 - a2));
  float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
  float sinPhi   = sin(phi);
  float cosPhi   = cos(phi);

  return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 ImportanceSampleGTR2_aniso(float ax, float ay, float r1, float r2)
{
  float phi = r1 * TWO_PI;

  float sinPhi   = ay * sin(phi);
  float cosPhi   = ax * cos(phi);
  float tanTheta = sqrt(r2 / (1 - r2));

  return vec3(tanTheta * cosPhi, tanTheta * sinPhi, 1.0);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 ImportanceSampleGTR2(float rgh, float r1, float r2)
{
  float a = max(0.001, rgh);

  float phi = r1 * TWO_PI;

  float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
  float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
  float sinPhi   = sin(phi);
  float cosPhi   = cos(phi);

  return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float SchlickFresnel(float u)
{
  float m  = clamp(1.0 - u, 0.0, 1.0);
  float m2 = m * m;
  return m2 * m2 * m;  // pow(m,5)
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float DielectricFresnel(float cos_theta_i, float eta)
{
  float sinThetaTSq = eta * eta * (1.0f - cos_theta_i * cos_theta_i);

  // Total internal reflection
  if(sinThetaTSq > 1.0)
    return 1.0;

  float cos_theta_t = sqrt(max(1.0 - sinThetaTSq, 0.0));

  float rs = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);
  float rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

  return 0.5f * (rs * rs + rp * rp);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float GTR1(float NdotH, float a)
{
  if(a >= 1.0)
    return M_1_OVER_PI;  //(1.0 / PI);
  float a2 = a * a;
  float t  = 1.0 + (a2 - 1.0) * NdotH * NdotH;
  return (a2 - 1.0) / (PI * log(a2) * t);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float GTR2(float NdotH, float a)
{
  float a2 = a * a;
  float t  = 1.0 + (a2 - 1.0) * NdotH * NdotH;
  return a2 / (PI * t * t);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
  float a = HdotX / ax;
  float b = HdotY / ay;
  float c = a * a + b * b + NdotH * NdotH;
  return 1.0 / (PI * ax * ay * c * c);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float SmithG_GGX(float NdotV, float alphaG)
{
  float a = alphaG * alphaG;
  float b = NdotV * NdotV;
  return 1.0 / (NdotV + sqrt(a + b - a * b));
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float SmithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
  float a = VdotX * ax;
  float b = VdotY * ay;
  float c = NdotV;
  return 1.0 / (NdotV + sqrt(a * a + b * b + c * c));
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 CosineSampleHemisphere(float r1, float r2)
{
  vec3  dir;
  float r   = sqrt(r1);
  float phi = TWO_PI * r2;
  dir.x     = r * cos(phi);
  dir.y     = r * sin(phi);
  dir.z     = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));

  return dir;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 UniformSampleHemisphere(float r1, float r2)
{
  float r   = sqrt(max(0.0, 1.0 - r1 * r1));
  float phi = TWO_PI * r2;

  return vec3(r * cos(phi), r * sin(phi), r1);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 UniformSampleSphere(float r1, float r2)
{
  float z   = 1.0 - 2.0 * r1;
  float r   = sqrt(max(0.0, 1.0 - z * z));
  float phi = TWO_PI * r2;

  return vec3(r * cos(phi), r * sin(phi), z);
}

//-----------------------------------------------------------------------
float powerHeuristic(float a, float b)
//-----------------------------------------------------------------------
{
  float t = a * a;
  return t / (b * b + t);
}

//const int numOfLights = 0;
////-----------------------------------------------------------------------
////-----------------------------------------------------------------------
//void sampleSphereLight(in Light light, inout LightSampleRec lightSampleRec, in vec2 rand)
//{
//  float r1 = rand.x;
//  float r2 = rand.y;
//
//  lightSampleRec.surfacePos = light.position + UniformSampleSphere(r1, r2) * light.radius;
//  lightSampleRec.normal     = normalize(lightSampleRec.surfacePos - light.position);
//  lightSampleRec.emission   = light.emission * float(numOfLights);
//}
//
////-----------------------------------------------------------------------
////-----------------------------------------------------------------------
//void sampleRectLight(in Light light, inout LightSampleRec lightSampleRec, in vec2 rand)
//{
//  float r1 = rand.x;
//  float r2 = rand.y;
//
//  lightSampleRec.surfacePos = light.position + light.u * r1 + light.v * r2;
//  lightSampleRec.normal     = normalize(cross(light.u, light.v));
//  lightSampleRec.emission   = light.emission * float(numOfLights);
//}
//
////-----------------------------------------------------------------------
////-----------------------------------------------------------------------
//void sampleLight(in Light light, inout LightSampleRec lightSampleRec, in vec2 rand)
//{
//  if(int(light.type) == 0)  // Rect Light
//    sampleRectLight(light, lightSampleRec, rand);
//  else
//    sampleSphereLight(light, lightSampleRec, rand);
//}

#ifdef _ENVMAP_
#ifndef CONSTANT_BG

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float EnvPdf(in Ray r)
{
  float theta = acos(clamp(r.direction.y, -1.0, 1.0));
  vec2  uv    = vec2((PI + atan(r.direction.z, r.direction.x)) * (1.0 / TWO_PI), theta * (1.0 / PI));
  float pdf   = texture(hdrCondDistTex, uv).y * texture(hdrMarginalDistTex, vec2(uv.y, 0.)).y;
  return (pdf * hdrResolution) / (2.0 * PI * PI * sin(theta));
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec4 EnvSample(inout vec3 color)
{
  float r1 = rand();
  float r2 = rand();

  float v = texture(hdrMarginalDistTex, vec2(r1, 0.)).x;
  float u = texture(hdrCondDistTex, vec2(r2, v)).x;

  color     = texture(hdrTex, vec2(u, v)).xyz * hdrMultiplier;
  float pdf = texture(hdrCondDistTex, vec2(u, v)).y * texture(hdrMarginalDistTex, vec2(v, 0.)).y;

  float phi   = u * TWO_PI;
  float theta = v * PI;

  if(sin(theta) == 0.0)
    pdf = 0.0;

  return vec4(-sin(theta) * cos(phi), cos(theta), -sin(theta) * sin(phi), (pdf * hdrResolution) / (2.0 * PI * PI * sin(theta)));
}

#endif
#endif

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EmitterSample(in Ray r, in State state, in LightSampleRec lightSampleRec, in BsdfSampleRec bsdfSampleRec)
{
  vec3 Le;

  if(state.depth == 0 || state.specularBounce)
    Le = lightSampleRec.emission;
  else
    Le = powerHeuristic(bsdfSampleRec.pdf, lightSampleRec.pdf) * lightSampleRec.emission;

  return Le;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalDielectricReflection(State state, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
  if(dot(N, L) < 0.0)
    return vec3(0.0);

  float F = DielectricFresnel(dot(V, H), state.eta);
  float D = GTR2(dot(N, H), state.mat.roughness);

  pdf = D * dot(N, H) * F / (4.0 * dot(V, H));

  float G = SmithG_GGX(abs(dot(N, L)), state.mat.roughness) * SmithG_GGX(dot(N, V), state.mat.roughness);
  return state.mat.albedo * F * D * G;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalDielectricRefraction(State state, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
  float F = DielectricFresnel(abs(dot(V, H)), state.eta);
  float D = GTR2(dot(N, H), state.mat.roughness);

  float denomSqrt = dot(L, H) * state.eta + dot(V, H);
  pdf             = D * dot(N, H) * (1.0 - F) * abs(dot(L, H)) / (denomSqrt * denomSqrt);

  float G = SmithG_GGX(abs(dot(N, L)), state.mat.roughness) * SmithG_GGX(dot(N, V), state.mat.roughness);
  return state.mat.albedo * (1.0 - F) * D * G * abs(dot(V, H)) * abs(dot(L, H)) * 4.0 * state.eta * state.eta
         / (denomSqrt * denomSqrt);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalSpecular(State state, vec3 Cspec0, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
  if(dot(N, L) < 0.0)
    return vec3(0.0);

  float D = GTR2_aniso(dot(N, H), dot(H, state.tangent), dot(H, state.bitangent), state.mat.ax, state.mat.ay);
  pdf     = D * dot(N, H) / (4.0 * dot(V, H));

  float FH = SchlickFresnel(dot(L, H));
  vec3  F  = mix(Cspec0, vec3(1.0), FH);
  float G  = SmithG_GGX_aniso(dot(N, L), dot(L, state.tangent), dot(L, state.bitangent), state.mat.ax, state.mat.ay);
  G *= SmithG_GGX_aniso(dot(N, V), dot(V, state.tangent), dot(V, state.bitangent), state.mat.ax, state.mat.ay);
  return F * D * G;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalClearcoat(State state, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
  if(dot(N, L) < 0.0)
    return vec3(0.0);

  float D = GTR1(dot(N, H), state.mat.clearcoatRoughness);
  pdf     = D * dot(N, H) / (4.0 * dot(V, H));

  float FH = SchlickFresnel(dot(L, H));
  float F  = mix(0.04, 1.0, FH);
  float G  = SmithG_GGX(dot(N, L), 0.25) * SmithG_GGX(dot(N, V), 0.25);
  return vec3(0.25 * state.mat.clearcoat * F * D * G);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalDiffuse(State state, vec3 Csheen, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
  if(dot(N, L) < 0.0)
    return vec3(0.0);

  pdf = dot(N, L) * (1.0 / PI);

  float FL     = SchlickFresnel(dot(N, L));
  float FV     = SchlickFresnel(dot(N, V));
  float FH     = SchlickFresnel(dot(L, H));
  float Fd90   = 0.5 + 2.0 * dot(L, H) * dot(L, H) * state.mat.roughness;
  float Fd     = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);
  vec3  Fsheen = FH * state.mat.sheen * Csheen;
  return ((1.0 / PI) * Fd * (1.0 - state.mat.subsurface) * state.mat.albedo + Fsheen) * (1.0 - state.mat.metallic);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalSubsurface(State state, vec3 V, vec3 N, vec3 L, inout float pdf)
{
  pdf = (1.0 / TWO_PI);

  float FL = SchlickFresnel(abs(dot(N, L)));
  float FV = SchlickFresnel(dot(N, V));
  float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
  return sqrt(state.mat.albedo) * state.mat.subsurface * (1.0 / PI) * Fd * (1.0 - state.mat.metallic) * (1.0 - state.mat.transmission);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 DisneySample(inout State state, vec3 V, vec3 N, inout vec3 L, inout float pdf, inout RngStateType seed)
{
  state.isSubsurface = false;
  pdf                = 0.0;
  vec3 f             = vec3(0.0);

  float r1 = rand(seed);
  float r2 = rand(seed);

  float diffuseRatio = 0.5 * (1.0 - state.mat.metallic);
  float transWeight  = (1.0 - state.mat.metallic) * state.mat.transmission;

  vec3  Cdlin = state.mat.albedo;
  float Cdlum = 0.3 * Cdlin.x + 0.6 * Cdlin.y + 0.1 * Cdlin.z;  // luminance approx.

  vec3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : vec3(1.0f);  // normalize lum. to isolate hue+sat
  vec3 Cspec0 = mix(state.mat.specular * 0.08 * mix(vec3(1.0), Ctint, state.mat.specularTint), Cdlin, state.mat.metallic);
  vec3 Csheen = state.mat.sheenTint;  //mix(vec3(1.0), Ctint, state.mat.sheenTint);

  // BSDF
  if(rand(seed) < transWeight)
  {
    vec3 H = ImportanceSampleGTR2(state.mat.roughness, r1, r2);
    H      = state.tangent * H.x + state.bitangent * H.y + N * H.z;

    vec3  R = reflect(-V, H);
    float F = DielectricFresnel(abs(dot(R, H)), state.eta);

    if(state.mat.thinwalled)
    {
      if(dot(state.ffnormal, state.normal) < 0.0)
        F = 0;
      state.eta = 1.001;
    }

    // Reflection/Total internal reflection
    if(rand(seed) < F)
    {
      L = normalize(R);
      f = EvalDielectricReflection(state, V, N, L, H, pdf);
    }
    else  // Transmission
    {
      L = normalize(refract(-V, H, state.eta));
      f = EvalDielectricRefraction(state, V, N, L, H, pdf);
    }

    f *= transWeight;
    pdf *= transWeight;
  }
  else  // BRDF
  {
    if(rand(seed) < diffuseRatio)
    {
      // Diffuse transmission. A way to approximate subsurface scattering
      if(rand(seed) < state.mat.subsurface)
      {
        L = UniformSampleHemisphere(r1, r2);
        L = state.tangent * L.x + state.bitangent * L.y - N * L.z;

        f = EvalSubsurface(state, V, N, L, pdf);
        pdf *= state.mat.subsurface * diffuseRatio;

        state.isSubsurface = true;  // Required when sampling lights from inside surface
      }
      else  // Diffuse
      {
        L = CosineSampleHemisphere(r1, r2);
        L = state.tangent * L.x + state.bitangent * L.y + N * L.z;

        vec3 H = normalize(L + V);

        f = EvalDiffuse(state, Csheen, V, N, L, H, pdf);
        pdf *= (1.0 - state.mat.subsurface) * diffuseRatio;
      }
    }
    else  // Specular
    {
      float primarySpecRatio = 1.0 / (1.0 + state.mat.clearcoat);

      // Sample primary specular lobe
      if(rand(seed) < primarySpecRatio)
      {
        // TODO: Implement http://jcgt.org/published/0007/04/01/
        vec3 H = ImportanceSampleGTR2_aniso(state.mat.ax, state.mat.ay, r1, r2);
        H      = state.tangent * H.x + state.bitangent * H.y + N * H.z;
        L      = normalize(reflect(-V, H));

        f = EvalSpecular(state, Cspec0, V, N, L, H, pdf);
        pdf *= primarySpecRatio * (1.0 - diffuseRatio);
      }
      else  // Sample clearcoat lobe
      {
        vec3 H = ImportanceSampleGTR1(state.mat.clearcoatRoughness, r1, r2);
        H      = state.tangent * H.x + state.bitangent * H.y + N * H.z;
        L      = normalize(reflect(-V, H));

        f = EvalClearcoat(state, V, N, L, H, pdf);
        pdf *= (1.0 - primarySpecRatio) * (1.0 - diffuseRatio);
      }
    }

    f *= (1.0 - transWeight);
    pdf *= (1.0 - transWeight);
  }
  return f;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 DisneyEval(State state, vec3 V, vec3 N, vec3 L, inout float pdf)
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
    // Transmission
    if(dot(N, L) < 0.0)
    {
      bsdf = EvalDielectricRefraction(state, V, N, L, H, bsdfPdf);
    }
    else  // Reflection
    {
      bsdf = EvalDielectricReflection(state, V, N, L, H, bsdfPdf);
    }
  }

  float m_pdf;

  if(transWeight < 1.0)
  {
    // Subsurface
    if(dot(N, L) < 0.0)
    {
      // TODO: Double check this. Fails furnace test when used with rough transmission
      if(state.mat.subsurface > 0.0)
      {
        brdf    = EvalSubsurface(state, V, N, L, m_pdf);
        brdfPdf = m_pdf * state.mat.subsurface * diffuseRatio;
      }
    }
    // BRDF
    else
    {
      vec3  Cdlin = state.mat.albedo;
      float Cdlum = 0.3 * Cdlin.x + 0.6 * Cdlin.y + 0.1 * Cdlin.z;  // luminance approx.

      vec3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : vec3(1.0f);  // normalize lum. to isolate hue+sat
      vec3 Cspec0 = mix(state.mat.specular * 0.08 * mix(vec3(1.0), Ctint, state.mat.specularTint), Cdlin, state.mat.metallic);
      vec3 Csheen = state.mat.sheenTint;  //mix(vec3(1.0), Ctint, state.mat.sheenTint);

      // Diffuse
      brdf += EvalDiffuse(state, Csheen, V, N, L, H, m_pdf);
      brdfPdf += m_pdf * (1.0 - state.mat.subsurface) * diffuseRatio;

      // Specular
      brdf += EvalSpecular(state, Cspec0, V, N, L, H, m_pdf);
      brdfPdf += m_pdf * primarySpecRatio * (1.0 - diffuseRatio);

      // Clearcoat
      brdf += EvalClearcoat(state, V, N, L, H, m_pdf);
      brdfPdf += m_pdf * (1.0 - primarySpecRatio) * (1.0 - diffuseRatio);
    }
  }

  pdf = mix(brdfPdf, bsdfPdf, transWeight);
  return mix(brdf, bsdf, transWeight);
}


#endif  // PBR_DISNEY_GLSL

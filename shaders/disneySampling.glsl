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


#ifndef DISNEY_SAMPLING_GLSL
#define DISNEY_SAMPLING_GLSL


#include "globals.glsl"


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


#endif  // DISNEY_SAMPLING_GLSL

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

#version 450
layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;
layout(set = 0, binding = 0) uniform sampler2D inImage;


layout(push_constant) uniform _ShaderInfo
{
  float key;
  float Ywhite;
  float sat;
  float invGamma;
  float avgLum;
};


const mat3 RGB2XYZ = mat3(0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041);

const vec3 lum = vec3(0.2126f, 0.7152f, 0.0722f);

float luminance(vec3 color)
{
  return dot(color, lum);  //color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
}

vec3 tonemap(vec3 RGB, float logAvgLum)
{
  vec3  XYZ = RGB2XYZ * RGB;
  float Y   = (key / logAvgLum) * XYZ.y;
  float Yd  = (Y * (1.0 + Y / (Ywhite * Ywhite))) / (1.0 + Y);
  return pow(RGB / XYZ.y, vec3(sat, sat, sat)) * Yd;
}


float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x)
{
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemapFilmic(vec3 texColor, float logAvgLum)
{
  texColor *= logAvgLum;  // Exposure Adjustment

  vec3 curr       = Uncharted2Tonemap(texColor);
  vec3 whiteScale = 1.0f / Uncharted2Tonemap(vec3(W));

  return curr * whiteScale;
}


vec3 whiteBalance(in vec3 color)
{
  vec3 whitebalance = vec3(1, 1, 1);

  vec3  wb = 1.0 / whitebalance;
  float l  = luminance(wb);
  return color *= wb / l;
}


void main()
{
  vec4 hdr      = texture(inImage, outUV).rgba;
  vec3 ldrColor = tonemap(hdr.rgb, avgLum);
  // ldrColor      = tonemapFilmic(hdr.rgb, avgLum);
  fragColor.rgb = pow(ldrColor, vec3(invGamma));
  fragColor.a   = hdr.a;
}

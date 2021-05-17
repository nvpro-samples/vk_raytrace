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


#version 450
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec2 uvCoords;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D inImage;


layout(push_constant) uniform _ShaderInfo
{
  float brightness;
  float contrast;
  float saturation;
  float vignette;
  float avgLum;
  float zoom;
  vec2  renderingRatio;
};

#define TONEMAP_UNCHARTED
#include "random.glsl"
#include "tonemapping.glsl"

// http://www.thetenthplanet.de/archives/5367
// Apply dithering to hide banding artifacts.
vec3 dither(vec3 linear_color, vec3 noise, float quant)
{
  vec3 c0    = floor(linearTosRGB(linear_color) / quant) * quant;
  vec3 c1    = c0 + quant;
  vec3 discr = mix(sRGBToLinear(c0), sRGBToLinear(c1), noise);
  return mix(c0, c1, lessThan(discr, linear_color));
}

void main()
{
  vec4 hdr = texture(inImage, uvCoords * zoom).rgba;

  vec3 color = toneMap(hdr.rgb, avgLum);
  vec3 noise = pcg_div * vec3(pcg3d(uvec3(gl_FragCoord.xy, 0)));

  color = dither(sRGBToLinear(color), noise, 1. / 255.);

  //contrast
  color = clamp(mix(vec3(0.5), color, contrast), 0, 1);
  // brighness
  color = pow(color, vec3(1.0 / brightness));
  // saturation
  vec3 i = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
  color  = mix(i, color, saturation);
  // vignette
  vec2 uv = ((uvCoords * renderingRatio) - 0.5) * 2.0;
  color *= 1.0 - dot(uv, uv) * vignette;

  fragColor.xyz = color;
  fragColor.a   = hdr.a;
}

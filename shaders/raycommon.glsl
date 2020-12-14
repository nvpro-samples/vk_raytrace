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

#ifndef RAYCOMMON_GLSL
#define RAYCOMMON_GLSL

// Structures and constants : Hitpayload
#include "globals.glsl"

// cCamera matrices
#include "../structures.h"


// Using rnd in initialize Payload
#include "random.glsl"


// Debugging
vec3 IntegerToColor(uint val)
{
  const vec3 freq = vec3(1.33333f, 2.33333f, 3.33333f);
  return vec3(sin(freq * val) * .5 + .5);
}


#ifdef USE_STOREIMAGE
void StoreResult(in image2D img, in ivec2 coord, in int frameNb, in vec3 result)
{
  // Do accumulation over time
  if(frameNb > 0)
  {
    float a         = 1.0f / float(frameNb + 1);
    vec3  old_color = imageLoad(img, coord).xyz;
    imageStore(img, coord, vec4(mix(old_color, result, a), 1.f));
  }
  else
  {
    // First frame, replace the value in the buffer
    imageStore(img, coord, vec4(result, 1.f));
  }
}
#endif

#ifdef USE_INIT_PAYLOAD

// Using clockARB
#extension GL_ARB_shader_clock : enable

HitPayload InitializePayload(in ivec2 coordImage, in ivec2 sizeImage, in CameraMatrices cam, in int frame)
{
  // Initialize the random number
  uvec2 s    = pcg2d(coordImage * int(clockARB()));
  uint  seed = s.x + s.y;

  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  vec2 subpixel_jitter = frame == 0 ? vec2(0.5f, 0.5f) : rnd2(seed);

  // Compute sampling position between [-1 .. 1]
  const vec2 pixelCenter = vec2(coordImage) + subpixel_jitter;
  const vec2 inUV        = pixelCenter / vec2(sizeImage.xy);
  vec2       d           = inUV * 2.0 - 1.0;

  // Compute ray origin and direction
  vec4 origin    = cam.viewInverse * vec4(0, 0, 0, 1);
  vec4 target    = cam.projInverse * vec4(d.x, d.y, 1, 1);
  vec4 direction = cam.viewInverse * vec4(normalize(target.xyz), 0);

  // Payload default values
  HitPayload prd;
  prd.contribution = vec3(0);
  prd.seed         = seed;
  prd.rayOrigin    = origin.xyz;
  prd.rayDirection = direction.xyz;
  prd.weight       = vec3(1);
  prd.last_pdf     = -1.f;
  prd.flags        = FLAG_FIRST_PATH_SEGMENT;

  return prd;
}
#endif


#define SRGB_FAST_APPROXIMATION 1
// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec4 SRGBtoLINEAR(vec4 srgbIn)
{
#ifdef SRGB_FAST_APPROXIMATION
  vec3 linOut = pow(srgbIn.xyz, vec3(2.2));
#else   //SRGB_FAST_APPROXIMATION
  vec3 bLess  = step(vec3(0.04045), srgbIn.xyz);
  vec3 linOut = mix(srgbIn.xyz / vec3(12.92), pow((srgbIn.xyz + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
#endif  //SRGB_FAST_APPROXIMATION
  return vec4(linOut, srgbIn.w);
}


#endif  // RAYCOMMON_GLSL

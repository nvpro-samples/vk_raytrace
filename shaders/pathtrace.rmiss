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

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : enable
// Align structure layout to scalar
#extension GL_EXT_scalar_block_layout : enable

#include "layouts.glsl"
#include "sampling.glsl"


layout(location = 0) rayPayloadInEXT HitPayload prd;

// Push Constant
layout(push_constant) uniform _RtCoreState
{
  RtState rtstate;
};


///////////////////////////////////////////
vec3 environmentEval(in vec3 dir)
{
  vec3 radiance;

  if(_sunAndSky.in_use == 1)
    radiance = sun_and_sky(_sunAndSky, dir);
  else
  {
    vec2 uv  = GetSphericalUv(dir);  // See sampling.glsl
    radiance = texture(environmentTexture, uv).rgb;
  }
  return radiance.xyz;
}


void main()
{
  prd.contribution = environmentEval(gl_WorldRayDirectionEXT.xyz) * rtstate.environment_intensity_factor;
  prd.contribution *= prd.weight;
  prd.flags = FLAG_DONE;
}

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

#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL


#include "globals.glsl"

//-------------------------------------------------------------------------------------------------
// Environment Sampling
// See:  https://arxiv.org/pdf/1901.05423.pdf
//-------------------------------------------------------------------------------------------------

struct Environment_sample_data
{
  uint  alias;
  float q;
  float pdf;
};

Environment_sample_data GetSampleData(sampler2D sample_buffer, ivec2 idx)
{
  vec3 data = texelFetch(sample_buffer, idx, 0).xyz;

  Environment_sample_data sample_data;
  sample_data.alias = floatBitsToInt(data.x);
  sample_data.q     = data.y;
  sample_data.pdf   = data.z;
  return sample_data;
}

Environment_sample_data GetSampleData(sampler2D sample_buffer, uint idx)
{
  uvec2 size = textureSize(sample_buffer, 0);
  uint  px   = idx % size.x;
  uint  py   = idx / size.x;
  return GetSampleData(sample_buffer, ivec2(px, size.y - py - 1));  // Image is upside down
}

float Environment_pdf(sampler2D sample_buffer, vec2 uv)
{
  uvec2 size = textureSize(sample_buffer, 0);
  return texelFetch(sample_buffer, ivec2(uv.x * size.x, (1 - uv.y) * size.y), 0).z;
}


vec3 Environment_sample(sampler2D lat_long_tex, sampler2D sample_buffer, in vec3 randVal, out vec3 to_light, out float pdf)
{
  vec3 xi = randVal;

  uvec2 tsize  = textureSize(lat_long_tex, 0);
  uint  width  = tsize.x;
  uint  height = tsize.y;

  const uint size = width * height;
  const uint idx  = min(uint(xi.x * float(size)), size - 1);

  Environment_sample_data sample_data = GetSampleData(sample_buffer, idx);

  uint env_idx;
  if(xi.y < sample_data.q)
  {
    env_idx = idx;
    xi.y /= sample_data.q;
  }
  else
  {
    env_idx = sample_data.alias;
    xi.y    = (xi.y - sample_data.q) / (1.0f - sample_data.q);
  }

  uint       py = env_idx / width;
  const uint px = env_idx % width;
  pdf           = GetSampleData(sample_buffer, env_idx).pdf;
  py            = height - py - 1;  // Image is upside down


  // uniformly sample spherical area of pixel
  const float u       = float(px + xi.y) / float(width);
  const float phi     = u * (2.0f * M_PI) - M_PI;
  float       sin_phi = sin(phi);
  float       cos_phi = cos(phi);

  const float step_theta = M_PI / float(height);
  const float theta0     = float(py) * step_theta;
  const float cos_theta  = cos(theta0) * (1.0f - xi.z) + cos(theta0 + step_theta) * xi.z;
  const float theta      = acos(cos_theta);
  const float sin_theta  = sin(theta);
  to_light               = vec3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

  // lookup filtered value
  const float v = theta * M_1_OVER_PI;
  return texture(lat_long_tex, vec2(u, v)).xyz;
}


//-----------------------------------------------------------------------
// Return the tangent and binormal from the incoming normal
//-----------------------------------------------------------------------
void CreateCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
  // http://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors.html#CoordinateSystemfromaVector
  //if(abs(N.x) > abs(N.y))
  //  Nt = vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z);
  //else
  //  Nt = vec3(0, N.z, -N.y) / sqrt(N.y * N.y + N.z * N.z);
  //Nb = cross(N, Nt);

  Nt = normalize(((abs(N.z) > 0.99999f) ? vec3(-N.x * N.y, 1.0f - N.y * N.y, -N.y * N.z) :
                                          vec3(-N.x * N.z, -N.y * N.z, 1.0f - N.z * N.z)));
  Nb = cross(Nt, N);
}


//-----------------------------------------------------------------------
// Return the UV in a lat-long HDR map
//-----------------------------------------------------------------------
vec2 GetSphericalUv(vec3 v)
{
  float gamma = asin(-v.y);
  float theta = atan(v.z, v.x);

  vec2 uv = vec2(theta * M_1_OVER_PI * 0.5, gamma * M_1_OVER_PI) + 0.5;
  return uv;
}


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-----------------------------------------------------------------------
vec3 OffsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}


//-----------------------------------------------------------------------
// Sampling the HDR environment or scene lights
//-----------------------------------------------------------------------
vec4 EnvSample(inout vec3 radiance)
{
  vec3  lightDir;
  float pdf;

  // Sun & Sky or HDR
  if(_sunAndSky.in_use == 1)
  {
    // #TODO: find proper light direction + PDF
    vec3 T, B;
    CreateCoordinateSystem(_sunAndSky.sun_direction, T, B);
    vec3 dir;
    dir.x = rnd(prd.seed) * 0.1;
    dir.y = rnd(prd.seed) * 0.1;
    dir.z = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));

    lightDir = normalize(T * dir.x + B * dir.y + _sunAndSky.sun_direction * dir.z);
    radiance = sun_and_sky(_sunAndSky, lightDir);
    pdf      = 0.5;
  }
  else
  {
    vec3 randVal = vec3(rnd(prd.seed), rnd(prd.seed), rnd(prd.seed));

    // Sampling the HDR with importance sampling
    radiance = Environment_sample(  //
        environmentTexture,         // assuming lat long map
        environmentSamplingData,    // importance sampling data of the environment map
        randVal, lightDir, pdf);
  }

  radiance *= rtxState.hdrMultiplier;
  return vec4(lightDir, pdf);
}


#endif  // SAMPLING_GLSL

//! #version 430

/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

const float ONE_OVER_PI = 0.3183099;
const float PI          = 3.141592653589;

//-------------------------------------------------------------------------------------------------
// random number generator based on the Optix SDK
//-------------------------------------------------------------------------------------------------

uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

uint lcg2(inout uint prev)
{
  prev = (prev * 8121 + 28411) % 134456;
  return prev;
}

// Generate random float in [0, 1)
float rnd(inout uint prev)
{
  return (float(lcg(prev)) / float(0x01000000));
}


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-------------------------------------------------------------------------------------------------

vec3 offsetRay(in vec3 p, in vec3 n)
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


//-------------------------------------------------------------------------------------------------
// Environment
//-------------------------------------------------------------------------------------------------

struct Environment_sample_data
{
  uint  alias;
  float q;
  float pdf;
};

Environment_sample_data getSampleData(sampler2D sample_buffer, ivec2 idx)
{
  vec3 data = texelFetch(sample_buffer, idx, 0).xyz;

  Environment_sample_data sample_data;
  sample_data.alias = floatBitsToInt(data.x);
  sample_data.q     = data.y;
  sample_data.pdf   = data.z;
  return sample_data;
}

Environment_sample_data getSampleData(sampler2D sample_buffer, uint idx)
{
  uvec2 size = textureSize(sample_buffer, 0);
  uint  px   = idx % size.x;
  uint  py   = idx / size.x;
  return getSampleData(sample_buffer, ivec2(px, size.y - py - 1));  // Image is upside down
}


vec3 environment_sample(sampler2D lat_long_tex, sampler2D sample_buffer, inout uint seed, out vec3 to_light, out float pdf)
{
  const float environment_intensity_factor = 1.0f;
  const float M_PI                         = 3.141592653589793;
  const float M_ONE_OVER_PI                = 0.318309886183790671538;

  vec3 xi;
  xi.x = rnd(seed);
  xi.y = rnd(seed);
  xi.z = rnd(seed);

  uvec2 tsize  = textureSize(lat_long_tex, 0);
  uint  width  = tsize.x;
  uint  height = tsize.y;

  const uint size = width * height;
  const uint idx  = min(uint(xi.x * float(size)), size - 1);

  Environment_sample_data sample_data = getSampleData(sample_buffer, idx);

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
  pdf           = getSampleData(sample_buffer, env_idx).pdf;
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
  const float v = theta * M_ONE_OVER_PI;
  return environment_intensity_factor * texture(lat_long_tex, vec2(u, v)).xyz;
}


//-------------------------------------------------------------------------------------------------
// Environment Sampling / Convertion
//-------------------------------------------------------------------------------------------------

// Return the UV in a lat-long HDR map
vec2 get_spherical_uv(vec3 v)
{
  float gamma = asin(-v.y);
  float theta = atan(v.z, v.x);

  vec2 uv = vec2(theta * ONE_OVER_PI * 0.5, gamma * ONE_OVER_PI) + 0.5;
  return uv;
}


// Sampling hemisphere around +Z
void cosine_sample_hemisphere(const float u1, const float u2, out vec3 p)
{
  // Uniformly sample disk.
  const float r   = sqrt(u1);
  const float phi = 2.0f * M_PIf * u2;
  p.x             = r * cos(phi);
  p.y             = r * sin(phi);

  // Project up to hemisphere.
  p.z = sqrt(max(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

// Sampling hemisphere around +Z within a cone (angle)
void cosineMaxSampleHemisphere(const float r1, const float r2, const float angle, out vec3 d)
{
  float cosAngle = cos(angle);
  float phi      = 2.0f * M_PIf * r1;
  float r        = sqrt(1.0 - ((1.0 - r2 * (1.0 - cosAngle)) * (1.0 - r2 * (1.0 - cosAngle))));
  d.x            = cos(phi) * r;
  d.y            = sin(phi) * r;
  d.z            = 1.0 - r2 * (1.0 - cosAngle);
}


// Transform P to the domaine made by N-T-B
void inverse_transform(inout vec3 p, in vec3 normal, in vec3 tangent, in vec3 binormal)
{
  p = p.x * tangent + p.y * binormal + p.z * normal;
}

// Return the basis from the incoming normal: x=tangent, y=binormal, z=normal
void compute_default_basis(const vec3 normal, out vec3 x, out vec3 y, out vec3 z)
{
  // ZAP's default coordinate system for compatibility
  z              = normal;
  const float yz = -z.y * z.z;
  y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));
  x = cross(y, z);
}

// Randomly sampling around +Z
vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(1.0 - r2);

  vec3 direction = vec3(cos(2 * M_PIf * r1) * sq, sin(2 * M_PIf * r1) * sq, sqrt(r2));
  direction      = direction.x * x + direction.y * y + direction.z * z;
  seed++;

  return direction;
}


// Return the tangent and binormal from the incoming normal
void createCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
  if(abs(N.x) > abs(N.y))
    Nt = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
  else
    Nt = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
  Nb = cross(N, Nt);
}


// Uv range: [0, 1]
vec3 toPolar(in vec2 uv)
{
  float theta = 2.0 * PI * uv.x + -PI / 2.0;
  float phi   = PI * uv.y;

  vec3 n;
  n.x = cos(theta) * sin(phi);
  n.y = sin(theta) * sin(phi);
  n.z = cos(phi);

  //n = normalize(n);
  return n;
}

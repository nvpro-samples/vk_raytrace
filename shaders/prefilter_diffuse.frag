/******************************************************************************
 * Copyright 2018 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/
#version 450

//-------------------------------------------------------------------------------------------------
// This shader computes a diffuse irradiance IBL map using multiple importance sampling weighted
// hemisphere sampling and environment map importance sampling.
//-------------------------------------------------------------------------------------------------

// varying inputs
layout(location = 0) in vec3 world_position;
layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform sampler2D env_tex;
layout(binding = 1) uniform sampler2D env_accel_tex;


const float PI          = 3.14159265359;
const float ONE_OVER_PI = 0.3183099;

vec2 get_spherical_uv(vec3 v)
{
  float gamma = asin(v.y);
  float theta = atan(v.z, v.x);

  return vec2(theta * ONE_OVER_PI * 0.5, gamma * ONE_OVER_PI) + 0.5;
}

// Simple pseudo-random numbers.
uint rnd_next(uint state)
{
  // xorshift32
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

float rnd_val(uint state)
{

  return float(state) * 2.3283064365386963e-10;  // / 0x100000000
}
int hash(int seed, int i)
{
  return (i ^ seed) * 1075385539;
}
int rnd_init(ivec3 pos)
{
  return hash(hash(hash(0, pos.x), pos.y), pos.z);
}

struct Envmap_sample_value
{
  vec3  dir;
  vec3  value;
  float pdf;
};

Envmap_sample_value envmap_sample(vec3 xi)
{
  // Importance sample an envmap pixel using an alias map.
  ivec2               env_size = textureSize(env_accel_tex, 0);
  int                 size     = env_size.x * env_size.y;
  int                 idx      = min(int(xi.x * float(size)), size - 1);
  int                 env_idx;
  Envmap_sample_value val;
  float               xi_y = xi.y;

  int  px       = idx % env_size.x;
  int  py       = idx / env_size.x;
  vec3 env_node = texelFetch(env_accel_tex, ivec2(px, py), 0).rgb;

  if(xi_y < env_node.y)
  {
    env_idx = idx;
    xi_y /= env_node.y;
    val.pdf = env_node.z;
  }
  else
  {
    env_idx = floatBitsToInt(env_node.x);
    xi_y    = (xi_y - env_node.y) / (1.0f - env_node.y);

    px      = env_idx % env_size.x;
    py      = env_idx / env_size.x;
    val.pdf = texelFetch(env_accel_tex, ivec2(px, py), 0).b;
  }

  // Uniformly sample spherical area of pixel.
  float u       = (float(px) + xi_y) / float(env_size.x);
  float phi     = u * (2.0f * PI) - PI;
  float sin_phi = sin(phi);
  float cos_phi = cos(phi);

  float step_theta = PI / float(env_size.y);
  float theta0     = float(py) * step_theta;
  float cos_theta  = cos(theta0) * (1.0f - xi.z) + cos(theta0 + step_theta) * xi.z;
  float theta      = acos(cos_theta);
  float sin_theta  = sin(theta);


  val.dir = vec3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);
  float v = theta * (1.0f / PI);

  val.value = texture(env_tex, vec2(u, v)).rgb / val.pdf;
  return val;
}

void main()
{

  vec3 normal = normalize(world_position);
  vec3 tangent = normalize(abs(normal.x) > abs(normal.z) ? vec3(-normal.y, normal.x, 0.0) : vec3(0.0, -normal.z, normal.y));
  vec3 bitangent = cross(normal, tangent);

  uint state = uint(rnd_init(ivec3(normal * 1e4f)));

  vec3 result = vec3(0.0f);

  uint  nsamples     = 512u;
  float inv_nsamples = 1.0f / float(nsamples);
  for(uint i = 0u; i < nsamples; ++i)
  {
    // Importance sample diffuse BRDF.
    {
      float xi0 = (float(i) + 0.5f) * inv_nsamples;
      state     = rnd_next(state);
      float xi1 = rnd_val(state);

      float phi     = 2.0f * PI * xi0;
      float sin_phi = sin(phi);
      float cos_phi = cos(phi);

      float sin_theta = sqrt(1.0f - xi1);
      float cos_theta = sqrt(xi1);

      vec3 d = vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

      vec3 direction = d.x * tangent + d.y * bitangent + d.z * normal;
      vec2 uv        = get_spherical_uv(direction);

      float p_brdf_sqr = cos_theta * cos_theta * (ONE_OVER_PI * ONE_OVER_PI);
      float p_env      = texture(env_accel_tex, uv).b;
      float w          = p_brdf_sqr / (p_brdf_sqr + p_env * p_env);
      result += texture(env_tex, uv).rgb * w * PI;
    }

    // Importance sample environment.
    {
      state     = rnd_next(state);
      float xi2 = rnd_val(state);
      state     = rnd_next(state);
      float xi3 = rnd_val(state);
      state     = rnd_next(state);
      float xi4 = rnd_val(state);

      Envmap_sample_value val        = envmap_sample(vec3(xi2, xi3, xi4));
      float               cosine     = dot(val.dir, normal);
      float               p_brdf_sqr = cosine * cosine * (ONE_OVER_PI * ONE_OVER_PI);
      float               p_env_sqr  = val.pdf * val.pdf;
      if(cosine > 0.0f)
      {
        float w = p_env_sqr / (p_env_sqr + p_brdf_sqr);
        result += w * val.value * cosine;
      }
    }
  }

  FragColor = vec4(result * (1.0f / float(nsamples)), 1.0f);
}

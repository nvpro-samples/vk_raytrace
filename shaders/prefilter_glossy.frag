/******************************************************************************
 * Copyright 2018 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/
#version 450

//-------------------------------------------------------------------------------------------------
// This shader computes a glossy IBL map to be used with the Unreal 4 PBR shading model as
// described in
//
// "Real Shading in Unreal Engine 4" by Brian Karis
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
//
// As an extension to the original it uses multiple importance sampling weighted BRDF importance
// sampling and environment map importance sampling to yield good results for high dynamic range
// lighting.
//-------------------------------------------------------------------------------------------------

// varying inputs from passthrough.vert
layout(location = 0) in vec3 world_position;
layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform sampler2D env_tex;
layout(binding = 1) uniform sampler2D env_accel_tex;

layout(push_constant) uniform PushConsts
{
  layout(offset = 0) mat4   mvp;
  layout(offset = 64) float roughness;
  layout(offset = 68) uint  numSamples;
}
consts;


const float PI          = 3.14159265359;
const float ONE_OVER_PI = 0.3183099;

vec2 get_spherical_uv(vec3 v)
{
  float gamma = asin(v.y);
  float theta = atan(v.z, v.x);

  return vec2(theta * ONE_OVER_PI * 0.5, gamma * ONE_OVER_PI) + 0.5;
}

// Importance sample a GGX microfacet distribution.
vec3 ggx_sample(vec2 xi, float alpha)
{
  float phi       = 2.0 * PI * xi.x;
  float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
  float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

  return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Evaluate a GGX microfacet distribution.
float ggx_eval(float alpha, float nh)
{
  float a2   = alpha * alpha;
  float nh2  = nh * nh;
  float tan2 = (1.0f - nh2) / nh2;
  float f    = a2 + tan2;
  return a2 / (f * f * PI * nh2 * nh);
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

  return float(state) * 2.3283064365386963e-10;
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

  float alpha    = consts.roughness * consts.roughness;
  uint  nsamples = alpha > 0.0f ? 512u : 1u;

  uint state = uint(rnd_init(ivec3(normal * 1e4f)));

  // The integrals are additionally weighted by the cosine and normalized using the average cosine of
  // the importance sampled BRDF directions (as in the Unreal publication).
  float weight_sum = 0.0f;

  vec3  result       = vec3(0.0);
  float inv_nsamples = 1.0f / float(nsamples);
  for(uint i = 0u; i < nsamples; ++i)
  {
    // Importance sample BRDF.
    {
      float xi0 = (float(i) + 0.5f) * inv_nsamples;
      state     = rnd_next(state);
      float xi1 = rnd_val(state);

      vec3 h0 = alpha > 0.0f ? ggx_sample(vec2(xi0, xi1), alpha) : vec3(0.0f, 0.0f, 1.0f);
      vec3 h  = tangent * h0.x + bitangent * h0.y + normal * h0.z;

      vec3 direction = normalize(2.0 * dot(normal, h) * h - normal);

      float cos_theta = dot(normal, direction);
      if(cos_theta > 0.0)
      {
        vec2  uv = get_spherical_uv(direction);
        float w  = 1.0f;
        if(alpha > 0.0f)
        {
          float pdf_brdf_sqr = ggx_eval(alpha, h0.z) * 0.25f / dot(direction, h);
          pdf_brdf_sqr *= pdf_brdf_sqr;
          float pdf_env = texture(env_accel_tex, uv).b;
          w             = pdf_brdf_sqr / (pdf_brdf_sqr + pdf_env * pdf_env);
        }
        result += w * texture(env_tex, uv).rgb * cos_theta;
        weight_sum += cos_theta;
      }
    }

    // Importance sample environment.
    if(alpha > 0.0f)
    {
      state     = rnd_next(state);
      float xi2 = rnd_val(state);
      state     = rnd_next(state);
      float xi3 = rnd_val(state);
      state     = rnd_next(state);
      float xi4 = rnd_val(state);

      Envmap_sample_value val = envmap_sample(vec3(xi2, xi3, xi4));

      vec3  h  = normalize(normal + val.dir);
      float nh = dot(h, normal);
      float kh = dot(val.dir, h);
      float nk = dot(val.dir, normal);
      if(kh > 0.0f && nh > 0.0f && nk > 0.0f)
      {
        float pdf_env_sqr = val.pdf * val.pdf;
        float pdf_brdf    = ggx_eval(alpha, nh) * 0.25f / kh;

        float w = pdf_env_sqr / (pdf_env_sqr + pdf_brdf * pdf_brdf);
        result += w * val.value * pdf_brdf * nk * nk;
      }
    }
  }
  FragColor = vec4(result / float(weight_sum), 1.0);
}

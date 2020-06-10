#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

//-------------------------------------------------------------------------------------------------
// Default miss shader for the raytracer
// - Return the HDR
// - If roughness is greather than 0, use the preflitered map
//-------------------------------------------------------------------------------------------------


#include "binding.h"
#include "sampling.glsl"
#include "share.h"

layout(location = 0) rayPayloadInNV RadianceHitInfo payload;

layout(set = 1, binding = B_HDR) uniform sampler2D samplerEnv;
layout(set = 1, binding = B_FILTER_GLOSSY) uniform samplerCube prefilteredMap;


void main()
{

  vec3 radiance = vec3(0);

  // Not adding the contribution of the environment, this is done in CHIT
  if(has_flag(payload.flags, FLAG_FIRST_PATH_SEGMENT))
  {
    vec2 uv  = get_spherical_uv(gl_WorldRayDirectionNV);  // See sampling.glsl
    radiance = texture(samplerEnv, uv).rgb;
  }

  payload.contribution = radiance * payload.weight;
  payload.flags        = FLAG_DONE;
}

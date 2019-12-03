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

layout(location = 0) rayPayloadInNV PerRayData_raytrace prd;

layout(set = 1, binding = B_HDR) uniform sampler2D samplerEnv;
layout(set = 1, binding = B_FILTER_GLOSSY) uniform samplerCube prefilteredMap;


// Return a glossy reflection
vec3 prefilteredReflection(vec3 R, float roughness)
{
  int   levels = textureQueryLevels(prefilteredMap);
  float lod    = clamp(roughness * float(levels), 0.0, float(levels));
  return textureLod(prefilteredMap, R, lod).rgb;
}

void main()
{

  vec3 hitValue = vec3(0);
  if(prd.roughness > 0)
  {
    hitValue = prefilteredReflection(gl_WorldRayDirectionNV, prd.roughness);
  }
  else
  {
    vec2 uv  = get_spherical_uv(gl_WorldRayDirectionNV);  // See sampling.glsl
    hitValue = texture(samplerEnv, uv).rgb;
  }
  prd.result += hitValue * prd.importance;
  prd.depth = 1000;  // Will stop rendering
}

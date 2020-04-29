#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
//#extension GL_EXT_nonuniform_qualifier : enable

#include "share.h"

// Payload information of the ray returning: 0 hit, 2 shadow
layout(location = 0) rayPayloadInNV PerRayData_pick prd;

// Raytracing hit attributes: barycentrics
hitAttributeNV vec2 attribs;

void main()
{
  prd.worldPos =  vec4(gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV, 0);
  prd.barycentrics = vec4(1.0 - attribs.x - attribs.y, attribs.x, attribs.y, 0);
  prd.instanceID = gl_InstanceID;
  prd.primitiveID = gl_PrimitiveID;


}

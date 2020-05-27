#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "share.h"

//-------------------------------------------------------------------------------------------------
// This will be executed when sending shadow rays and missing all geometries
// - There are no hit shader for the shadow ray, therefore
// - Before calling Trace, set isShadowed=true
// - The default anyhit, closesthit won't change isShadowed, but if nothing is hit, it will be
//   set to false.
//-------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------
// Miss shader for the shadow rayPayloadInNV
//

layout(location = 1) rayPayloadInNV ShadowHitInfo payload;

void main()
{
  payload.isHit = false;
}

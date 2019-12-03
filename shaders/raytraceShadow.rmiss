#version 460
#extension GL_NV_ray_tracing : require

layout(location = 2) rayPayloadInNV bool isShadowed;

//-------------------------------------------------------------------------------------------------
// This will be executed when sending shadow rays and missing all geometries
// - There are no hit shader for the shadow ray, therefore
// - Before calling Trace, set isShadowed=true
// - The default anyhit, closesthit won't change isShadowed, but if nothing is hit, it will be
//   set to false.
//-------------------------------------------------------------------------------------------------

void main()
{
  isShadowed = false;
}

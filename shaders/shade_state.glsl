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

#ifndef SHADE_STATE_GLSL
#define SHADE_STATE_GLSL


#include "layouts.glsl"

// Shading information used by the material
struct ShadeState
{
  vec3 normal;
  vec3 geom_normal;
  vec3 position;
  vec2 text_coords[1];
  vec3 tangent_u[1];
  vec3 tangent_v[1];
  vec3 color;
  uint matIndex;
};


vec3 getVertex(uint index)
{
  uint i = 3 * index;
  return vec3(vertices[nonuniformEXT(i + 0)], vertices[nonuniformEXT(i + 1)], vertices[nonuniformEXT(i + 2)]);
}

vec3 getNormal(uint index)
{
  uint i = 3 * index;
  return vec3(normals[nonuniformEXT(i + 0)], normals[nonuniformEXT(i + 1)], normals[nonuniformEXT(i + 2)]);
}

vec2 getTexCoord(uint index)
{
  uint i = 2 * index;
  return vec2(texcoord0[nonuniformEXT(i + 0)], texcoord0[nonuniformEXT(i + 1)]);
}

vec4 getTangent(uint index)
{
  uint i = 4 * index;
  return vec4(tangents[nonuniformEXT(i + 0)], tangents[nonuniformEXT(i + 1)], tangents[nonuniformEXT(i + 2)],
              tangents[nonuniformEXT(i + 3)]);
}

vec4 getColor(uint index)
{
  uint i = 4 * index;
  return vec4(colors[nonuniformEXT(i + 0)], colors[nonuniformEXT(i + 1)], colors[nonuniformEXT(i + 2)],
              colors[nonuniformEXT(i + 3)]);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
ShadeState GetShadeState(in PtPayload hstate)
{
  ShadeState sstate;

  // Retrieve the Primitive mesh buffer information
  RtPrimitiveLookup pinfo = primInfo[hstate.instanceCustomIndex];

  // Getting the 'first index' for this mesh (offset of the mesh + offset of the triangle)
  const uint indexOffset  = pinfo.indexOffset + (3 * hstate.primitiveID);
  const uint vertexOffset = pinfo.vertexOffset;           // Vertex offset as defined in glTF
  const uint matIndex     = max(0, pinfo.materialIndex);  // material of primitive mesh

  // Getting the 3 indices of the triangle (local)
  ivec3 triangleIndex = ivec3(indices[nonuniformEXT(indexOffset + 0)],  //
                              indices[nonuniformEXT(indexOffset + 1)],  //
                              indices[nonuniformEXT(indexOffset + 2)]);
  triangleIndex += ivec3(vertexOffset);  // (global)

  const vec3 barycentrics = vec3(1.0 - hstate.baryCoord.x - hstate.baryCoord.y, hstate.baryCoord.x, hstate.baryCoord.y);

  // Vertex of the triangle
  const vec3 pos0           = getVertex(triangleIndex.x);
  const vec3 pos1           = getVertex(triangleIndex.y);
  const vec3 pos2           = getVertex(triangleIndex.z);
  const vec3 position       = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  const vec3 world_position = vec3(hstate.objectToWorld * vec4(position, 1.0));

  // Normal
  const vec3 nrm0         = getNormal(triangleIndex.x);
  const vec3 nrm1         = getNormal(triangleIndex.y);
  const vec3 nrm2         = getNormal(triangleIndex.z);
  const vec3 normal       = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 world_normal = normalize(vec3(normal * hstate.worldToObject));
  const vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));
  const vec3 wgeom_normal = normalize(vec3(geom_normal * hstate.worldToObject));


  // Tangent and Binormal
  const vec4 tng0     = getTangent(triangleIndex.x);
  const vec4 tng1     = getTangent(triangleIndex.y);
  const vec4 tng2     = getTangent(triangleIndex.z);
  vec3       tangent  = (tng0.xyz * barycentrics.x + tng1.xyz * barycentrics.y + tng2.xyz * barycentrics.z);
  tangent.xyz         = normalize(tangent.xyz);
  vec3 world_tangent  = normalize(vec3(mat4(hstate.objectToWorld) * vec4(tangent.xyz, 0)));
  world_tangent       = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
  vec3 world_binormal = cross(world_normal, world_tangent) * tng0.w;

  // TexCoord
  const vec2 uv0       = getTexCoord(triangleIndex.x);
  const vec2 uv1       = getTexCoord(triangleIndex.y);
  const vec2 uv2       = getTexCoord(triangleIndex.z);
  const vec2 texcoord0 = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  // Colors
  const vec4 col0  = getColor(triangleIndex.x);
  const vec4 col1  = getColor(triangleIndex.y);
  const vec4 col2  = getColor(triangleIndex.z);
  const vec4 color = col0 * barycentrics.x + col1 * barycentrics.y + col2 * barycentrics.z;

  sstate.normal         = world_normal;
  sstate.geom_normal    = wgeom_normal;
  sstate.position       = world_position;
  sstate.text_coords[0] = texcoord0;
  sstate.tangent_u[0]   = world_tangent;
  sstate.tangent_v[0]   = world_binormal;
  sstate.color          = color.rgb;
  sstate.matIndex       = matIndex;

  // Move normal to same side as geometric normal
  if(dot(sstate.normal, sstate.geom_normal) <= 0)
  {
    sstate.normal *= -1.0f;
  }

  return sstate;
}

#endif  // SHADE_STATE_GLSL

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef SHADE_STATE_GLSL
#define SHADE_STATE_GLSL


#include "compress.glsl"
#include "layouts.glsl"

//-----------------------------------------------------------------------
// Return the tangent and binormal from the incoming normal
//-----------------------------------------------------------------------
void CreateTangent(in vec3 N, out vec3 Nt, out vec3 Nb)
{
  Nt = normalize(((abs(N.z) > 0.99999f) ? vec3(-N.x * N.y, 1.0f - N.y * N.y, -N.y * N.z) :
                                          vec3(-N.x * N.z, -N.y * N.z, 1.0f - N.z * N.z)));
  Nb = cross(Nt, N);
}


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

/// Resetting the LSB of the V component (used by tangent handiness)
vec2 decode_texture(vec2 t)
{
  return vec2(t.x, uintBitsToFloat(floatBitsToUint(t.y) & ~1));
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
ShadeState GetShadeState(in PtPayload hstate)
{
  ShadeState sstate;

  const uint idGeo  = hstate.instanceCustomIndex;  // Geometry of this instance
  const uint idPrim = hstate.primitiveID;          // Triangle ID
  const vec3 bary   = vec3(1.0 - hstate.baryCoord.x - hstate.baryCoord.y, hstate.baryCoord.x, hstate.baryCoord.y);

  // Indices of this triangle primitive.
  uvec3 tri = indices[nonuniformEXT(idGeo)].i[idPrim];

  // All vertex attributes of the triangle.
  VertexAttributes attr0 = vertex[nonuniformEXT(idGeo)].v[tri.x];
  VertexAttributes attr1 = vertex[nonuniformEXT(idGeo)].v[tri.y];
  VertexAttributes attr2 = vertex[nonuniformEXT(idGeo)].v[tri.z];

  // Getting the material index on this geometry
  const uint matIndex = max(0, geoInfo[idGeo].materialIndex);  // material of primitive mesh

  // Vertex of the triangle
  const vec3 pos0           = attr0.position.xyz;
  const vec3 pos1           = attr1.position.xyz;
  const vec3 pos2           = attr2.position.xyz;
  const vec3 position       = pos0 * bary.x + pos1 * bary.y + pos2 * bary.z;
  const vec3 world_position = vec3(hstate.objectToWorld * vec4(position, 1.0));

  // Normal
  vec3 nrm0         = decompress_unit_vec(attr0.normal);
  vec3 nrm1         = decompress_unit_vec(attr1.normal);
  vec3 nrm2         = decompress_unit_vec(attr2.normal);
  vec3 normal       = normalize(nrm0 * bary.x + nrm1 * bary.y + nrm2 * bary.z);
  vec3 world_normal = normalize(vec3(normal * hstate.worldToObject));
  vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3 wgeom_normal = normalize(vec3(geom_normal * hstate.worldToObject));

  // Tangent and Binormal
  float h0 = (floatBitsToInt(attr0.texcoord.y) & 1) == 1 ? 1.0f : -1.0f;  // Handiness stored in the less
  float h1 = (floatBitsToInt(attr1.texcoord.y) & 1) == 1 ? 1.0f : -1.0f;  // significative bit of the
  float h2 = (floatBitsToInt(attr2.texcoord.y) & 1) == 1 ? 1.0f : -1.0f;  // texture coord V

  const vec4 tng0     = vec4(decompress_unit_vec(attr0.tangent.x), h0);
  const vec4 tng1     = vec4(decompress_unit_vec(attr1.tangent.x), h1);
  const vec4 tng2     = vec4(decompress_unit_vec(attr2.tangent.x), h2);
  vec3       tangent  = (tng0.xyz * bary.x + tng1.xyz * bary.y + tng2.xyz * bary.z);
  tangent.xyz         = normalize(tangent.xyz);
  vec3 world_tangent  = normalize(vec3(mat4(hstate.objectToWorld) * vec4(tangent.xyz, 0)));
  world_tangent       = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
  vec3 world_binormal = cross(world_normal, world_tangent) * tng0.w;

  // TexCoord

  const vec2 uv0       = decode_texture(attr0.texcoord);
  const vec2 uv1       = decode_texture(attr1.texcoord);
  const vec2 uv2       = decode_texture(attr2.texcoord);
  const vec2 texcoord0 = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

  // Colors
  const vec4 col0  = unpackUnorm4x8(attr0.color);  // RGBA in uint to 4 x float
  const vec4 col1  = unpackUnorm4x8(attr1.color);
  const vec4 col2  = unpackUnorm4x8(attr2.color);
  const vec4 color = col0 * bary.x + col1 * bary.y + col2 * bary.z;

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

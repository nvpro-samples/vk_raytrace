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


#include "shade_state.glsl"

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true is opaque
//----------------------------------------------------------
bool HitTest(in rayQueryEXT rayQuery, in Ray r)
{
  int InstanceCustomIndexEXT = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
  int PrimitiveID            = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);

  // Retrieve the Primitive mesh buffer information
  InstanceData      pinfo    = geoInfo[InstanceCustomIndexEXT];
  const uint        matIndex = max(0, pinfo.materialIndex);  // material of primitive mesh
  GltfShadeMaterial mat      = materials[nonuniformEXT(matIndex)];

  //// Back face culling defined by material
  //bool front_face = rayQueryGetIntersectionFrontFaceEXT(rayQuery, false);
  //if(mat.doubleSided == 0 && front_face == false)
  //{
  //  return false;
  //}

  //// Early out if there is no opacity function
  //if(mat.alphaMode == ALPHA_OPAQUE)
  //{
  //  return true;
  //}

  float baseColorAlpha = mat.pbrBaseColorFactor.a;
  if(mat.pbrBaseColorTexture > -1)
  {
    const uint idGeo  = InstanceCustomIndexEXT;  // Geometry of this instance
    const uint idPrim = PrimitiveID;             // Triangle ID

    // Primitive buffer addresses
    Indices  indices  = Indices(geoInfo[idGeo].indexAddress);
    Vertices vertices = Vertices(geoInfo[idGeo].vertexAddress);

    // Indices of this triangle primitive.
    uvec3 tri = indices.i[idPrim];

    // All vertex attributes of the triangle.
    VertexAttributes attr0 = vertices.v[tri.x];
    VertexAttributes attr1 = vertices.v[tri.y];
    VertexAttributes attr2 = vertices.v[tri.z];

    // Get the texture coordinate
    vec2       bary         = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
    const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    const vec2 uv0          = attr0.texcoord;
    const vec2 uv1          = attr1.texcoord;
    const vec2 uv2          = attr2.texcoord;
    vec2       texcoord0    = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    // Uv Transform
    texcoord0 = (vec4(texcoord0.xy, 1, 1) * mat.uvTransform).xy;

    baseColorAlpha *= texture(texturesMap[nonuniformEXT(mat.pbrBaseColorTexture)], texcoord0).a;
  }

  float opacity;
  if(mat.alphaMode == ALPHA_MASK)
  {
    opacity = baseColorAlpha > mat.alphaCutoff ? 1.0 : 0.0;
  }
  else
  {
    opacity = baseColorAlpha;
  }

  // do alpha blending the stochastically way
  if(rnd(prd.seed) > opacity)
    return false;

  return true;
}

//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void ClosestHit(Ray r)
{
  uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT;  // gl_RayFlagsNoneEXT
  prd.hitT      = INFINITY;

  // Initializes a ray query object but does not start traversal
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery,     //
                        topLevelAS,   // acceleration structure
                        rayFlags,     // rayFlags
                        0xFF,         // cullMask
                        r.origin,     // ray origin
                        0.0,          // ray min range
                        r.direction,  // ray direction
                        INFINITY);    // ray max range

  // Start traversal: return false if traversal is complete
  while(rayQueryProceedEXT(rayQuery))
  {
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
    {
      if(HitTest(rayQuery, r))
      {
        rayQueryConfirmIntersectionEXT(rayQuery);  // The hit was opaque
      }
    }
  }

  bool hit = (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
  if(hit)
  {
    prd.hitT                = rayQueryGetIntersectionTEXT(rayQuery, true);
    prd.primitiveID         = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    prd.instanceID          = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    prd.instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    prd.baryCoord           = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    prd.objectToWorld       = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    prd.worldToObject       = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
  }
}


//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool AnyHit(Ray r, float maxDist)
{
  shadow_payload.isHit = true;      // Asume hit, will be set to false if hit nothing (miss shader)
  shadow_payload.seed  = prd.seed;  // don't care for the update - but won't affect the rahit shader
  uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;

  // Initializes a ray query object but does not start traversal
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery,     //
                        topLevelAS,   // acceleration structure
                        rayFlags,     // rayFlags
                        0xFF,         // cullMask
                        r.origin,     // ray origin
                        0.0,          // ray min range
                        r.direction,  // ray direction
                        maxDist);     // ray max range

  // Start traversal: return false if traversal is complete
  while(rayQueryProceedEXT(rayQuery))
  {
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
    {
      if(HitTest(rayQuery, r))
      {
        rayQueryConfirmIntersectionEXT(rayQuery);  // The hit was opaque
      }
    }
  }


  // add to ray contribution from next event estimation
  return (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
}

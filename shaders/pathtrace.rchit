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

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require          // This is about ray tracing
#extension GL_EXT_nonuniform_qualifier : enable  // To access unsized descriptor arrays
// Align structure layout to scalar
#extension GL_EXT_scalar_block_layout : enable

// C++ shared structure: RtxState
#include "../structures.h"
// Payload and other structures
#include "globals.glsl"


// Payloads
layout(location = 0) rayPayloadInEXT HitPayload prd;
layout(location = 1) rayPayloadEXT ShadowHitPayload shadow_payload;

// Push Constant
layout(push_constant) uniform _RtxState
{
  RtxState rtstate;
};


#include "brdf.glsl"
#include "raycommon.glsl"
#include "sampling.glsl"
#include "shade_state.glsl"


//-----------------------------------------------------------------------
// Retrieving the ray hit state (different when using Ray Query)
//
hitAttributeEXT vec2 bary;

HitState GetState()
{
  HitState state;
  state.InstanceID          = gl_InstanceID;
  state.PrimitiveID         = gl_PrimitiveID;
  state.InstanceCustomIndex = gl_InstanceCustomIndexEXT;
  state.ObjectToWorld       = gl_ObjectToWorldEXT;
  state.WorldToObject       = gl_WorldToObjectEXT;
  state.WorldRayOrigin      = gl_WorldRayOriginEXT;
  state.bary                = bary;
  return state;
}


//-----------------------------------------------------------------------
// Sampling the HDR environment or scene lights
//
vec4 EnvSample(inout vec3 radiance)
{
  vec3  lightDir;
  float pdf;

  // #TODO adding scene lights (point, spot, infinit)

  // Sun & Sky or HDR
  if(_sunAndSky.in_use == 1)
  {
    // #TODO: find proper light direction + PDF
    lightDir = _sunAndSky.sun_direction;
    radiance = sun_and_sky(_sunAndSky, -lightDir);
    pdf      = 0.5;
  }
  else
  {
    vec3 randVal = vec3(rnd(prd.seed), rnd(prd.seed), rnd(prd.seed));

    // Sampling the HDR with importance sampling
    radiance = Environment_sample(  //
        environmentTexture,         // assuming lat long map
        environmentSamplingData,    // importance sampling data of the environment map
        randVal, lightDir, pdf);
  }

  radiance *= rtstate.environment_intensity_factor;
  return vec4(lightDir, pdf);
}


//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool AnyHit(Ray ray, float maxDist)
{
  // cast a shadow ray; assuming light is always outside
  vec3 origin    = ray.origin;
  vec3 direction = ray.direction;

  // prepare the ray and payload but trace at the end to reduce the amount of data that has
  // to be recovered after coming back from the shadow trace
  shadow_payload.isHit = true;
  shadow_payload.seed  = 0;
  uint rayFlags        = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;

  traceRayEXT(topLevelAS,  // acceleration structure
              rayFlags,    // rayFlags
              0xFF,        // cullMask
              0,           // sbtRecordOffset
              0,           // sbtRecordStride
              1,           // missIndex
              origin,      // ray origin
              0.0,         // ray min range
              direction,   // ray direction
              maxDist,     // ray max range
              1            // payload layout(location = 1)
  );

  // add to ray contribution from next event estimation
  return shadow_payload.isHit;
}


//-----------------------------------------------------------------------
// Evaluate the contribution at the hit point
//
vec3 EvalPoint(in Ray r, in Material bsdfMat, in ShadeState sstate, in float anisotropy)
{
  vec3 L = vec3(0.0);

  // Environment Lighting
  // Find the light contribution from the environment, the direction and the PDF
  vec3  lightColor;
  vec4  dirPdf   = EnvSample(lightColor);
  vec3  lightDir = dirPdf.xyz;
  float lightPdf = dirPdf.w;

  if(dot(lightDir, sstate.geom_normal) > 0)
  {
    // Shoot a ray toward the environment and check if it is not
    // occluded by an object
    vec3 surfacePos = OffsetRay(sstate.position, sstate.geom_normal);
    Ray  shadowRay  = Ray(surfacePos, lightDir);
    bool inShadow   = AnyHit(shadowRay, 1e32);

    // Compute the light contribution on the surface of the object
    if(!inShadow)
    {
      float bsdfPdf = BsdfPdf(r, sstate.normal, sstate.tangent_u[0], sstate.tangent_v[0], bsdfMat, lightDir);
      vec3  f       = BsdfEval(r, sstate.normal, sstate.tangent_u[0], sstate.tangent_v[0], bsdfMat, lightDir);

      if(lightPdf > 0.0 && bsdfPdf > 0.0f)
      {
        float misWeight = powerHeuristic(lightPdf, bsdfPdf);
        L += misWeight * f * abs(dot(lightDir, sstate.normal)) * lightColor / lightPdf;
      }
    }
  }
  return L;
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


// Retrieve the diffuse and specular color base on the shading model: Metal-Roughness or Specular-Glossiness
Material GetMetallicRoughness(GltfShadeMaterial material, ShadeState sstate)
{
  Material bsdfMat;
  float    perceptualRoughness = 0.0;
  float    metallic            = 0.0;
  vec4     baseColor           = vec4(0.0, 0.0, 0.0, 1.0);
  vec3     f0                  = vec3(0.04);

  // Metallic and Roughness material properties are packed together
  // In glTF, these factors can be specified by fixed scalar values
  // or from a metallic-roughness map
  perceptualRoughness = material.pbrRoughnessFactor;
  metallic            = material.pbrMetallicFactor;
  if(material.pbrMetallicRoughnessTexture > -1)
  {
    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
    vec4 mrSample = texture(texturesMap[nonuniformEXT(material.pbrMetallicRoughnessTexture)], sstate.text_coords[0]);
    perceptualRoughness = mrSample.g * perceptualRoughness;
    metallic            = mrSample.b * metallic;
  }

  // The albedo may be defined from a base texture or a flat color
  baseColor = material.pbrBaseColorFactor;
  if(material.pbrBaseColorTexture > -1)
  {
    baseColor *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], sstate.text_coords[0]));
  }

  // baseColor.rgb = mix(baseColor.rgb * (vec3(1.0) - f0), vec3(0), metallic);
  // Specular color (ior 1.4)
  f0 = mix(vec3(0.04), baseColor.xyz, metallic);

  bsdfMat.albedo    = baseColor;
  bsdfMat.metallic  = metallic;
  bsdfMat.roughness = perceptualRoughness;
  bsdfMat.f0        = f0;

  return bsdfMat;
}


// Specular-Glossiness which will be converted to metallic-roughness
Material GetSpecularGlossiness(GltfShadeMaterial material, ShadeState sstate)
{
  Material bsdfMat;
  float    perceptualRoughness = 0.0;
  float    metallic            = 0.0;
  vec4     baseColor           = vec4(0.0, 0.0, 0.0, 1.0);
  vec3     f0                  = vec3(0.04);

  f0                  = material.khrSpecularFactor;
  perceptualRoughness = 1.0 - material.khrGlossinessFactor;

  if(material.khrSpecularGlossinessTexture > -1)
  {
    vec4 sgSample =
        SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.khrSpecularGlossinessTexture)], sstate.text_coords[0]));
    perceptualRoughness = 1 - material.khrGlossinessFactor * sgSample.a;  // glossiness to roughness
    f0 *= sgSample.rgb;                                                   // specular
  }

  vec3  specularColor            = f0;  // f0 = specular
  float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);

  vec4 diffuseColor = material.khrDiffuseFactor;
  if(material.khrDiffuseTexture > -1)
    diffuseColor *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.khrDiffuseTexture)], sstate.text_coords[0]));

  baseColor.rgb = diffuseColor.rgb * oneMinusSpecularStrength;
  metallic      = solveMetallic(diffuseColor.rgb, specularColor, oneMinusSpecularStrength);

  bsdfMat.albedo    = baseColor;
  bsdfMat.albedo.a  = diffuseColor.a;
  bsdfMat.metallic  = metallic;
  bsdfMat.roughness = perceptualRoughness;
  bsdfMat.f0        = f0;

  return bsdfMat;
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void main()
{
  HitState          hstate   = GetState();
  ShadeState        sstate   = GetShadeState(hstate);
  GltfShadeMaterial material = materials[nonuniformEXT(sstate.matIndex)];

  // Uv Transform
  sstate.text_coords[0] = (vec4(sstate.text_coords[0].xy, 1, 1) * material.uvTransform).xy;


  // KHR_materials_anisotropy .. rotates the tangents
  if(material.anisotropy > 0)
  {
    mat3 TBN            = mat3(sstate.tangent_u[0], sstate.tangent_v[0], sstate.normal);
    sstate.tangent_u[0] = normalize(TBN * material.anisotropyDirection);
    sstate.tangent_v[0] = normalize(cross(sstate.normal, sstate.tangent_u[0]));
  }


  // Perturbating the normal if a normal map is present
  if(material.normalTexture > -1)
  {
    mat3 TBN          = mat3(sstate.tangent_u[0], sstate.tangent_v[0], sstate.normal);
    vec3 normalVector = texture(texturesMap[nonuniformEXT(material.normalTexture)], sstate.text_coords[0]).xyz;
    normalVector      = normalize(normalVector * 2.0 - 1.0);
    normalVector *= vec3(material.normalTextureScale, material.normalTextureScale, 1.0);
    sstate.normal = normalize(TBN * normalVector);
  }

  // Emissive term
  vec3 emissive = material.emissiveFactor;
  if(material.emissiveTexture > -1)
    emissive *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.emissiveTexture)], sstate.text_coords[0])).rgb;


  // --- BRDF Setting
  vec3 radiance   = prd.contribution;
  vec3 throughput = prd.weight;

  // Bsdf State initialization
  Ray      R = Ray(gl_WorldRayOriginEXT, gl_WorldRayDirectionEXT);
  Material bsdfMat;
  if(material.shadingModel == MATERIAL_METALLICROUGHNESS)
    bsdfMat = GetMetallicRoughness(material, sstate);
  else
    bsdfMat = GetSpecularGlossiness(material, sstate);

  // Color at vertices
  bsdfMat.albedo.rgb *= sstate.color;

  bsdfMat.anisotropy = material.anisotropy;

  // KHR_materials_unlit
  if(material.unlit == 1)
  {
    prd.contribution += bsdfMat.albedo.rgb * throughput;
    prd.flags = FLAG_DONE;
    return;
  }


  // Adding emissive
  radiance += emissive * throughput;

  // Adding light contribution
  radiance += EvalPoint(R, bsdfMat, sstate, material.anisotropy) * throughput;

  // Sampling and weight for next ray event
  vec3  rndVal  = vec3(rnd(prd.seed), rnd(prd.seed), rnd(prd.seed));
  vec3  bsdfDir = BsdfSample(R, sstate.normal, sstate.tangent_u[0], sstate.tangent_v[0], bsdfMat, rndVal);
  float pdf     = BsdfPdf(R, sstate.normal, sstate.tangent_u[0], sstate.tangent_v[0], bsdfMat, bsdfDir);

  if(pdf > 0.0 && dot(sstate.normal, bsdfDir) > 0.0)
  {
    throughput *= BsdfEval(R, sstate.normal, sstate.tangent_u[0], sstate.tangent_v[0], bsdfMat, bsdfDir)
                  * abs(dot(sstate.normal, bsdfDir)) / pdf;
  }
  else
  {
    prd.flags = FLAG_DONE;
  }

  // Storage -- Next ray
  prd.contribution = radiance;
  prd.weight       = throughput;
  prd.rayOrigin    = OffsetRay(sstate.position, sstate.geom_normal);
  prd.rayDirection = bsdfDir;


  // Debugging info
  if(rtstate.debugging_mode != 0)
  {

    prd.flags  = FLAG_DONE;
    prd.weight = vec3(1.0);
    switch(rtstate.debugging_mode)
    {
      case eMetallic:
        prd.contribution = vec3(bsdfMat.metallic);
        return;
      case eNormal:
        prd.contribution = (sstate.normal + vec3(1)) * .5;
        return;
      case eBaseColor:
        prd.contribution = bsdfMat.albedo.xyz;
        return;
      case eEmissive:
        prd.contribution = emissive;
        return;
      case eAlpha:
        prd.contribution = vec3(bsdfMat.albedo.a);
        return;
      case eRoughness:
        prd.contribution = vec3(bsdfMat.roughness);
        return;
      case eTextcoord:
        prd.contribution = vec3(sstate.text_coords[0], 1);
        return;
      case eTangent:
        prd.contribution = vec3(sstate.tangent_u[0].xyz + vec3(1)) * .5;
        return;
      case eRadiance:
        prd.contribution = radiance;
        return;
      case eWeight:
        prd.contribution = throughput;
        return;
      case eRayDir:
        prd.contribution = (bsdfDir + vec3(1)) * .5;
        return;
    };
  }
}

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
#extension GL_EXT_ray_tracing : require  // This is about ray tracing


#define USE_ACCEL
#define USE_RANDOM
#define USE_SAMPLING
#define USE_SHADING
#define USE_SCENE
#define USE_SUN_AND_SKY
#include "raycommon.h.glsl"

hitAttributeEXT vec2 bary;

// Payloads
layout(location = 0) rayPayloadInEXT HitPayload prd;
layout(location = 1) rayPayloadEXT ShadowHitPayload shadow_payload;

// Push Constant
layout(push_constant) uniform _RtCoreState
{
  RtState rtstate;
};


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


ShadeState GetShadeState(in HitState hstate)
{
  ShadeState sstate;

  // Retrieve the Primitive mesh buffer information
  RtPrimitiveLookup pinfo = primInfo[hstate.InstanceCustomIndex];

  // Getting the 'first index' for this mesh (offset of the mesh + offset of the triangle)
  const uint indexOffset  = pinfo.indexOffset + (3 * hstate.PrimitiveID);
  const uint vertexOffset = pinfo.vertexOffset;           // Vertex offset as defined in glTF
  const uint matIndex     = max(0, pinfo.materialIndex);  // material of primitive mesh

  // Getting the 3 indices of the triangle (local)
  ivec3 triangleIndex = ivec3(indices[nonuniformEXT(indexOffset + 0)],  //
                              indices[nonuniformEXT(indexOffset + 1)],  //
                              indices[nonuniformEXT(indexOffset + 2)]);
  triangleIndex += ivec3(vertexOffset);  // (global)

  const vec3 barycentrics = vec3(1.0 - hstate.bary.x - hstate.bary.y, hstate.bary.x, hstate.bary.y);

  // Vertex of the triangle
  const vec3 pos0           = getVertex(triangleIndex.x);
  const vec3 pos1           = getVertex(triangleIndex.y);
  const vec3 pos2           = getVertex(triangleIndex.z);
  const vec3 position       = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  const vec3 world_position = vec3(hstate.ObjectToWorld * vec4(position, 1.0));

  // Normal
  const vec3 nrm0         = getNormal(triangleIndex.x);
  const vec3 nrm1         = getNormal(triangleIndex.y);
  const vec3 nrm2         = getNormal(triangleIndex.z);
  const vec3 normal       = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 world_normal = normalize(vec3(normal * hstate.WorldToObject));
  const vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));
  const vec3 wgeom_normal = normalize(vec3(geom_normal * hstate.WorldToObject));


  // Tangent and Binormal
  const vec4 tng0     = getTangent(triangleIndex.x);
  const vec4 tng1     = getTangent(triangleIndex.y);
  const vec4 tng2     = getTangent(triangleIndex.z);
  vec4       tangent  = (tng0 * barycentrics.x + tng1 * barycentrics.y + tng2 * barycentrics.z);
  tangent.xyz         = normalize(tangent.xyz);
  vec3 world_tangent  = normalize(vec3(mat4(hstate.ObjectToWorld) * vec4(tangent.xyz, 0)));
  world_tangent       = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
  vec3 world_binormal = cross(world_normal, world_tangent) * tangent.w;

  // TexCoord
  const vec2 uv0       = getTexCoord(triangleIndex.x);
  const vec2 uv1       = getTexCoord(triangleIndex.y);
  const vec2 uv2       = getTexCoord(triangleIndex.z);
  const vec2 texcoord0 = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;


  sstate.normal         = world_normal;
  sstate.geom_normal    = wgeom_normal;
  sstate.position       = world_position;
  sstate.text_coords[0] = texcoord0;
  sstate.tangent_u[0]   = world_tangent;
  sstate.tangent_v[0]   = world_binormal;
  sstate.matIndex       = matIndex;

  // Move normal to same side as geometric normal
  if(dot(sstate.normal, sstate.geom_normal) <= 0)
  {
    sstate.normal *= -1.0f;
  }


  return sstate;
}


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


//-------------------------------------------------------------------------------------------------
// Environment
//-------------------------------------------------------------------------------------------------

struct Environment_sample_data
{
  uint  alias;
  float q;
  float pdf;
};

Environment_sample_data getSampleData(sampler2D sample_buffer, ivec2 idx)
{
  vec3 data = texelFetch(sample_buffer, idx, 0).xyz;

  Environment_sample_data sample_data;
  sample_data.alias = floatBitsToInt(data.x);
  sample_data.q     = data.y;
  sample_data.pdf   = data.z;
  return sample_data;
}

Environment_sample_data getSampleData(sampler2D sample_buffer, uint idx)
{
  uvec2 size = textureSize(sample_buffer, 0);
  uint  px   = idx % size.x;
  uint  py   = idx / size.x;
  return getSampleData(sample_buffer, ivec2(px, size.y - py - 1));  // Image is upside down
}


vec3 environment_sample(sampler2D lat_long_tex, sampler2D sample_buffer, inout uint seed, out vec3 to_light, out float pdf)
{
  vec3 xi;
  xi.x = rnd(seed);
  xi.y = rnd(seed);
  xi.z = rnd(seed);

  uvec2 tsize  = textureSize(lat_long_tex, 0);
  uint  width  = tsize.x;
  uint  height = tsize.y;

  const uint size = width * height;
  const uint idx  = min(uint(xi.x * float(size)), size - 1);

  Environment_sample_data sample_data = getSampleData(sample_buffer, idx);

  uint env_idx;
  if(xi.y < sample_data.q)
  {
    env_idx = idx;
    xi.y /= sample_data.q;
  }
  else
  {
    env_idx = sample_data.alias;
    xi.y    = (xi.y - sample_data.q) / (1.0f - sample_data.q);
  }

  uint       py = env_idx / width;
  const uint px = env_idx % width;
  pdf           = getSampleData(sample_buffer, env_idx).pdf;
  py            = height - py - 1;  // Image is upside down


  // uniformly sample spherical area of pixel
  const float u       = float(px + xi.y) / float(width);
  const float phi     = u * (2.0f * M_PI) - M_PI;
  float       sin_phi = sin(phi);
  float       cos_phi = cos(phi);

  const float step_theta = M_PI / float(height);
  const float theta0     = float(py) * step_theta;
  const float cos_theta  = cos(theta0) * (1.0f - xi.z) + cos(theta0 + step_theta) * xi.z;
  const float theta      = acos(cos_theta);
  const float sin_theta  = sin(theta);
  to_light               = vec3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

  // lookup filtered value
  const float v = theta * M_1_PI;
  return texture(lat_long_tex, vec2(u, v)).xyz;
}


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


const float c_MinReflectance = 0.04;

struct MaterialInfo
{
  float perceptualRoughness;  // roughness value, as authored by the model creator (input to shader)
  vec3  reflectance0;         // full reflectance color (normal incidence angle)
  float alphaRoughness;       // roughness mapped to a more linear change in the roughness (proposed by [2])
  vec3  diffuseColor;         // color contribution from diffuse lighting
  vec3  reflectance90;        // reflectance color at grazing angle
  vec3  specularColor;        // color contribution from specular lighting
};

struct AngularInfo
{
  float NdotL;  // cos angle between normal and light direction
  float NdotV;  // cos angle between normal and view direction
  float NdotH;  // cos angle between normal and half vector
  float LdotH;  // cos angle between light direction and half vector
  float VdotH;  // cos angle between view direction and half vector
  vec3  padding;
};


float getPerceivedBrightness(vec3 vector)
{
  return sqrt(0.299 * vector.r * vector.r + 0.587 * vector.g * vector.g + 0.114 * vector.b * vector.b);
}


// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/js/three.pbrUtilities.js#L34
float solveMetallic(vec3 diffuse, vec3 specular, float oneMinusSpecularStrength)
{
  float specularBrightness = getPerceivedBrightness(specular);

  if(specularBrightness < c_MinReflectance)
  {
    return 0.0;
  }

  float diffuseBrightness = getPerceivedBrightness(diffuse);

  float a = c_MinReflectance;
  float b = diffuseBrightness * oneMinusSpecularStrength / (1.0 - c_MinReflectance) + specularBrightness - 2.0 * c_MinReflectance;
  float c = c_MinReflectance - specularBrightness;
  float D = b * b - 4.0 * a * c;

  return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}


// Lambert lighting
// see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
vec3 diffuse(MaterialInfo materialInfo)
{
  return materialInfo.diffuseColor / M_PI;
}

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 specularReflection(MaterialInfo materialInfo, AngularInfo angularInfo)
{
  return materialInfo.reflectance0
         + (materialInfo.reflectance90 - materialInfo.reflectance0) * pow(clamp(1.0 - angularInfo.VdotH, 0.0, 1.0), 5.0);
}

// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
float visibilityOcclusion(MaterialInfo materialInfo, AngularInfo angularInfo)
{
  float NdotL            = angularInfo.NdotL;
  float NdotV            = angularInfo.NdotV;
  float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;

  float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
  float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

  float GGX = GGXV + GGXL;
  if(GGX > 0.0)
  {
    return 0.5 / GGX;
  }
  return 0.0;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float microfacetDistribution(MaterialInfo materialInfo, AngularInfo angularInfo)
{
  float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;
  float f                = (angularInfo.NdotH * alphaRoughnessSq - angularInfo.NdotH) * angularInfo.NdotH + 1.0;
  return alphaRoughnessSq / (M_PI * f * f);
}

AngularInfo getAngularInfo(vec3 pointToLight, vec3 normal, vec3 view)
{
  // Standard one-letter names
  vec3 n = normalize(normal);        // Outward direction of surface point
  vec3 v = normalize(view);          // Direction from surface point to view
  vec3 l = normalize(pointToLight);  // Direction from surface point to light
  vec3 h = normalize(l + v);         // Direction of the vector between l and v

  float NdotL = clamp(dot(n, l), 0.05, 1.0);
  float NdotV = clamp(dot(n, v), 0.0, 1.0);
  float NdotH = clamp(dot(n, h), 0.0, 1.0);
  float LdotH = clamp(dot(l, h), 0.0, 1.0);
  float VdotH = clamp(dot(v, h), 0.0, 1.0);

  return AngularInfo(NdotL, NdotV, NdotH, LdotH, VdotH, vec3(0, 0, 0));
}

// Return shading value
vec3 getPointShade(vec3 pointToLight, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  AngularInfo angularInfo = getAngularInfo(pointToLight, normal, view);

  // If one of the dot products is larger than zero, no division by zero can happen. Avoids black borders.
  if(angularInfo.NdotL > 0.0 || angularInfo.NdotV > 0.0)
  {
    // Calculate the shading terms for the microfacet specular shading model
    vec3  F   = specularReflection(materialInfo, angularInfo);
    float Vis = visibilityOcclusion(materialInfo, angularInfo);
    float D   = microfacetDistribution(materialInfo, angularInfo);

    // Calculation of analytical lighting contribution
    vec3 diffuseContrib = (1.0 - F) * diffuse(materialInfo);
    vec3 specContrib    = F * Vis * D;

    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
    return angularInfo.NdotL * (diffuseContrib + specContrib);
  }

  return vec3(0.0, 0.0, 0.0);
}


// Utility function to get a vector perpendicular to an input vector
//    (from "Efficient Construction of Perpendicular Vectors Without Branching")
vec3 getPerpendicularStark(vec3 u)
{
  vec3 a  = abs(u);
  uint xm = ((a.x - a.y) < 0 && (a.x - a.z) < 0) ? 1 : 0;
  uint ym = (a.y - a.z) < 0 ? (1 ^ xm) : 0;
  uint zm = 1 ^ (xm | ym);
  return cross(u, vec3(xm, ym, zm));
}

// Get a GGX half vector / microfacet normal, sampled according to the GGX distribution
//    When using this function to sample, the probability density is pdf = D * NdotH / (4 * HdotV)
//
//    \param[in] u Uniformly distributed random numbers between 0 and 1
//    \param[in] N Surface normal
//    \param[in] roughness Roughness^2 of material
//
vec3 getGGXMicrofacet(vec2 u, vec3 N, float roughness)
{
  float a2 = roughness * roughness;

  float phi      = 2 * M_PI * u.x;
  float cosTheta = sqrt(max(0, (1 - u.y)) / (1 + (a2 - 1) * u.y));
  float sinTheta = sqrt(max(0, 1 - cosTheta * cosTheta));

  // Tangent space H
  vec3 tH;
  tH.x = sinTheta * cos(phi);
  tH.y = sinTheta * sin(phi);
  tH.z = cosTheta;

  vec3 T = getPerpendicularStark(N);
  vec3 B = normalize(cross(N, T));

  // World space H
  return normalize(T * tH.x + B * tH.y + N * tH.z);
}


// selects one light source randomly, currently, only IBL is supported
vec3 sample_lights(out vec3 to_light, out float pdf, inout uint seed)
{
  vec3 radiance;  // Result of light emission

  // #TODO adding scene lights (point, spot, infinit)

  // Sun & Sky or HDR
  if(_sunAndSky.in_use == 1)
  {
    // #TODO: find proper light direction + PDF
    to_light = _sunAndSky.sun_direction;
    radiance = sun_and_sky(_sunAndSky, to_light);
    pdf      = 0.5;
  }
  else
  {
    // Sampling the HDR with importance sampling
    radiance = environment_sample(  //
        environmentTexture,         // assuming lat long map
        environmentSamplingData,    // importance sampling data of the environment map
        seed, to_light, pdf);
    radiance *= rtstate.environment_intensity_factor;
  }

  return radiance / pdf;
}


// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec4 SRGBtoLINEAR(vec4 srgbIn, float gamma)
{
  return vec4(pow(srgbIn.xyz, vec3(gamma)), srgbIn.w);
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void main()
{
  HitState   hstate = GetState();
  ShadeState sstate = GetShadeState(hstate);

  GltfMaterial mat = materials[nonuniformEXT(sstate.matIndex)];


  // Metallic and Roughness material properties are packed together
  // In glTF, these factors can be specified by fixed scalar values
  // or from a metallic-roughness map
  float perceptualRoughness = 0.0;
  float metallic            = 0.0;
  vec4  baseColor           = vec4(0.0, 0.0, 0.0, 1.0);
  vec3  diffuseColor        = vec3(0.0);
  vec3  specularColor       = vec3(0.0);
  vec3  f0                  = vec3(0.04);


  // The albedo may be defined from a base texture or a flat color
  baseColor = mat.pbrBaseColorFactor;
  if(mat.pbrBaseColorTexture > -1)
  {
    baseColor *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(mat.pbrBaseColorTexture)], sstate.text_coords[0]), 2.2);
  }

  // Normal Map
  if(mat.normalTexture > -1)
  {
    mat3 TBN          = mat3(sstate.tangent_u[0], sstate.tangent_v[0], sstate.normal);
    vec3 normalVector = texture(texturesMap[nonuniformEXT(mat.normalTexture)], sstate.text_coords[0]).xyz;
    normalVector      = normalize(normalVector * 2.0 - 1.0);
    sstate.normal     = normalize(TBN * normalVector);
  }

  // Retrieve the diffuse and specular color base on the shading model: Metal-Roughness or Specular-Glossiness
  if(mat.shadingModel == MATERIAL_METALLICROUGHNESS)
  {
    perceptualRoughness = mat.pbrRoughnessFactor;
    metallic            = mat.pbrMetallicFactor;

    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
    if(mat.pbrMetallicRoughnessTexture > -1)
    {
      vec4 mrSample = texture(texturesMap[nonuniformEXT(mat.pbrMetallicRoughnessTexture)], sstate.text_coords[0]);
      perceptualRoughness *= mrSample.g;
      metallic *= mrSample.b;
    }

    diffuseColor  = baseColor.rgb * (vec3(1.0) - f0) * (1.0 - metallic);
    specularColor = mix(f0, baseColor.rgb, metallic);
  }
  else if(mat.shadingModel == MATERIAL_SPECULARGLOSSINESS)
  {
    f0                  = mat.khrSpecularFactor * 0.08;
    perceptualRoughness = 1.0 - mat.khrGlossinessFactor;

    specularColor                  = f0;  // f0 = specular
    float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);

    baseColor = mat.khrDiffuseFactor;
    if(mat.khrDiffuseTexture > -1)
      baseColor *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(mat.khrDiffuseTexture)], sstate.text_coords[0]), 2.2);

    diffuseColor = baseColor.rgb * oneMinusSpecularStrength;
    metallic     = solveMetallic(baseColor.rgb, specularColor, oneMinusSpecularStrength);
  }

  // Clamping to valid values
  perceptualRoughness = clamp(perceptualRoughness, 0.0, 1.0);
  metallic            = clamp(metallic, 0.0, 1.0);

  // Roughness is authored as perceptual roughness; as is convention,
  // convert to material roughness by squaring the perceptual roughness [2].
  float alphaRoughness = perceptualRoughness * perceptualRoughness;

  // Compute reflectance.
  float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);

  vec3 specularEnvironmentR0 = specularColor.rgb;
  // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
  vec3 specularEnvironmentR90 = vec3(clamp(reflectance * 50.0, 0.0, 1.0));

  // Retrieving material info for shading
  MaterialInfo materialInfo = MaterialInfo(perceptualRoughness, specularEnvironmentR0, alphaRoughness, diffuseColor,
                                           specularEnvironmentR90, specularColor);


  vec3 contribution = vec3(0.0, 0.0, 0.0);
  vec3 viewDir      = normalize(hstate.WorldRayOrigin - sstate.position);


  // Calculate lighting contribution from image based lighting source (IBL)
  vec3  to_light;
  float pdf;
  vec3  radiance_over_pdf = sample_lights(to_light, pdf, prd.seed);

  if(dot(to_light, sstate.geom_normal) > 0)
  {
    // Shading contribution
    vec3 bsdf_color = getPointShade(to_light, materialInfo, sstate.normal, viewDir);
    contribution += bsdf_color * prd.weight * radiance_over_pdf;
  }

  // Ambient occulsion
  float ao = 1.0;
  if(mat.occlusionTexture > -1)
  {
    ao *= texture(texturesMap[nonuniformEXT(mat.occlusionTexture)], sstate.text_coords[0]).r * mat.occlusionTextureStrength;
    contribution = mix(contribution, contribution * ao, 1.0 /*u_OcclusionStrength*/);
  }

  // Emissive term
  vec3 emissive = mat.emissiveFactor;
  if(mat.emissiveTexture > -1)
    emissive *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(mat.emissiveTexture)], sstate.text_coords[0]), 2.2).rgb;
  prd.contribution += emissive;


  // Debugging info - Early out
  if(rtstate.debugging_mode != 0)
  {
    prd.flags  = FLAG_DONE;
    prd.weight = vec3(1.0);
    switch(rtstate.debugging_mode)
    {
      case eMetallic:
        prd.contribution = vec3(metallic);
        return;
      case eNormal:
        prd.contribution = (sstate.normal + vec3(1)) * .5;
        return;
      case eBaseColor:
        prd.contribution = baseColor.xyz;
        return;
      case eAO:
        prd.contribution = vec3(ao);
        return;
      case eEmissive:
        prd.contribution = emissive;
        return;
      case eAlpha:
        prd.contribution = vec3(baseColor.a);
        return;
      case eRoughness:
        prd.contribution = vec3(perceptualRoughness);
        return;
      case eTextcoord:
        prd.contribution = vec3(sstate.text_coords[0], 1);
        return;
      case eTangent:
        prd.contribution = vec3(sstate.tangent_u[0].xyz + vec3(1)) * .5;
        return;
    };
  }


  // Result Color
  {
    // New Ray  (origin)
    prd.rayOrigin = offsetRay(sstate.position, sstate.geom_normal);

    // New direction and reflectance
    vec3        reflectance = vec3(0);
    vec3        rayDir;
    AngularInfo angularInfo = getAngularInfo(to_light, sstate.normal, viewDir);
    float       F           = specularReflection(materialInfo, angularInfo).x;
    float       e           = rnd(prd.seed);
    if(e > F)
    {
      vec3 T, B, N;
      N = sstate.normal;
      createCoordinateSystem(N, T, B);
      // Randomly sample the hemisphere
      prd.rayDirection = samplingHemisphere(prd.seed, T, B, N);
      prd.weight *= baseColor.rgb;
    }
    else
    {
      vec3 V = viewDir;
      vec3 N = sstate.normal;
      // Randomly sample the NDF to get a microfacet in our BRDF to reflect off
      vec3 H = getGGXMicrofacet(rnd2(prd.seed), N, materialInfo.perceptualRoughness);
      // Compute the outgoing direction based on this (perfectly reflective) microfacet
      prd.rayDirection = normalize(2.f * dot(V, H) * H - V);
      prd.weight *= materialInfo.specularColor;
    }
  }


  // Add contribution from next event estimation if not shadowed
  //---------------------------------------------------------------------------------------------

  // cast a shadow ray; assuming light is always outside
  vec3 origin    = offsetRay(sstate.position, sstate.geom_normal);
  vec3 direction = normalize(to_light);

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
              1e32,        // ray max range
              1            // payload layout(location = 1)
  );

  // add to ray contribution from next event estimation
  if(!shadow_payload.isHit)
    prd.contribution += contribution;
}

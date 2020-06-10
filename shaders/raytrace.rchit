#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

#include "binding.h"
#include "sampling.glsl"
#include "share.h"

const float c_MinReflectance = 0.04;

// Payload information of the ray returning: 0 hit, 2 shadow
layout(location = 0) rayPayloadInNV RadianceHitInfo payload;
layout(location = 1) rayPayloadNV ShadowHitInfo shadow_payload;

layout(push_constant) uniform Constants
{
  int   frame;         // Current frame
  int   maxDepth;      // Trace depth
  float maxRayLenght;  // Trace depth
  int   samples;       // Number of samples per pixel
  float environmentIntensityFactor;
  float fireflyClampThreshold;
}
pushC;


// Raytracing hit attributes: barycentrics
hitAttributeNV vec2 attribs;

// clang-format off
layout(set = 0, binding = 0) uniform accelerationStructureNV topLevelAS;
layout(set = 0, binding = 2) readonly buffer _InstanceInfo {PrimMeshInfo i[];} InstanceInfo;

layout(set = 1, binding = B_SCENE) uniform _ubo {Scene s;} ubo;
layout(set = 1, binding = B_VERTICES) readonly buffer _VertexBuf {float v[];} VertexBuf;
layout(set = 1, binding = B_INDICES) readonly buffer _Indices {uint i[]; } indices;
layout(set = 1, binding = B_NORMALS) readonly buffer _NormalBuf {float v[];} NormalBuf;
layout(set = 1, binding = B_TEXCOORDS) readonly buffer _UvBuf {float v[];} UvBuf;
layout(set = 1, binding = B_TANGENTS) readonly buffer _TangentBuf {float v[];} TangentBuf;
layout(set = 1, binding = B_MATERIAL) readonly buffer _MaterialBuffer {Material m[];} MaterialBuffer;
layout(set = 1, binding = B_HDR) uniform sampler2D environmentTexture;
//layout(set = 1, binding = B_FILTER_DIFFUSE) uniform samplerCube samplerIrradiance;
layout(set = 1, binding = B_LUT_BRDF) uniform sampler2D samplerBRDFLUT;
//layout(set = 1, binding = B_FILTER_GLOSSY) uniform samplerCube prefilteredMap;
layout(set = 1, binding = B_IMPORT_SMPL) uniform sampler2D environmentSamplingData;
layout(set = 1, binding = B_TEXTURES) uniform sampler2D texturesMap[]; // all textures

// clang-format on


#define GGX_MIN_ROUGHNESS 0.0001f


// Return the vertex position
vec3 getVertex(uint index)
{
  vec3 vp;
  vp.x = VertexBuf.v[3 * index + 0];
  vp.y = VertexBuf.v[3 * index + 1];
  vp.z = VertexBuf.v[3 * index + 2];
  return vp;
}

vec3 getNormal(uint index)
{
  vec3 vp;
  vp.x = NormalBuf.v[3 * index + 0];
  vp.y = NormalBuf.v[3 * index + 1];
  vp.z = NormalBuf.v[3 * index + 2];
  return vp;
}

vec2 getTexCoord(uint index)
{
  vec2 vp;
  vp.x = UvBuf.v[2 * index + 0];
  vp.y = UvBuf.v[2 * index + 1];
  return vp;
}

vec4 getTangents(uint index)
{
  vec4 vp;
  vp.x = TangentBuf.v[4 * index + 0];
  vp.y = TangentBuf.v[4 * index + 1];
  vp.z = TangentBuf.v[4 * index + 2];
  vp.w = TangentBuf.v[4 * index + 3];
  return vp;
}

// Structure of what a vertex is
struct ShadingState
{
  vec3 pos;
  vec3 normal;
  vec3 geom_normal;
  vec2 texcoord0;
  vec3 tangent;
  vec3 bitangent;
  uint matIndex;
};

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

//--------------------------------------------------------------
// Getting the interpolated vertex
// gl_InstanceID gives the Instance Info
// gl_PrimitiveID gives the triangle for this instance
//
ShadingState getShadingState()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfo pinfo = InstanceInfo.i[gl_InstanceCustomIndexNV];

  // Getting the 'first index' for this mesh (offset of the mesh + offset of the triangle)
  uint indexOffset  = pinfo.indexOffset + (3 * gl_PrimitiveID);
  uint vertexOffset = pinfo.vertexOffset;           // Vertex offset as defined in glTF
  uint matIndex     = max(0, pinfo.materialIndex);  // material of primitive mesh

  // Getting the 3 indices of the triangle (local)
  ivec3 triangleIndex = ivec3(indices.i[indexOffset + 0], indices.i[indexOffset + 1], indices.i[indexOffset + 2]);
  triangleIndex += ivec3(vertexOffset);  // (global)

  const vec3 barycentric = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Position
  const vec3 pos0           = getVertex(triangleIndex.x);
  const vec3 pos1           = getVertex(triangleIndex.y);
  const vec3 pos2           = getVertex(triangleIndex.z);
  const vec3 position       = pos0 * barycentric.x + pos1 * barycentric.y + pos2 * barycentric.z;
  const vec3 world_position = vec3(gl_ObjectToWorldNV * vec4(position, 1.0));

  // Normal
  const vec3 nrm0              = getNormal(triangleIndex.x);
  const vec3 nrm1              = getNormal(triangleIndex.y);
  const vec3 nrm2              = getNormal(triangleIndex.z);
  const vec3 normal            = normalize(nrm0 * barycentric.x + nrm1 * barycentric.y + nrm2 * barycentric.z);
  const vec3 world_normal      = normalize(vec3(normal * gl_WorldToObjectNV));
  const vec3 geom_normal       = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       world_geom_normal = normalize(vec3(geom_normal * gl_WorldToObjectNV));

  // flip geometry normal to the side of the incident ray
  if(dot(world_geom_normal, gl_WorldRayDirectionNV) > 0.0)
  {
    world_geom_normal *= -1.0f;
  }

  // Texture coord
  const vec2 uv0       = getTexCoord(triangleIndex.x);
  const vec2 uv1       = getTexCoord(triangleIndex.y);
  const vec2 uv2       = getTexCoord(triangleIndex.z);
  const vec2 texcoord0 = uv0 * barycentric.x + uv1 * barycentric.y + uv2 * barycentric.z;


  ShadingState state;
  state.pos         = world_position;
  state.normal      = world_normal;
  state.geom_normal = world_geom_normal;
  state.texcoord0   = texcoord0;
  state.matIndex    = matIndex;

  // Tangent and Binormal
  int normalTexture = MaterialBuffer.m[state.matIndex].normalTexture;
  if(normalTexture > -1)
  {
    const vec4 tangent0  = getTangents(triangleIndex.x);
    const vec4 tangent1  = getTangents(triangleIndex.y);
    const vec4 tangent2  = getTangents(triangleIndex.z);
    const vec4 tangent   = tangent0 * barycentric.x + tangent1 * barycentric.y + tangent2 * barycentric.z;
    const vec3 bitangent = normalize(cross(normal, tangent.xyz)) * tangent.w;

    vec3 normalVector  = texture(texturesMap[nonuniformEXT(normalTexture)], texcoord0).xyz;
    normalVector       = normalize(normalVector * 2.0 - 1.0);
    mat3 TBN           = mat3(tangent.xyz, bitangent, normal);
    vec3 pnormal       = normalize(TBN * normalVector);
    vec3 world_pnormal = normalize(vec3(pnormal * gl_WorldToObjectNV));

    state.normal    = world_pnormal;
    state.tangent   = tangent.xyz;
    state.bitangent = bitangent.xyz;
  }


  // Move normal to same side as geometric normal
  if(dot(state.normal, state.geom_normal) <= 0)
  {
    state.normal *= -1.0f;
  }

  return state;
}


//
// This fragment shader defines a reference implementation for Physically Based Shading of
// a microfacet surface material defined by a glTF model.
// See https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/master/src/shaders/pbr.frag
//
// References:
// [1] Real Shading in Unreal Engine 4
//     http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
// [2] Physically Based Shading at Disney
//     http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] README.md - Environment Maps
//     https://github.com/KhronosGroup/glTF-WebGL-PBR/#environment-maps
// [4] "An Inexpensive BRDF Model for Physically based Rendering" by Christophe Schlick
//     https://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf


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

  // light from the environment
  vec3 radiance = environment_sample(  // (see common.hlsl)
      environmentTexture,              // assuming lat long map
      environmentSamplingData,         // importance sampling data of the environment map
      seed, to_light, pdf);

  radiance *= pushC.environmentIntensityFactor;

  return radiance / pdf;
}


void main()
{
  // Get the shading information
  ShadingState state = getShadingState();  //ind, vertexOffset, barycentrics);

  // Retrieve the material
  Material material = MaterialBuffer.m[state.matIndex];

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
  baseColor = material.pbrBaseColorFactor;
  if(material.pbrBaseColorTexture > -1)
    baseColor *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], state.texcoord0), 2.2);

  if(material.shadingModel == MATERIAL_METALLICROUGHNESS)
  {
    perceptualRoughness = material.pbrRoughnessFactor;
    metallic            = material.pbrMetallicFactor;

    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
    if(material.pbrMetallicRoughnessTexture > -1)
    {
      vec4 mrSample = texture(texturesMap[nonuniformEXT(material.pbrMetallicRoughnessTexture)], state.texcoord0);
      perceptualRoughness *= mrSample.g;
      metallic *= mrSample.b;
    }

    diffuseColor  = baseColor.rgb * (vec3(1.0) - f0) * (1.0 - metallic);
    specularColor = mix(f0, baseColor.rgb, metallic);
  }
  else if(material.shadingModel == MATERIAL_SPECULARGLOSSINESS)
  {
    f0                  = material.khrSpecularFactor * 0.08;
    perceptualRoughness = 1.0 - material.khrGlossinessFactor;

    specularColor                  = f0;  // f0 = specular
    float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);

    baseColor = material.khrDiffuseFactor;
    if(material.khrDiffuseTexture > -1)
      baseColor *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.khrDiffuseTexture)], state.texcoord0), 2.2);

    diffuseColor = baseColor.rgb * oneMinusSpecularStrength;
    metallic     = solveMetallic(baseColor.rgb, specularColor, oneMinusSpecularStrength);
  }


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

  MaterialInfo materialInfo = MaterialInfo(perceptualRoughness, specularEnvironmentR0, alphaRoughness, diffuseColor,
                                           specularEnvironmentR90, specularColor);


  payload.contribution = vec3(0.0, 0.0, 0.0);
  vec3 contribution    = vec3(0.0, 0.0, 0.0);
  vec3 viewDir         = normalize(gl_WorldRayOriginNV - state.pos);

  // Calculate lighting contribution from image based lighting source (IBL)
  vec3  to_light;
  float pdf;
  vec3  radiance_over_pdf = sample_lights(to_light, pdf, payload.seed);

  if(dot(to_light, state.geom_normal) > 0)
  {
    // Shading contribution
    vec3 bsdf_color = getPointShade(to_light, materialInfo, state.normal, viewDir);
    contribution += bsdf_color * payload.weight * radiance_over_pdf;
  }

  // Ambient occulsion
  float ao = 1.0;
  if(material.occlusionTexture > -1)
    ao *= texture(texturesMap[nonuniformEXT(material.occlusionTexture)], state.texcoord0).r * material.occlusionTextureStrength;
  contribution = mix(contribution, contribution * ao, 1.0 /*u_OcclusionStrength*/);

  // Emissive term
  vec3 emissive = material.emissiveFactor;
  if(material.emissiveTexture > -1)
    emissive *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.emissiveTexture)], state.texcoord0), 2.2).rgb;
  payload.contribution += emissive;


  // Result Color
  if(ubo.s.debugMode == 0)
  {
    // New Ray  (origin)
    payload.rayOrigin = offsetRay(state.pos, state.geom_normal);

    // New direction and reflectance
    vec3        reflectance = vec3(0);
    vec3        rayDir;
    AngularInfo angularInfo = getAngularInfo(to_light, state.normal, viewDir);
    float       F           = specularReflection(materialInfo, angularInfo).x;
    float       e           = rnd(payload.seed);
    if(e > F)
    {
      vec3 T, B, N;
      N = state.normal;
      createCoordinateSystem(N, T, B);
      // Randomly sample the hemisphere
      payload.rayDir = samplingHemisphere(payload.seed, T, B, N);
      payload.weight *= baseColor.rgb;
    }
    else
    {
      vec3 V = viewDir;
      vec3 N = state.normal;
      // Randomly sample the NDF to get a microfacet in our BRDF to reflect off
      vec3 H = getGGXMicrofacet(rnd2(payload.seed), N, materialInfo.perceptualRoughness);
      // Compute the outgoing direction based on this (perfectly reflective) microfacet
      payload.rayDir = normalize(2.f * dot(V, H) * H - V);
      payload.weight *= materialInfo.specularColor;
    }
  }
  else
  {
    payload.flags  = FLAG_DONE;
    payload.weight = vec3(1.0);
    switch(ubo.s.debugMode)
    {
      case 1:
        payload.contribution = vec3(metallic);
        return;
      case 2:
        payload.contribution = (state.normal + vec3(1)) * .5;
        return;
      case 3:
        payload.contribution = baseColor.xyz;
        return;
      case 4:
        payload.contribution = vec3(ao);
        return;
      case 5:
        payload.contribution = emissive;
        return;
      case 6:
        payload.contribution = f0;
        return;
      case 7:
        payload.contribution = vec3(baseColor.a);
        return;
      case 8:
        payload.contribution = vec3(perceptualRoughness);
        return;
      case 9:
        payload.contribution = vec3(state.texcoord0, 1);
        return;
      case 10:
        payload.contribution = vec3(state.tangent + vec3(1)) * .5;
        return;
    };
  }


  // Add contribution from next event estimation if not shadowed
  //---------------------------------------------------------------------------------------------

  // cast a shadow ray; assuming light is always outside
  vec3 origin    = offsetRay(state.pos, state.geom_normal);
  vec3 direction = normalize(to_light);

  // prepare the ray and payload but trace at the end to reduce the amount of data that has
  // to be recovered after coming back from the shadow trace
  shadow_payload.isHit = true;
  shadow_payload.seed  = 0;
  uint rayFlags        = gl_RayFlagsTerminateOnFirstHitNV | gl_RayFlagsSkipClosestHitShaderNV;

  traceNV(topLevelAS,          // acceleration structure
          rayFlags,            // rayFlags
          0xFF,                // cullMask
          0,                   // sbtRecordOffset
          0,                   // sbtRecordStride
          1,                   // missIndex
          origin,              // ray origin
          0.0,                 // ray min range
          direction,           // ray direction
          pushC.maxRayLenght,  // ray max range
          1                    // payload layout(location = 1)
  );

  // add to ray contribution from next event estiation
  if(!shadow_payload.isHit)
    payload.contribution += contribution;
}

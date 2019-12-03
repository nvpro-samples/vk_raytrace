#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
//#extension GL_KHR_vulkan_glsl : enable

#include "binding.h"
#include "share.h"


const float M_PI             = 3.141592653589793;
const float c_MinReflectance = 0.04;

// Payload information of the ray returning: 0 hit, 2 shadow
layout(location = 0) rayPayloadInNV PerRayData_raytrace prd;
layout(location = 2) rayPayloadNV bool prdIsInShadow;

layout(push_constant) uniform Constants
{
  int   frame;
  int   depth;
  int   samples;
  float hdrMultiplier;
};


// Raytracing hit attributes: barycentrics
hitAttributeNV vec3 attribs;

layout(set = 0, binding = 0) uniform accelerationStructureNV topLevelAS;

// clang-format off
layout(set = 1, binding = B_SCENE) uniform _ubo {Scene s;} ubo;
layout(set = 1, binding = B_PRIM_INFO) readonly buffer _instanceInfo {primInfo i[];} instanceInfo;
layout(set = 1, binding = B_VERTICES) readonly buffer _VertexBuf {float v[];} VertexBuf;
layout(set = 1, binding = B_INDICES) readonly buffer _Indices {uint i[]; } indices;
layout(set = 1, binding = B_NORMALS) readonly buffer _NormalBuf {float v[];} NormalBuf;
layout(set = 1, binding = B_TEXCOORDS) readonly buffer _UvBuf {float v[];} UvBuf;
layout(set = 1, binding = B_MATRICES) readonly buffer _MatrixBuffer {InstancesMatrices m[];} MatrixBuffer;
layout(set = 1, binding = B_MATERIAL) readonly buffer _MaterialBuffer {Material m[];} MaterialBuffer;
layout(set = 1, binding = B_HDR) uniform sampler2D samplerEnv;
//layout(set = 1, binding = B_FILTER_DIFFUSE) uniform samplerCube samplerIrradiance;
layout(set = 1, binding = B_LUT_BRDF) uniform sampler2D samplerBRDFLUT;
//layout(set = 1, binding = B_FILTER_GLOSSY) uniform samplerCube prefilteredMap;
layout(set = 1, binding = B_IMPORT_SMPL) uniform sampler2D env_accel_tex;

// clang-format on

// All textures
layout(set = 2, binding = 0) uniform sampler2D albedoMap[];
layout(set = 2, binding = 1) uniform sampler2D normalMap[];
layout(set = 2, binding = 2) uniform sampler2D occlusionMap[];
layout(set = 2, binding = 3) uniform sampler2D metallicRoughness[];
layout(set = 2, binding = 4) uniform sampler2D emissiveMap[];


#include "sampling.glsl"


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

// Structure of what a vertex is
struct ShadingState
{
  vec3 pos;
  vec3 nrm;
  vec3 geo_nrm;
  vec2 texcoord0;
  vec3 tangent;
  vec3 binormal;
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


void computeTangent(in vec3 pos0, in vec3 pos1, in vec3 pos2, in vec2 uv0, in vec2 uv1, in vec2 uv2, in vec3 geo_normal, out vec3 tangent, out vec3 binormal)
{
  // Tangent and Binormal
  // http://www.terathon.com/code/tangent.html
  float x1 = pos1.x - pos0.x;
  float x2 = pos2.x - pos0.x;
  float y1 = pos1.y - pos0.y;
  float y2 = pos2.y - pos0.y;
  float z1 = pos1.z - pos0.z;
  float z2 = pos2.z - pos0.z;

  float s1 = uv1.x - uv0.x;
  float s2 = uv2.x - uv0.x;
  float t1 = uv1.y - uv0.y;
  float t2 = uv2.y - uv0.y;

  float r    = 1.0F / (s1 * t2 - s2 * t1);
  vec3  sdir = vec3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
  vec3  tdir = vec3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

  vec3  N          = normalize(geo_normal);
  vec3  T          = normalize(sdir - N * dot(N, sdir));
  float handedness = (dot(cross(N, sdir), tdir) < 0.0F) ? -1.0F : 1.0F;
  vec3  B          = normalize(cross(N, T)) * handedness;

  tangent  = T;
  binormal = B;
}


//--------------------------------------------------------------
// Getting the interpolated vertex
// gl_InstanceID gives the Instance Info
// gl_PrimitiveID gives the triangle for this instance
//
ShadingState getShadingState()
{
  // Getting the 'first index' for this instance (offset of the instance + offset of the triangle)
  uint indexOffset = instanceInfo.i[gl_InstanceID].indexOffset + (3 * gl_PrimitiveID);

  // Vertex offset as defined in glTF
  uint vertexOffset = instanceInfo.i[gl_InstanceID].vertexOffset;

  uint matIndex = instanceInfo.i[gl_InstanceID].materialIndex;

  // Getting the 3 indices of the triangle
  ivec3 trianglIndex = ivec3(indices.i[indexOffset + 0], indices.i[indexOffset + 1], indices.i[indexOffset + 2]);
  trianglIndex += ivec3(vertexOffset);

  const vec3 barycentric = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  const vec3 pos0        = getVertex(trianglIndex.x);
  const vec3 pos1        = getVertex(trianglIndex.y);
  const vec3 pos2        = getVertex(trianglIndex.z);
  const vec3 geom_normal = normalize(cross(pos1 - pos0, pos2 - pos0));
  const vec3 position    = getVertex(trianglIndex.x) * barycentric.x + getVertex(trianglIndex.y) * barycentric.y
                        + getVertex(trianglIndex.z) * barycentric.z;

  const vec3 normal = normalize(getNormal(trianglIndex.x) * barycentric.x    //
                                + getNormal(trianglIndex.y) * barycentric.y  //
                                + getNormal(trianglIndex.z) * barycentric.z);

  const vec3 world_position = vec3(MatrixBuffer.m[gl_InstanceID].world * vec4(position, 1.0));

  // transform normals using inverse transpose
  vec3 world_geom_normal = normalize(vec3(MatrixBuffer.m[gl_InstanceID].worldIT * vec4(geom_normal, 0.0)));
  vec3 world_normal      = normalize(vec3(MatrixBuffer.m[gl_InstanceID].worldIT * vec4(normal, 0.0)));

  const vec2 uv0       = getTexCoord(trianglIndex.x);
  const vec2 uv1       = getTexCoord(trianglIndex.y);
  const vec2 uv2       = getTexCoord(trianglIndex.z);
  const vec2 texcoord0 = uv0 * barycentric.x + uv1 * barycentric.y + uv2 * barycentric.z;


  // flip geometry normal to the side of the incident ray
  if(dot(world_geom_normal, gl_WorldRayDirectionNV) > 0.0)
  {
    world_geom_normal *= -1.0f;
  }

  ShadingState state;
  state.pos       = world_position;
  state.nrm       = world_normal;
  state.geo_nrm   = world_geom_normal;
  state.texcoord0 = texcoord0;
  state.matIndex  = matIndex;

  // Tangent and Binormal
  vec3 tangentNormal = texture(normalMap[matIndex], texcoord0).xyz;
  if(length(tangentNormal) > 0.)
  {
    vec3 tangent, binormal;
    computeTangent(pos0, pos1, pos2, uv0, uv1, uv2, normal, tangent, binormal);
    mat3 TBN           = mat3(tangent, binormal, normal);
    tangentNormal      = normalize(tangentNormal * 2.0 - 1.0);
    vec3 pnormal       = normalize(TBN * tangentNormal);
    vec3 world_pnormal = normalize(vec3(MatrixBuffer.m[gl_InstanceID].worldIT * vec4(pnormal, 0.0)));

    const vec3 world_tangent  = normalize(vec3(MatrixBuffer.m[gl_InstanceID].worldIT * vec4(tangent, 0.0)));
    const vec3 world_binormal = normalize(vec3(MatrixBuffer.m[gl_InstanceID].worldIT * vec4(binormal, 0.0)));

    state.nrm      = world_pnormal;
    state.tangent  = world_tangent;
    state.binormal = world_binormal;
  }


  // Move normal to same side as geometric normal
  if(dot(state.nrm, state.geo_nrm) <= 0)
  {
    state.nrm *= -1.0f;
  }


  return state;
}


//
// This fragment shader defines a reference implementation for Physically Based Shading of
// a microfacet surface material defined by a glTF model.
// See https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/master/src/shaders/metallic-roughness.frag
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

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
float getRangeAttenuation(float range, float distance)
{
  if(range < 0.0)
  {
    // negative range means unlimited
    return 1.0;
  }
  return max(min(1.0 - pow(distance / range, 4.0), 1.0), 0.0) / pow(distance, 2.0);
}

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#inner-and-outer-cone-angles
float getSpotAttenuation(vec3 pointToLight, vec3 spotDirection, float outerConeCos, float innerConeCos)
{
  float actualCos = dot(normalize(spotDirection), normalize(-pointToLight));
  if(actualCos > outerConeCos)
  {
    if(actualCos < innerConeCos)
    {
      return smoothstep(outerConeCos, innerConeCos, actualCos);
    }
    return 1.0;
  }
  return 0.0;
}

vec3 applyDirectionalLight(Light light, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  vec3 pointToLight = -light.direction;
  vec3 shade        = getPointShade(pointToLight, materialInfo, normal, view);
  return light.intensity * light.color * shade;
}

vec3 applyPointLight(ShadingState state, Light light, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  vec3  pointToLight = light.position - state.pos;
  float distance     = length(pointToLight);
  float attenuation  = getRangeAttenuation(light.range, distance);
  vec3  shade        = getPointShade(pointToLight, materialInfo, normal, view);
  return attenuation * light.intensity * light.color * shade;
}

vec3 applySpotLight(ShadingState state, Light light, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  vec3  pointToLight     = light.position - state.pos;
  float distance         = length(pointToLight);
  float rangeAttenuation = getRangeAttenuation(light.range, distance);
  float spotAttenuation  = getSpotAttenuation(pointToLight, light.direction, light.outerConeCos, light.innerConeCos);
  vec3  shade            = getPointShade(pointToLight, materialInfo, normal, view);
  return rangeAttenuation * spotAttenuation * light.intensity * light.color * shade;
}

//---------------------------------------------------------------------------------------------------------
// Calculation of the lighting contribution from an optional Image Based Light source.
// Precomputed Environment Maps are required uniform inputs and are computed as outlined in [1].
// See our README.md on Environment Maps [3] for additional discussion.
vec3 getIBLContribution(MaterialInfo materialInfo, vec3 n, vec3 v, vec3 origin)
{
  vec3  to_light;
  float pdf;

  // sampling light from the environment
  vec3 radiance = environment_sample(  // (see sampling.glsl)
      samplerEnv,                      // assuming lat long map
      env_accel_tex,                   // importance sampling data of the environment map
      prd.seed, to_light, pdf);

  radiance *= hdrMultiplier;
  const float cos_theta = dot(to_light, n);
  if((cos_theta > 0.0f) && pdf != 0.0f)
  {
    // Shoot shadow ray towar the light
    prdIsInShadow = true;

    uint rayFlags = gl_RayFlagsTerminateOnFirstHitNV | gl_RayFlagsSkipClosestHitShaderNV;
    traceNV(topLevelAS,  // acceleration structure
            rayFlags,    // rayFlags
            0xFF,        // cullMask
            0,           // sbtRecordOffset
            0,           // sbtRecordStride
            1,           // missIndex
            origin,      // ray origin
            0.0,         // ray min range
            to_light,    // ray direction
            100000.0,    // ray max range
            2            // payload layout(location = 2)
    );

    // Hit an object, no light contribution
    if(prdIsInShadow)
      return vec3(0, 0, 0);

    // Shading contribution
    vec3 shade = getPointShade(to_light, materialInfo, n, v);

    // Adding light contribution
    vec3 radiance_over_pdf = radiance / pdf;
    return radiance_over_pdf * shade;
  }
  else
  {
    return vec3(0, 0, 0);
  }
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
  baseColor = SRGBtoLINEAR(texture(albedoMap[state.matIndex], state.texcoord0), 2.2) * material.baseColorFactor;

  if(material.shadingModel == MATERIAL_METALLICROUGHNESS)
  {
    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
    vec4 mrSample       = texture(metallicRoughness[state.matIndex], state.texcoord0);
    perceptualRoughness = mrSample.g * material.roughnessFactor;
    metallic            = mrSample.b * material.metallicFactor;

    diffuseColor  = baseColor.rgb * (vec3(1.0) - f0) * (1.0 - metallic);
    specularColor = mix(f0, baseColor.rgb, metallic);
  }
  else if(material.shadingModel == MATERIAL_SPECULARGLOSSINESS)
  {
    f0                  = material.specularFactor;
    perceptualRoughness = 1.0 - material.glossinessFactor;

    specularColor                  = f0;  // f0 = specular
    float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);
    diffuseColor                   = baseColor.rgb * oneMinusSpecularStrength;
    metallic                       = solveMetallic(baseColor.rgb, specularColor, oneMinusSpecularStrength);
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


  vec3 color  = vec3(0.0, 0.0, 0.0);
  vec3 normal = state.nrm;
  vec3 view   = normalize(gl_WorldRayOriginNV - state.pos);

  // Calculate lighting contribution from image based lighting source (IBL)
  vec3 origin = offsetRay(state.pos, state.geo_nrm);
  color += getIBLContribution(materialInfo, normal, view, origin);

  // Ambient occulsion
  float ao = 1.0;
  ao       = texture(occlusionMap[state.matIndex], state.texcoord0).r;
  color    = mix(color, color * ao, 1.0 /*u_OcclusionStrength*/);

  // Emissive term
  vec3 emissive = vec3(0);
  emissive = SRGBtoLINEAR(texture(emissiveMap[state.matIndex], state.texcoord0), 2.2).rgb * material.emissiveFactor * 10.0f;
  color += emissive;

  // Result Color
  prd.result += color * prd.importance * baseColor.a;

  // For the next Trace
  vec3 rayDir;

  {
    float NdotV           = clamp(dot(normal, view), 0.0, 1.0);
    vec3  reflection      = normalize(reflect(-view, normal));
    vec2  brdfSamplePoint = clamp(vec2(NdotV, materialInfo.perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2  brdf = texture(samplerBRDFLUT, brdfSamplePoint).rg;  // retrieve a scale and bias to F0. See [1], Figure 3

    prd.importance *= metallic * (materialInfo.specularColor * brdf.x + brdf.y);
    prd.roughness = perceptualRoughness;

    origin = offsetRay(state.pos, state.geo_nrm);
    rayDir = reflection;
  }

  // Incrementing the depth
  prd.depth += 1;

  bool traceContinue = (prd.depth < depth) && (length(prd.importance) > 0.01);
  if(traceContinue)
  {
    traceNV(topLevelAS,         // acceleration structure
            gl_RayFlagsNoneNV,  // rayFlags
            0xFF,               // cullMask
            0,                  // sbtRecordOffset
            0,                  // sbtRecordStride
            0,                  // missIndex
            origin,             // ray origin
            0.0,                // ray min range
            rayDir,             // ray direction
            100000.0,           // ray max range
            0                   // payload
    );
  }

  // Back to current depth
  prd.depth -= 1;
}

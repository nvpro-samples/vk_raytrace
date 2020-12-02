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


// Align structure layout to scalar
#extension GL_EXT_scalar_block_layout : enable


// Math defines
const highp float M_PI   = 3.14159265358979323846;   // pi
const highp float M_PI_2 = 1.57079632679489661923;   // pi/2
const highp float M_PI_4 = 0.785398163397448309616;  // pi/4
const highp float M_1_PI = 0.318309886183790671538;  // 1/pi
const highp float M_2_PI = 0.636619772367581343076;  // 2/pi


// Sets and Bindings for resources
#include "../binding.h"  // Descriptor binding values
#include "../structures.h"

// Sun and Sky environment shader
#ifdef USE_SUN_AND_SKY
#include "sun_and_sky.h"
#endif


// Flags for ray and material
const uint FLAG_NONE               = 0u;
const uint FLAG_INSIDE             = 1u;
const uint FLAG_DONE               = 2u;
const uint FLAG_FIRST_PATH_SEGMENT = 4u;


// clang-format off
void add_flag(inout uint flags, uint to_add) { flags |= to_add; }
void toggle_flag(inout uint flags, uint to_toggle) { flags ^= to_toggle; }
void remove_flag(inout uint flags, uint to_remove) {flags &= ~to_remove; }
bool has_flag(uint flags, uint to_check) { return (flags & to_check) != 0; }
// clang-format on

//----------------------------------------------
// Common structures
//----------------------------------------------

// Hit payload structure, returned information after a hit
struct HitPayload
{
  uint  seed;
  vec3  contribution;  // Hit value
  vec3  weight;        // weight of the contribution
  vec3  rayOrigin;
  vec3  rayDirection;
  float last_pdf;
  uint  flags;
};

// Payload for Shadow
struct ShadowHitPayload
{
  uint seed;  // Need to be in first position as it is shared with HitPayload
  bool isHit;
};

// Hit state
// Information on the hit shared between Rtx Pipeline and RayQuery for shading
struct HitState
{
  uint   InstanceID;
  uint   PrimitiveID;
  vec2   bary;
  int    InstanceCustomIndex;
  vec3   WorldRayOrigin;
  mat4x3 ObjectToWorld;
  mat4x3 WorldToObject;
};


// Shading information used by the material
struct ShadeState
{
  vec3 normal;
  vec3 geom_normal;
  vec3 position;
  vec2 text_coords[1];
  vec3 tangent_u[1];
  vec3 tangent_v[1];
  uint matIndex;
};


// GLTF material
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2
struct GltfMaterial
{
  int shadingModel;  // 0: metallic-roughness, 1: specular-glossiness

  // PbrMetallicRoughness
  vec4  pbrBaseColorFactor;
  int   pbrBaseColorTexture;
  float pbrMetallicFactor;
  float pbrRoughnessFactor;
  int   pbrMetallicRoughnessTexture;

  // KHR_materials_pbrSpecularGlossiness
  vec4  khrDiffuseFactor;
  int   khrDiffuseTexture;
  vec3  khrSpecularFactor;
  float khrGlossinessFactor;
  int   khrSpecularGlossinessTexture;

  int   emissiveTexture;
  vec3  emissiveFactor;
  int   alphaMode;
  float alphaCutoff;
  int   doubleSided;

  int   normalTexture;
  float normalTextureScale;
  int   occlusionTexture;
  float occlusionTextureStrength;
};


//----------------------------------------------
// Descriptor Set Layout
//----------------------------------------------

// clang-format off
#ifdef USE_ACCEL
#extension GL_EXT_ray_tracing : require         // This is about ray tracing
layout(set = S_ACCEL, binding = B_TLAS) uniform accelerationStructureEXT topLevelAS;
#endif

#ifdef USE_STOREIMAGE
#extension GL_EXT_shader_image_load_formatted : enable
layout(set = S_OUT, binding = 1) uniform image2D resultImage;
#endif

#ifdef USE_SCENE
layout(set = S_SCENE, binding = B_PRIMLOOKUP) readonly buffer _InstanceInfo {RtPrimitiveLookup primInfo[];};
layout(set = S_SCENE, binding = B_CAMERA) uniform _CameraMatrices { CameraMatrices cameraMatrices; };
layout(set = S_SCENE, binding = B_VERTICES) readonly buffer _VertexBuf {float vertices[];};
layout(set = S_SCENE, binding = B_INDICES) readonly buffer _Indices {uint indices[];};
layout(set = S_SCENE, binding = B_NORMALS) readonly buffer _NormalBuf {float normals[];};
layout(set = S_SCENE, binding = B_TEXCOORDS) readonly buffer _TexCoordBuf {float texcoord0[];};
layout(set = S_SCENE, binding = B_TANGENTS) readonly buffer _TangentBuf {float tangents[];};
layout(set = S_SCENE, binding = B_MATERIALS) readonly buffer _MaterialBuffer {GltfMaterial materials[];};
layout(set = S_SCENE, binding = B_TEXTURES) uniform sampler2D texturesMap[]; // all textures
layout(set = S_SCENE, binding = B_MATRICES) buffer _Matrices { InstanceMatrices matrices[]; };
#endif


#ifdef USE_SUN_AND_SKY
layout(set = S_ENV, binding = B_SUNANDSKY) uniform _SunAndSkyBuffer {SunAndSky _sunAndSky;};
layout(set = S_ENV, binding = B_HDR) uniform sampler2D environmentTexture;
layout(set = S_ENV, binding = B_IMPORT_SMPL) uniform sampler2D environmentSamplingData;
#endif


// clang-format on


#ifdef USE_RANDOM

// https://www.pcg-random.org/
uint pcg(uint v)
{
  uint state = v * 747796405u + 2891336453u;
  uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

uvec2 pcg2d(uvec2 v)
{
  v = v * 1664525u + 1013904223u;

  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;

  v = v ^ (v >> 16u);

  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;

  v = v ^ (v >> 16u);

  return v;
}

// 9 imad (+ 6 iops with final shuffle)
uvec3 pcg3d(uvec3 v)
{

  v = v * 1664525u + 1013904223u;

  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;

  v ^= v >> 16u;

  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;

  return v;
}

// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float in [0, 1) given the previous RNG state
float rnd(inout uint prev)
{
  prev = pcg(prev);
  return (float(prev) * (1.0 / float(0xffffffffu)));
  //return (float(lcg(prev)) / float(0x01000000));
}

vec2 rnd2(inout uint prev)
{
  return vec2(rnd(prev), rnd(prev));
  //  return vec2(float(lcg(prev)) / float(0x01000000), float(lcg(prev)) / float(0x01000000));
}
#endif


//-------------------------------------------------------------------------------------------------
// Sampling
//-------------------------------------------------------------------------------------------------
#ifdef USE_SAMPLING
// Randomly sampling around +Z
vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(1.0 - r2);

  vec3 direction = vec3(cos(2 * M_PI * r1) * sq, sin(2 * M_PI * r1) * sq, sqrt(r2));
  direction      = direction.x * x + direction.y * y + direction.z * z;

  return direction;
}

// Return the tangent and binormal from the incoming normal
void createCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
  if(abs(N.x) > abs(N.y))
    Nt = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
  else
    Nt = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
  Nb = cross(N, Nt);
}
#endif

// Debugging
vec3 integerToColor(uint val)
{
  const vec3 freq = vec3(1.33333f, 2.33333f, 3.33333f);
  return vec3(sin(freq * val) * .5 + .5);
}

///////////////////////////////////////////

#ifdef USE_SHADING

#extension GL_EXT_nonuniform_qualifier : enable  // To access unsized descriptor arrays

// Return the vertex position
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


void ShadeNothing(in HitState state, inout HitPayload prd)
{
  prd.contribution = vec3(.1 * integerToColor(state.PrimitiveID));
  prd.weight       = vec3(1 / M_PI);
  prd.flags        = FLAG_DONE;
}


void Shade(in HitState state, inout HitPayload prd)
{
  //  ShadeNothing(state, prd);
  //  return;


  // Retrieve the Primitive mesh buffer information
  RtPrimitiveLookup pinfo = primInfo[state.InstanceCustomIndex];

  // Getting the 'first index' for this mesh (offset of the mesh + offset of the triangle)
  const uint indexOffset  = pinfo.indexOffset + (3 * state.PrimitiveID);
  const uint vertexOffset = pinfo.vertexOffset;           // Vertex offset as defined in glTF
  const uint matIndex     = max(0, pinfo.materialIndex);  // material of primitive mesh

  // Getting the 3 indices of the triangle (local)
  ivec3 triangleIndex = ivec3(indices[nonuniformEXT(indexOffset + 0)],  //
                              indices[nonuniformEXT(indexOffset + 1)],  //
                              indices[nonuniformEXT(indexOffset + 2)]);
  triangleIndex += ivec3(vertexOffset);  // (global)

  const vec3 barycentrics = vec3(1.0 - state.bary.x - state.bary.y, state.bary.x, state.bary.y);

  // Vertex of the triangle
  const vec3 pos0           = getVertex(triangleIndex.x);
  const vec3 pos1           = getVertex(triangleIndex.y);
  const vec3 pos2           = getVertex(triangleIndex.z);
  const vec3 position       = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  const vec3 world_position = vec3(state.ObjectToWorld * vec4(position, 1.0));

  // Normal
  const vec3 nrm0         = getNormal(triangleIndex.x);
  const vec3 nrm1         = getNormal(triangleIndex.y);
  const vec3 nrm2         = getNormal(triangleIndex.z);
  const vec3 normal       = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 world_normal = normalize(vec3(normal * state.WorldToObject));
  const vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));


  // Tangent and Binormal
  const vec4 tng0     = getTangent(triangleIndex.x);
  const vec4 tng1     = getTangent(triangleIndex.y);
  const vec4 tng2     = getTangent(triangleIndex.z);
  vec4       tangent  = (tng0 * barycentrics.x + tng1 * barycentrics.y + tng2 * barycentrics.z);
  tangent.xyz         = normalize(tangent.xyz);
  vec3 world_tangent  = normalize(vec3(mat4(state.ObjectToWorld) * vec4(tangent.xyz, 0)));
  world_tangent       = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
  vec3 world_binormal = cross(world_normal, world_tangent) * tangent.w;

  // TexCoord
  const vec2 uv0       = getTexCoord(triangleIndex.x);
  const vec2 uv1       = getTexCoord(triangleIndex.y);
  const vec2 uv2       = getTexCoord(triangleIndex.z);
  const vec2 texcoord0 = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  // https://en.wikipedia.org/wiki/Path_tracing
  // Material of the object
  GltfMaterial mat       = materials[nonuniformEXT(matIndex)];
  const vec3   emittance = mat.emissiveFactor;

  // Pick a random direction from here and keep going.
  vec3 rtangent, rbitangent;
  createCoordinateSystem(world_normal, rtangent, rbitangent);
  const vec3 rayOrigin    = world_position;
  const vec3 rayDirection = samplingHemisphere(prd.seed, rtangent, rbitangent, world_normal);

  // Probability of the newRay (cosine distributed)
  const float p = 1 / M_PI;

  // Compute the BRDF for this ray (assuming Lambertian reflection)
  const float cos_theta = dot(rayDirection, world_normal);
  vec3        albedo    = mat.pbrBaseColorFactor.xyz;
  if(mat.pbrBaseColorTexture > -1)
  {
    uint txtId = mat.pbrBaseColorTexture;
    albedo *= texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
  }
  const vec3 BRDF = albedo / M_PI;


  prd.rayOrigin    = rayOrigin;
  prd.rayDirection = rayDirection;
  prd.contribution = emittance;
  prd.weight       = BRDF * cos_theta / p;
}
#endif

// Return the UV in a lat-long HDR map
vec2 get_spherical_uv(vec3 v)
{
  float gamma = asin(-v.y);
  float theta = atan(v.z, v.x);

  vec2 uv = vec2(theta * M_1_PI * 0.5, gamma * M_1_PI) + 0.5;
  return uv;
}


#ifdef USE_SUN_AND_SKY

vec3 environmentEval(in vec3 dir)
{
  // SunAndSky ss       = SunAndSky_default();
  vec3 radiance;

  if(_sunAndSky.in_use == 1)
    radiance = sun_and_sky(_sunAndSky, dir);
  else
  {
    vec2 uv  = get_spherical_uv(dir);  // See sampling.glsl
    radiance = texture(environmentTexture, uv).rgb;
  }
  return radiance.xyz;
}

#endif

#ifdef USE_STOREIMAGE
void StoreResult(in image2D img, in ivec2 coord, in int frameNb, in vec3 result)
{
  // Do accumulation over time
  if(frameNb > 0)
  {
    float a         = 1.0f / float(frameNb + 1);
    vec3  old_color = imageLoad(img, coord).xyz;
    imageStore(img, coord, vec4(mix(old_color, result, a), 1.f));
  }
  else
  {
    // First frame, replace the value in the buffer
    imageStore(img, coord, vec4(result, 1.f));
  }
}
#endif

#ifdef USE_INIT_PAYLOAD

// Using clockARB
#extension GL_ARB_shader_clock : enable

HitPayload InitializePayload(in ivec2 coordImage, in ivec2 sizeImage, in CameraMatrices cam, in int frame)
{
  // Initialize the random number
  uvec2 s    = pcg2d(coordImage * int(clockARB()));
  uint  seed = s.x + s.y;

  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  vec2 subpixel_jitter = frame == 0 ? vec2(0.5f, 0.5f) : rnd2(seed);

  // Compute sampling position between [-1 .. 1]
  const vec2 pixelCenter = vec2(coordImage) + subpixel_jitter;
  const vec2 inUV        = pixelCenter / vec2(sizeImage.xy);
  vec2       d           = inUV * 2.0 - 1.0;

  // Compute ray origin and direction
  vec4 origin    = cam.viewInverse * vec4(0, 0, 0, 1);
  vec4 target    = cam.projInverse * vec4(d.x, d.y, 1, 1);
  vec4 direction = cam.viewInverse * vec4(normalize(target.xyz), 0);

  // Payload default values
  HitPayload prd;
  prd.contribution = vec3(0);
  prd.seed         = seed;
  prd.rayOrigin    = origin.xyz;
  prd.rayDirection = direction.xyz;
  prd.weight       = vec3(1);
  prd.last_pdf     = -1.f;
  prd.flags        = FLAG_FIRST_PATH_SEGMENT;

  return prd;
}
#endif


#ifdef USE_COMPUTE__
layout(local_size_x = 32, local_size_y = 2) in;
#extension GL_EXT_shader_8bit_storage : enable  // Using uint_8 ...

ivec2 SampleSizzled()
{
  // Sampling Swizzling
  // Convert 32x2 to 8x8, where the sampling will follow how invocation are done in a subgroup.
  // layout(local_size_x = 32, local_size_y = 2) in;
  ivec2 base   = ivec2(gl_WorkGroupID.xy) * 8;
  ivec2 subset = ivec2(int(gl_LocalInvocationID.x) & 1, int(gl_LocalInvocationID.x) / 2);
  subset += gl_LocalInvocationID.x >= 16 ? ivec2(2, -8) : ivec2(0, 0);
  subset += ivec2(gl_LocalInvocationID.y * 4, 0);
  return base + subset;
}
#elif defined(USE_COMPUTE)
#extension GL_EXT_shader_8bit_storage : enable

layout(local_size_x = 8, local_size_y = 8) in;
ivec2 SampleSizzled()
{
  return ivec2(gl_GlobalInvocationID.xy);
}
#endif

//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-------------------------------------------------------------------------------------------------

vec3 offsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}

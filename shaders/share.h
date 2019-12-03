
struct PerRayData_raytrace
{
  vec3  result;
  vec3  importance;
  float roughness;
  uint  seed;
  int   depth;
};

struct PerRayData_pick
{
  vec4 worldPos;
  vec4 barycentrics;
  uint instanceID;
  uint primitiveID;
};


// see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual
struct Light
{
  vec3  direction;
  float range;
  vec3  color;
  float intensity;
  vec3  position;
  float innerConeCos;
  float outerConeCos;
  int   type;
  vec2  padding;
};

// Per Instance information
struct primInfo
{
  uint indexOffset;
  uint vertexOffset;
  uint materialIndex;
};

// Matrices buffer for all instances
struct InstancesMatrices
{
  mat4 world;
  mat4 worldIT;
};

struct Scene
{
  mat4  projection;
  mat4  model;
  vec4  camPos;
  int   nbLights;  // w = lightRadiance
  int   _pad1;
  int   _pad2;
  int   _pad3;
  Light lights[10];
};

struct Material
{
  vec4  baseColorFactor;
  vec3  emissiveFactor;
  float metallicFactor;  // 8
  vec3  specularFactor;
  float roughnessFactor;  // 12 -
  int   alphaMode;        // 0: opaque, 1: mask, 2: blend
  float alphaCutoff;
  float glossinessFactor;
  int   shadingModel;  // 16 - 0: metallic-roughness, 1: specular-glossiness
  int   doubleSided;
  int   pad0;
  int   pad1;
  int   pad2;
};

#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1


// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec4 SRGBtoLINEAR(vec4 srgbIn, float gamma)
{
  return vec4(pow(srgbIn.xyz, vec3(gamma)), srgbIn.w);
}

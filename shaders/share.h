
const uint FLAG_NONE               = 0;
const uint FLAG_INSIDE             = 1;
const uint FLAG_DONE               = 2;
const uint FLAG_FIRST_PATH_SEGMENT = 4;

const uint MATERIAL_FLAG_NONE         = 0;
const uint MATERIAL_FLAG_OPAQUE       = 1;  // allows to skip opacity evaluation
const uint MATERIAL_FLAG_DOUBLE_SIDED = 2;  // geometry is only visible from the front side


// clang-format off
void add_flag(inout uint flags, uint to_add) { flags |= to_add; }
void toggle_flag(inout uint flags, uint to_toggle) { flags ^= to_toggle; }
void remove_flag(inout uint flags, uint to_remove) {flags &= ~to_remove; }
bool has_flag(uint flags, uint to_check) { return (flags & to_check) != 0; }
// clang-format on

struct RadianceHitInfo
{
  vec3  contribution;
  vec3  weight;
  vec3  rayOrigin;
  vec3  rayDir;
  uint  seed;
  float last_pdf;
  uint  flags;
};

// Payload for Shadow
struct ShadowHitInfo
{
  bool isHit;
  uint seed;
};


struct PerRayData_pick
{
  vec4 worldPos;
  vec4 barycentrics;
  uint instanceID;
  uint instanceCustomID;
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

// Primitive Mesh information
struct PrimMeshInfo
{
  uint indexOffset;
  uint vertexOffset;
  int  materialIndex;
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
  int   debugMode;
  int   nbLights;  // w = lightRadiance
  int   _pad1;
  int   _pad2;
  Light lights[10];
};

struct Material
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
  bool  doubleSided;

  int   normalTexture;
  float normalTextureScale;
  int   occlusionTexture;
  float occlusionTextureStrength;
};

#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1


// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec4 SRGBtoLINEAR(vec4 srgbIn, float gamma)
{
  return vec4(pow(srgbIn.xyz, vec3(gamma)), srgbIn.w);
}

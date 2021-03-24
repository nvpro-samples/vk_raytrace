
#ifndef GLTFMATERIAL_GLSL
#define GLTFMATERIAL_GLSL 1

#include "sampling.glsl"

//-----------------------------------------------------------------------
#define SRGB_FAST_APPROXIMATION 1
// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
//-----------------------------------------------------------------------
vec4 SRGBtoLINEAR(vec4 srgbIn)
{
#ifdef SRGB_FAST_APPROXIMATION
  vec3 linOut = pow(srgbIn.xyz, vec3(2.2));
#else   //SRGB_FAST_APPROXIMATION
  vec3 bLess  = step(vec3(0.04045), srgbIn.xyz);
  vec3 linOut = mix(srgbIn.xyz / vec3(12.92), pow((srgbIn.xyz + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
#endif  //SRGB_FAST_APPROXIMATION
  return vec4(linOut, srgbIn.w);
}


//-----------------------------------------------------------------------
// Retrieve the diffuse and specular color base on the shading model: Metal-Roughness or Specular-Glossiness
//-----------------------------------------------------------------------
void GetMetallicRoughness(inout State state, in GltfShadeMaterial material)
{
  // KHR_materials_ior
  float dielectricSpecular = (material.ior - 1) / (material.ior + 1);
  dielectricSpecular *= dielectricSpecular;

  float perceptualRoughness = 0.0;
  float metallic            = 0.0;
  vec4  baseColor           = vec4(0.0, 0.0, 0.0, 1.0);
  vec3  f0                  = vec3(dielectricSpecular);

  // Metallic and Roughness material properties are packed together
  // In glTF, these factors can be specified by fixed scalar values
  // or from a metallic-roughness map
  perceptualRoughness = material.pbrRoughnessFactor;
  metallic            = material.pbrMetallicFactor;
  if(material.pbrMetallicRoughnessTexture > -1)
  {
    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
    vec4 mrSample = textureLod(texturesMap[nonuniformEXT(material.pbrMetallicRoughnessTexture)], state.texCoord, 0);
    perceptualRoughness = mrSample.g * perceptualRoughness;
    metallic            = mrSample.b * metallic;
  }

  // The albedo may be defined from a base texture or a flat color
  baseColor = material.pbrBaseColorFactor;
  if(material.pbrBaseColorTexture > -1)
  {
    baseColor *= SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], state.texCoord, 0));
  }

  // baseColor.rgb = mix(baseColor.rgb * (vec3(1.0) - f0), vec3(0), metallic);
  // Specular color (ior 1.4)
  f0 = mix(vec3(dielectricSpecular), baseColor.xyz, metallic);

  state.mat.albedo    = baseColor.xyz;
  state.mat.metallic  = metallic;
  state.mat.roughness = perceptualRoughness;
  state.mat.f0        = f0;
  state.mat.alpha     = baseColor.a;
}

//-------------------------------------------------------------------------------------------------
// Specular-Glossiness converter
// See: // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/js/three.pbrUtilities.js#L34
//-------------------------------------------------------------------------------------------------
const float c_MinReflectance = 0.04;

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float getPerceivedBrightness(vec3 vector)
{
  return sqrt(0.299 * vector.r * vector.r + 0.587 * vector.g * vector.g + 0.114 * vector.b * vector.b);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float solveMetallic(vec3 diffuse, vec3 specular, float oneMinusSpecularStrength)
//-----------------------------------------------------------------------
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
  float D = max(b * b - 4.0 * a * c, 0);

  return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}


//-----------------------------------------------------------------------
// Specular-Glossiness which will be converted to metallic-roughness
//-----------------------------------------------------------------------
void GetSpecularGlossiness(inout State state, in GltfShadeMaterial material)
{
  float perceptualRoughness = 0.0;
  float metallic            = 0.0;
  vec4  baseColor           = vec4(0.0, 0.0, 0.0, 1.0);

  vec3 f0             = material.khrSpecularFactor;
  perceptualRoughness = 1.0 - material.khrGlossinessFactor;

  if(material.khrSpecularGlossinessTexture > -1)
  {
    vec4 sgSample =
        SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(material.khrSpecularGlossinessTexture)], state.texCoord, 0));
    perceptualRoughness = 1 - material.khrGlossinessFactor * sgSample.a;  // glossiness to roughness
    f0 *= sgSample.rgb;                                                   // specular
  }

  vec3  specularColor            = f0;  // f0 = specular
  float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);

  vec4 diffuseColor = material.khrDiffuseFactor;
  if(material.khrDiffuseTexture > -1)
    diffuseColor *= SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(material.khrDiffuseTexture)], state.texCoord, 0));

  baseColor.rgb = diffuseColor.rgb * oneMinusSpecularStrength;
  metallic      = solveMetallic(diffuseColor.rgb, specularColor, oneMinusSpecularStrength);

  state.mat.albedo    = baseColor.xyz;
  state.mat.metallic  = metallic;
  state.mat.roughness = perceptualRoughness;
  state.mat.f0        = f0;
  state.mat.alpha     = baseColor.a;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void GetMaterialsAndTextures(inout State state, in Ray r)
{
  GltfShadeMaterial material = materials[nonuniformEXT(state.matID)];

  state.mat.specular     = 0.5;
  state.mat.subsurface   = 0;
  state.mat.specularTint = 1;
  state.mat.sheen        = 0;
  state.mat.sheenTint    = 0;

  // Uv Transform
  state.texCoord = (vec4(state.texCoord.xy, 1, 1) * material.uvTransform).xy;
  mat3 TBN       = mat3(state.tangent, state.bitangent, state.normal);

  // Perturbating the normal if a normal map is present
  if(material.normalTexture > -1)
  {
    vec3 normalVector = textureLod(texturesMap[nonuniformEXT(material.normalTexture)], state.texCoord, 0).xyz;
    normalVector      = normalize(normalVector * 2.0 - 1.0);
    normalVector *= vec3(material.normalTextureScale, material.normalTextureScale, 1.0);
    state.normal   = normalize(TBN * normalVector);
    state.ffnormal = dot(state.normal, r.direction) <= 0.0 ? state.normal : -state.normal;
    CreateCoordinateSystem(state.ffnormal, state.tangent, state.bitangent);
  }

  // Emissive term
  state.mat.emission = material.emissiveFactor;
  if(material.emissiveTexture > -1)
    state.mat.emission *=
        SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(material.emissiveTexture)], state.texCoord, 0)).rgb;

  // Basic material
  if(material.shadingModel == MATERIAL_METALLICROUGHNESS)
    GetMetallicRoughness(state, material);
  else
    GetSpecularGlossiness(state, material);

  // Clamping roughness
  state.mat.roughness = max(state.mat.roughness, 0.001);


  // KHR_materials_transmission
  state.mat.transmission = material.transmissionFactor;
  if(material.transmissionTexture > -1)
  {
    state.mat.transmission *= textureLod(texturesMap[nonuniformEXT(material.transmissionTexture)], state.texCoord, 0).r;
  }

  // KHR_materials_ior
  state.mat.ior = material.ior;
  state.eta     = dot(state.normal, state.ffnormal) > 0.0 ? (1.0 / state.mat.ior) : state.mat.ior;

  // KHR_materials_unlit
  state.mat.unlit = (material.unlit == 1);

  // KHR_materials_anisotropy
  state.mat.anisotropy = material.anisotropy;
  // Calculate anisotropic roughness along the tangent and bitangent directions
  float aspect = sqrt(1.0 - material.anisotropy * 0.9);
  state.mat.ax = max(0.001, state.mat.roughness / aspect);
  state.mat.ay = max(0.001, state.mat.roughness * aspect);

  // KHR_materials_anisotropy .. rotates the tangents
  if(material.anisotropy > 0)
  {
    state.tangent   = normalize(TBN * material.anisotropyDirection);
    state.bitangent = normalize(cross(state.normal, state.tangent));
  }

  // KHR_materials_volume
  state.mat.attenuationColor    = material.attenuationColor;
  state.mat.attenuationDistance = material.attenuationDistance;
  state.mat.thinwalled          = material.thicknessFactor == 0;

  //KHR_materials_clearcoat
  state.mat.clearcoat          = material.clearcoatFactor;
  state.mat.clearcoatRoughness = material.clearcoatRoughness;
  if(material.clearcoatTexture > -1)
  {
    state.mat.clearcoat *= textureLod(texturesMap[nonuniformEXT(material.clearcoatTexture)], state.texCoord, 0).r;
  }
  if(material.clearcoatRoughnessTexture > -1)
  {
    state.mat.clearcoatRoughness *=
        textureLod(texturesMap[nonuniformEXT(material.clearcoatRoughnessTexture)], state.texCoord, 0).g;
  }
  state.mat.clearcoatRoughness = max(state.mat.clearcoatRoughness, 0.001);
}

#endif  // GLTFMATERIAL_GLSL

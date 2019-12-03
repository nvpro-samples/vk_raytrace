#define NO_DEBUG_OUTPUT 0
#define DEBUG_METALLIC 1
#define DEBUG_NORMAL 2
#define DEBUG_BASECOLOR 3
#define DEBUG_OCCLUSION 4
#define DEBUG_EMISSIVE 5
#define DEBUG_UV 6
#define DEBUG_ALPHA 7
#define DEBUG_ROUGHNESS 8

// Gamma Correction in Computer Graphics
// see https://www.teamten.com/lawrence/graphics/gamma/
vec3 gammaCorrection(vec3 color, float gamma)
{
  return pow(color, vec3(1.0 / gamma));
}

// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec4 SRGBtoLINEAR(vec4 srgbIn, float gamma)
{
  return vec4(pow(srgbIn.xyz, vec3(gamma)), srgbIn.w);
}

// Uncharted 2 tone map
// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 toneMapUncharted2Impl(vec3 color)
{
  const float A = 0.15;
  const float B = 0.50;
  const float C = 0.10;
  const float D = 0.20;
  const float E = 0.02;
  const float F = 0.30;
  return ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
}

vec3 toneMapUncharted(vec3 color, float gamma)
{
  const float W            = 11.2;
  const float ExposureBias = 2.0f;
  color                    = toneMapUncharted2Impl(color * ExposureBias);
  vec3 whiteScale          = 1.0 / toneMapUncharted2Impl(vec3(W));
  return gammaCorrection(color * whiteScale, gamma);
}

// Hejl Richard tone map
// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 toneMapHejlRichard(vec3 color)
{
  color = max(vec3(0.0), color - vec3(0.004));
  return (color * (6.2 * color + .5)) / (color * (6.2 * color + 1.7) + 0.06);
}

// ACES tone map
// see: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 toneMapACES(vec3 color, float gamma)
{
  const float A = 2.51;
  const float B = 0.03;
  const float C = 2.43;
  const float D = 0.59;
  const float E = 0.14;
  return gammaCorrection(clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0, 1.0), gamma);
}

#define TONEMAP_DEFAULT 0
#define TONEMAP_UNCHARTED 1
#define TONEMAP_HEJLRICHARD 2
#define TONEMAP_ACES 3

vec3 toneMap(vec3 color, int tonemap, float gamma, float exposure)
{
  color *= exposure;

  switch(tonemap)
  {
    case TONEMAP_UNCHARTED:
      return toneMapUncharted(color, gamma);
    case TONEMAP_HEJLRICHARD:
      return toneMapHejlRichard(color);
    case TONEMAP_ACES:
      return toneMapACES(color, gamma);
    default:
      return gammaCorrection(color, gamma);
  }
}

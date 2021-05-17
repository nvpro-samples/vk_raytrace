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


#ifndef SUN_AND_SKY_GLSL
#define SUN_AND_SKY_GLSL

#ifndef M_PI
#define M_PI 3.1415926535f
#endif

#ifdef CPP
#define _INLINE_ inline
#else
#define _INLINE_
#endif

struct SunAndSky
{
  vec3  rgb_unit_conversion;
  float multiplier;

  float haze;
  float redblueshift;
  float saturation;
  float horizon_height;

  vec3  ground_color;
  float horizon_blur;

  vec3  night_color;
  float sun_disk_intensity;

  vec3  sun_direction;
  float sun_disk_scale;

  float sun_glow_intensity;
  int   y_is_up;
  int   physically_scaled_sun;
  int   in_use;
};

_INLINE_ SunAndSky SunAndSky_default()
{
  SunAndSky ss;
  ss.multiplier            = 0.0000101320f;
  ss.rgb_unit_conversion   = vec3(1);
  ss.haze                  = 0.0;
  ss.redblueshift          = 0.0;
  ss.saturation            = 1.0;
  ss.horizon_height        = 0.0;
  ss.horizon_blur          = 0.1f;
  ss.ground_color          = vec3(0.4f, 0.4f, 0.4f);
  ss.night_color           = vec3(0.0, 0.0, 0.01f);
  ss.sun_direction         = vec3(0.00, 0.78, 0.62f);
  ss.sun_disk_intensity    = 0.8f;
  ss.sun_disk_scale        = 5.0;
  ss.sun_glow_intensity    = 1.0;
  ss.y_is_up               = 1;
  ss.physically_scaled_sun = 1;
  ss.in_use                = 0;
  return ss;
}

#ifndef CPP
/*helper functions for sun_and_sky*/

float luminance(vec3 rgb)
{
  return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}


vec3 xyz2dir(vec3 in_main, float x, float y, float z)
{
  vec3 u;
  vec3 v;

  vec3 omain = in_main;

  if(abs(omain.x) < abs(omain.y))
  {
    // u = n x x_axis
    u = vec3(0.0, -omain.z, omain.y);
  }
  else
  {
    // u = n x y_axis
    u = vec3(omain.z, 0.0, -omain.x);
  }


  // degenerate transform
  if(length(u) == 0.0)
  {
    if(abs(in_main.x) < abs(in_main.y))
    {
      u = vec3(0.0, -in_main.z, in_main.y);
    }
    else
    {
      u = vec3(in_main.z, 0.0, -in_main.x);
    }
  }
  u = normalize(u);
  v = cross(in_main, u);
  return x * u + y * v + z * in_main;
}


vec2 mi_lib_square_to_disk(float inout_r, float inout_phi, float in_x, float in_y)
{
  // map to [-1, 1] x [-1, 1]
  float local_x = 2 * in_x - 1;
  float local_y = 2 * in_y - 1;
  // pathological: avoid 0.0/0.0
  if(local_x == 0.0 && local_y == 0.0)
  {
    inout_phi = 0.0;
    inout_r   = 0.0;
  }
  else
  {
    if(local_x > -local_y)
    {
      if(local_x > local_y)
      {
        inout_r   = local_x;
        inout_phi = (M_PI / 4.0f) * (1.0f + local_y / local_x);
      }
      else
      {
        inout_r   = local_y;
        inout_phi = (M_PI / 4.0f) * (3.0f - local_x / local_y);
      }
    }
    else
    {
      if(local_x < local_y)
      {
        inout_r   = -local_x;
        inout_phi = (M_PI / 4.0f) * (5.0f + local_y / local_x);
      }
      else
      {
        inout_r   = -local_y;
        inout_phi = (M_PI / 4.0f) * (7.0f - local_x / local_y);
      }
    }
  }
  return vec2(inout_r, inout_phi);
}


vec3 mi_reflection_dir_diffuse_x(vec3 in_normal, vec2 in_sample)
{

  vec2  r_phi = mi_lib_square_to_disk(0, 0, in_sample.x, in_sample.y);
  float x     = r_phi.x * cos(r_phi.y);
  float y     = r_phi.x * sin(r_phi.y);
  // compute the z component by "lifting" the point onto the unit
  // hemisphere
  float z2 = 1.0f - x * x - y * y;
  float z;
  if(z2 > 0.0f)
  {
    z = sqrt(z2);
  }
  else
  {
    z = 0.0;
  }

  return xyz2dir(in_normal, x, y, z);
}


vec3 calc_sun_color(vec3 sun_dir, float turbidity)
{
  vec3 sun_color  = vec3(0.0);
  vec3 ko         = vec3(12.0, 8.5, 0.9);
  vec3 wavelength = vec3(0.610, 0.550, 0.470);
  vec3 solRad     = vec3(1.0 * 127500 / 0.9878, 0.992 * 127500 / 0.9878, 0.911 * 127500 / 0.9878);
  if(sun_dir.z > 0.0)
  {
    float m     = (1.0f / (sun_dir.z + 0.15f * pow(93.885f - acos(sun_dir.z) * 180 / M_PI, -1.253f)));
    float beta  = 0.04608f * turbidity - 0.04586f;
    float alpha = 1.3f;
    vec3  ta, to, tr;
    // aerosol (water + dust) attenuation
    ta = exp(-m * beta * pow(wavelength, vec3(-alpha)));
    // ozone absorption
    float l = 0.0035f;
    to      = exp(-m * ko * l);
    // Rayleigh scattering
    tr = exp(-m * 0.008735f * pow(wavelength, vec3(-4.08f)));
    // result
    sun_color = tr * ta * to * solRad;
  }
  return sun_color;
}


vec3 sky_color_xyz(vec3 in_dir, vec3 in_sun_pos, float in_turbidity, float in_luminance)
{
  vec3  xyz;
  float A, B, C, D, E;
  float cos_gamma = dot(in_sun_pos, in_dir);
  if(cos_gamma > 1.0)
  {
    cos_gamma = 2.0f - cos_gamma;
  }
  float gamma         = acos(cos_gamma);
  float cos_theta     = in_dir.z;
  float cos_theta_sun = in_sun_pos.z;
  float theta_sun     = acos(cos_theta_sun);
  float t2            = in_turbidity * in_turbidity;
  float ts2           = theta_sun * theta_sun;
  float ts3           = ts2 * theta_sun;
  // determine x and y at zenith
  float zenith_x = ((+0.001650f * ts3 - 0.003742f * ts2 + 0.002088f * theta_sun + 0) * t2
                    + (-0.029028f * ts3 + 0.063773f * ts2 - 0.032020f * theta_sun + 0.003948f) * in_turbidity
                    + (+0.116936f * ts3 - 0.211960f * ts2 + 0.060523f * theta_sun + 0.258852f));
  float zenith_y = ((+0.002759f * ts3 - 0.006105f * ts2 + 0.003162f * theta_sun + 0) * t2
                    + (-0.042149f * ts3 + 0.089701f * ts2 - 0.041536f * theta_sun + 0.005158f) * in_turbidity
                    + (+0.153467f * ts3 - 0.267568f * ts2 + 0.066698f * theta_sun + 0.266881f));
  xyz.y          = in_luminance;
  // TODO: Preetham/Utah

  A = -0.019257f * in_turbidity - (0.29f - pow(cos_theta_sun, 0.5f) * 0.09f);
  // use flags (see above)
  B                      = -0.066513f * in_turbidity + 0.000818f;
  C                      = -0.000417f * in_turbidity + 0.212479f;
  D                      = -0.064097f * in_turbidity - 0.898875f;
  E                      = -0.003251f * in_turbidity + 0.045178f;
  float x                = (((1.f + A * exp(B / cos_theta)) * (1.f + C * exp(D * gamma) + E * cos_gamma * cos_gamma))
             / ((1 + A * exp(B / 1.0)) * (1 + C * exp(D * theta_sun) + E * cos_theta_sun * cos_theta_sun)));
  A                      = -0.016698f * in_turbidity - 0.260787f;
  B                      = -0.094958f * in_turbidity + 0.009213f;
  C                      = -0.007928f * in_turbidity + 0.210230f;
  D                      = -0.044050f * in_turbidity - 1.653694f;
  E                      = -0.010922f * in_turbidity + 0.052919f;
  float y                = (((1 + A * exp(B / cos_theta)) * (1 + C * exp(D * gamma) + E * cos_gamma * cos_gamma))
             / ((1 + A * exp(B / 1.0)) * (1 + C * exp(D * theta_sun) + E * cos_theta_sun * cos_theta_sun)));
  float local_saturation = 1.0;
  x                      = zenith_x * ((x * local_saturation) + (1.0 - local_saturation));
  y                      = zenith_y * ((y * local_saturation) + (1.0 - local_saturation));
  // convert chromaticities x and y to CIE
  xyz.x = (x / y) * xyz.y;
  xyz.z = ((1.0 - x - y) / y) * xyz.y;
  return xyz;
}


float sky_luminance(vec3 in_dir, vec3 in_sun_pos, float in_turbidity)
{
  float cos_gamma = dot(in_sun_pos, in_dir);
  if(cos_gamma < 0.0)
  {
    cos_gamma = 0.0;
  }
  if(cos_gamma > 1.0)
  {
    cos_gamma = 2.0 - cos_gamma;
  }
  float gamma         = acos(cos_gamma);
  float cos_theta     = in_dir.z;
  float cos_theta_sun = in_sun_pos.z;
  float theta_sun     = acos(cos_theta_sun);

  float A = 0.178721 * in_turbidity - 1.463037;
  float B = -0.355402 * in_turbidity + 0.427494;
  float C = -0.022669 * in_turbidity + 5.325056;
  float D = 0.120647 * in_turbidity - 2.577052;
  float E = -0.066967 * in_turbidity + 0.370275;

  float Y = (((1 + A * exp(B / cos_theta)) * (1 + C * exp(D * gamma) + E * cos_gamma * cos_gamma))
             / ((1 + A * exp(B / 1.0)) * (1 + C * exp(D * theta_sun) + E * cos_theta_sun * cos_theta_sun)));
  return Y;
}


vec3 calc_env_color(vec3 in_sun_dir, vec3 in_dir, float in_turbidity)
{
  // start with absolute value of zenith luminance in K cd/m2
  float theta_sun = acos(in_sun_dir.z);
  float chi       = (4.0 / 9.0 - in_turbidity / 120.0) * (M_PI - 2 * theta_sun);
  float luminance = 1000.0 * ((4.0453 * in_turbidity - 4.9710) * tan(chi) - 0.2155 * in_turbidity + 2.4192);
  luminance *= sky_luminance(in_dir, in_sun_dir, in_turbidity);
  // calculate the sky color - this uses 2 matrices (for 'x' and for 'y')
  vec3 XYZ = sky_color_xyz(in_dir, in_sun_dir, in_turbidity, luminance);
  // use result
  vec3 env_color = vec3(3.241 * XYZ.x - 1.537 * XYZ.y - 0.499 * XYZ.z, -0.969 * XYZ.x + 1.876 * XYZ.y + 0.042 * XYZ.z,
                        0.056 * XYZ.x - 0.204 * XYZ.y + 1.057 * XYZ.z);
  env_color *= M_PI;
  return env_color;
}

vec3 calc_irrad(vec3 in_data_sun_dir, float in_data_sun_dir_haze)
{
  vec3 colaccu        = vec3(0.0);
  vec3 nuState_normal = vec3(0.0, 0.0, 1.0);

  vec3 sun_dir = in_data_sun_dir;

  vec3 work = vec3(0.0);
  for(float u = 1. / 10.; u < 1.; u += 1. / 5.)
  {
    for(float v = 1. / 10.; v < 1.; v += 1. / 5.)
    {
      vec3 diff;
      diff = mi_reflection_dir_diffuse_x(nuState_normal, vec2(u, v));
      work = calc_env_color(sun_dir, diff, in_data_sun_dir_haze);
      colaccu += work;
    }
  }
  colaccu /= 25.0;
  return colaccu;
}


float tweak_saturation(float inout_saturation, float in_haze)
{
  float lowsat = pow(inout_saturation, 3.0);
  if(inout_saturation <= 1.0)
  {
    float local_haze = in_haze;
    local_haze -= 2.0;
    local_haze /= 15.0;
    if(local_haze < 0.0)
      local_haze = 0.0;
    if(local_haze > 1.0)
      local_haze = 1.0;
    local_haze = pow(local_haze, 3.0);
    return ((inout_saturation * (1.0 - local_haze)) + lowsat * local_haze);
  }
  return 1.;
}


vec3 arch_vectortweak(vec3 dir, int y_is_up, float horiz_height)
{
  vec3 out_dir = dir;
  if(y_is_up == 1)
  {
    out_dir = vec3(dir.x, dir.z, dir.y);
  }
  if(horiz_height != 0)
  {
    out_dir.z -= horiz_height;
    out_dir = normalize(out_dir);
  }
  return out_dir;
}


vec3 arch_colortweak(vec3 tint, float saturation, float redness)
{

  float intensity = luminance(tint);
  vec3  out_tint;
  // clamp down negatives (should never happen, but ...)
  if(saturation <= 0.0)
  {
    out_tint = vec3(intensity);
  }
  else
  {
    out_tint = tint * saturation + intensity * (1.0 - saturation);
    // boosted saturation can cause negatives
    if(saturation > 1.0)
    {
      vec3 rgb_color = vec3(tint);
      if(rgb_color.x < 0.0)
        rgb_color.x = 0.0;
      if(rgb_color.y < 0.0)
        rgb_color.y = 0.0;
      if(rgb_color.z < 0.0)
        rgb_color.z = 0.0;
      tint = rgb_color;
    }
  }
  // redness
  out_tint *= vec3(1.0 + redness, 1., 1.0 - redness);
  return out_tint;
}


vec2 calc_physical_scale(float sun_disk_scale, float sun_glow_intensity, float sun_disk_intensity)
{
  float sun_angular_radius = 0.00465f;

  /* This is the angular radius of the sun in radians, scaled according to the user's wishes
    and further scaled by 10 which is the radius of the glow */
  float sun_disk_radius = sun_angular_radius * sun_disk_scale;
  float sun_glow_radius = sun_disk_radius * 10.0f;

  /* The contribution of the sun disk & glow is ultimately driven by these expressions:
    miScalar factor = (1.0f - sun_angle / sun_radius) * 10.0f;
    factor = (miScalar) pow(factor / 10.0, 3.0) * 2.0f * glow_intensity +   // SUN GLOW
    smoothstep(8.5f, 9.5f + (haze / 50.0f), factor) * 100.0f * disk_int;     // SUN DISK
    color.r += data->sun_color.r * factor;
    color.g += data->sun_color.g * factor;
    color.b += data->sun_color.b * factor;

    Our goal is:
    a. the integration of factor==disk_int,
        such that we get a physically-scaled sun for disk_int=1
    b. the glow intensity is capped at 50% of the total
        (50% is an arbitrary number, we have to cap somewhere)
    ==> To achieve this goal, we simply calculate the integrals of
    the sun GLOW & DISK functions,
    calculate their ratio and scale them accordingly.
    */

  /* We calculate the integral of the glow intensity function */
  float glow_func_integral;
  {
    /* Calculate the integral of the glow function over its, i.e.:
        integral[x=0 to x=sun_glow_radius] (pow(factor / 10.0, 3.0) *
                2.0f * glow_intensity * sin(x) dx)
        With x being "sun_angle". */
    // flattened code:
    glow_func_integral = sun_glow_intensity
                         * ((4. * M_PI) - (24. * M_PI) / (sun_glow_radius * sun_glow_radius)
                            + (24. * M_PI) * sin(sun_glow_radius) / (sun_glow_radius * sun_glow_radius * sun_glow_radius));
  }

  /* Calculate the target sun disk intensity integral (the value towards which
    we must scale to attain a physically-scaled sun intensity */
  float target_sundisk_integral = sun_disk_intensity * M_PI;

  /* Subtract the glow integral from the target disk integral,
    limiting the glow power to 50% of the sun disk */
  float sky_sunglow_scale = 1.0;
  float max_glow_integral = 0.5f * target_sundisk_integral;
  if(glow_func_integral > max_glow_integral)
  {
    sky_sunglow_scale *= max_glow_integral / glow_func_integral;
    target_sundisk_integral -= max_glow_integral;
  }
  else
  {
    target_sundisk_integral -= glow_func_integral;
  }

  float sundisk_area             = 2 * M_PI * (1 - cos(sun_disk_radius));
  float target_sundisk_intensity = target_sundisk_integral / sundisk_area;

  /* Calculate the actual sun disk intensity, before scaling is applied */
  /* The integral of the sun disk intensity function should be taken into
    account, however the average value of the sun disk smoothing function
    is very close to 1 though probably not exactly 1), and I've so far failed
    at calculating the integral in a way where I can match the intensity of
    the mia_physicalsun shader. The results are actually closer if I assume
    the sun disk intensity has an average value of 1, and only deviate slightly
    with very large sun disk radii. I can't quite explain this, so I'll
    accept this "approximation" which apparently yields very acceptable
    results.
    So, TODO: re-calculate the integral for that sun smooth function! */
  float actual_sundisk_integral = 1.0f * sundisk_area;
  /* approximation! needs to be re-calculated from the integral of
    the function */
  float actual_sundisk_intensity = sun_disk_intensity * 100.0f * actual_sundisk_integral / sundisk_area; /* average value of the */

  /* Apply the proper scaling to get to the target value */
  return vec2((target_sundisk_intensity == 0.0) ? 0.0 : target_sundisk_intensity / actual_sundisk_intensity, sky_sunglow_scale);
}


float night_brightness_adjustment(vec3 sun_dir)
{
  float lmt = 0.30901699437494742410229341718282;
  if(sun_dir.z <= -lmt)
    return 0.0;
  float factor = (sun_dir.z + lmt) / lmt;
  factor *= factor;
  factor *= factor;
  return factor;
}


vec3 sun_and_sky(in SunAndSky ss, in vec3 in_direction)
{
  vec3 result = vec3(0.0);

  float factor       = 1.0;
  float night_factor = 1.0;
  vec3  out_color    = vec3(0.0);
  vec3  rgb_scale    = ss.rgb_unit_conversion;
  vec3  dir          = in_direction;
  float horiz_height = ss.horizon_height / 10.0;  // done in *_init
  dir                = arch_vectortweak(dir, ss.y_is_up, horiz_height);
  // haze
  float local_haze = 2.0 + ss.haze;
  if(local_haze < 2.0)
  {
    local_haze = 2.0;
  }
  float local_saturation = tweak_saturation(ss.saturation, local_haze);
  if(luminance(rgb_scale) < 0.0)
  {
    rgb_scale = vec3(1.0 / 80000.0);
  }
  rgb_scale *= ss.multiplier;
  if(ss.multiplier <= 0.0)
  {
    return vec3(0);
  }

  // downness
  float downness = dir.z;
  vec3  real_dir = dir;

  // only calc for above-the-horizon
  if(dir.z < 0.001)
  {
    dir.z = 0.001;
    dir   = normalize(dir);
  }

  // sun_dir
  vec3 sun_dir      = ss.sun_direction;
  sun_dir           = normalize(sun_dir);
  sun_dir           = arch_vectortweak(sun_dir, ss.y_is_up, horiz_height);
  vec3 real_sun_dir = sun_dir;
  if(sun_dir.z < 0.001)
  {
    if(sun_dir.z < 0.0)
    {
      factor = night_brightness_adjustment(sun_dir);
    }
    sun_dir.z = 0.001;
    sun_dir   = normalize(sun_dir);
  }

  vec3 tint;
  if(factor > 0.0)
  {
    tint = calc_env_color(sun_dir, dir, local_haze);
    if(factor < 1.0)
    {
      tint *= factor;
    }
  }
  else
  {
    tint = vec3(0.);
  }
  vec3 data_sun_color = calc_sun_color(sun_dir, downness > 0 ? local_haze : 2.0);
  if(ss.sun_disk_intensity > 0.0 && ss.sun_disk_scale > 0.0)
  {
    float sun_angle  = acos(dot(real_dir, real_sun_dir));
    float sun_radius = 0.00465 * ss.sun_disk_scale * 10.0;
    if(sun_angle < sun_radius)
    {
      /* Calculate the scales necessary to get a sun with physical intensity */
      /* default values */
      float sky_sundisk_scale = 1.0f;
      float sky_sunglow_scale = 1.0f;
      if(ss.physically_scaled_sun == 1)
      {
        vec2 return_value = calc_physical_scale(ss.sun_disk_scale, ss.sun_glow_intensity, ss.sun_disk_intensity);
        sky_sundisk_scale = return_value.x;
        sky_sunglow_scale = return_value.y;
      }

      float sun_factor = (1.0 - sun_angle / sun_radius) * 10.0;

      sun_factor = (pow(sun_factor / 10.0, 3.0) * 2.0 * ss.sun_glow_intensity * sky_sunglow_scale
                    + smoothstep(8.5, 9.5 + (local_haze / 50.0), sun_factor) * 100.0 * ss.sun_disk_intensity * sky_sundisk_scale);
      tint += data_sun_color * sun_factor;
    }
  }
  // set the output
  out_color = tint * rgb_scale;
  if(downness <= 0.0)
  {
    vec3 irrad     = vec3(0.0);
    vec3 downcolor = ss.ground_color;

    irrad = calc_irrad(sun_dir, 2.0);
    downcolor *= (irrad + data_sun_color * sun_dir.z) * rgb_scale;
    // apply 1+sun_dir.z night factor to downcolor
    // otherwise at sun_dir.z==-1 (midnight) we get a brightly
    // illuminated ground plane!
    if(factor < 1)
    {
      downcolor *= factor;
    }
    float hor_blur = ss.horizon_blur / 10.0;
    if(hor_blur > 0.0)
    {
      float dness = -downness;
      dness /= hor_blur;
      if(dness > 1.0)
      {
        dness = 1.0;
      }
      dness        = smoothstep(0.0, 1.0, dness);
      out_color    = out_color * (1.0 - dness) + downcolor * dness;
      night_factor = 1.0 - dness;
    }
    else
    {
      out_color    = downcolor;
      night_factor = 0.0;
    }
  }

  out_color = arch_colortweak(out_color, local_saturation, ss.redblueshift);
  result    = out_color;
  if(night_factor > 0.0)
  {
    vec3 night = ss.night_color;
    night *= night_factor;
    /*rgb code*/
    vec3 rgb_result = vec3(result);
    vec3 rgb_night  = vec3(night);
    if(rgb_result.x < rgb_night.x)
      rgb_result.x = rgb_night.x;
    if(rgb_result.y < rgb_night.y)
      rgb_result.y = rgb_night.y;
    if(rgb_result.z < rgb_night.z)
      rgb_result.z = rgb_night.z;
    result = vec3(rgb_result);
  }
  result *= M_PI;

  return result;
}

#endif  // CPP
#endif  // SUN_AND_SKY_GLSL

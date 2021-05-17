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


// This file can compress normal or tangent to a single uint
// all oct functions derived from "A Survey of Efficient Representations for Independent Unit Vectors"
// http://jcgt.org/published/0003/02/01/paper.pdf


#ifndef COMPRESS_GLSL
#define COMPRESS_GLSL


#ifdef __cplusplus
#define INLINE inline

INLINE float uintBitsToFloat(uint32_t const& v)
{
  union
  {
    uint  in;
    float out;
  } u;

  u.in = v;

  return u.out;
};

INLINE uint32_t floatBitsToUint(float v)
{
  union
  {
    float in;
    uint  out;
  } u;

  u.in = v;

  return u.out;
};

INLINE uint packUnorm4x8(vec4 const& v)
{
  union
  {
    unsigned char in[4];
    uint          out;
  } u;

  u.in[0] = (unsigned char)std::round(std::min(std::max(v.x, 0.0f), 1.0f) * 255.f);
  u.in[1] = (unsigned char)std::round(std::min(std::max(v.y, 0.0f), 1.0f) * 255.f);
  u.in[2] = (unsigned char)std::round(std::min(std::max(v.z, 0.0f), 1.0f) * 255.f);
  u.in[3] = (unsigned char)std::round(std::min(std::max(v.w, 0.0f), 1.0f) * 255.f);

  return u.out;
}

INLINE float roundEven(float x)
{
  int   Integer        = static_cast<int>(x);
  float IntegerPart    = static_cast<float>(Integer);
  float FractionalPart = (x - floor(x));

  if(FractionalPart > 0.5f || FractionalPart < 0.5f)
  {
    return std::round(x);
  }
  else if((Integer % 2) == 0)
  {
    return IntegerPart;
  }
  else if(x <= 0)  // Work around...
  {
    return IntegerPart - 1;
  }
  else
  {
    return IntegerPart + 1;
  }
}

#else
#define INLINE
#endif


//-----------------------------------------------------------------------
// Compression - can be done on host or device
//-----------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////
#define C_Stack_Max 3.402823466e+38f
INLINE uint compress_unit_vec(vec3 nv)
{
  // map to octahedron and then flatten to 2D (see 'Octahedron Environment Maps' by Engelhardt & Dachsbacher)
  if((nv.x < C_Stack_Max) && !isinf(nv.x))
  {
    const float d = 32767.0f / (abs(nv.x) + abs(nv.y) + abs(nv.z));
    int         x = int(roundEven(nv.x * d));
    int         y = int(roundEven(nv.y * d));

    if(nv.z < 0.0f)
    {
      const int maskx = x >> 31;
      const int masky = y >> 31;
      const int tmp   = 32767 + maskx + masky;
      const int tmpx  = x;
      x               = (tmp - (y ^ masky)) ^ maskx;
      y               = (tmp - (tmpx ^ maskx)) ^ masky;
    }

    uint packed = (uint(y + 32767) << 16) | uint(x + 32767);
    if(packed == ~0u)
      return ~0x1u;
    return packed;
  }
  else
  {
    return ~0u;
  }
}


///
float short_to_floatm11(const int v)  // linearly maps a short 32767-32768 to a float -1-+1 //!! opt.?
{
  return (v >= 0) ? (uintBitsToFloat(0x3F800000u | (uint(v) << 8)) - 1.0f) :
                    (uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-v) << 8)) + 1.0f);
}

vec3 decompress_unit_vec(uint packed)
{
  if(packed != ~0u)  // sanity check, not needed as isvalid_unit_vec is called earlier
  {
    int x = int(packed & 0xFFFFu) - 32767;
    int y = int(packed >> 16) - 32767;

    const int maskx = x >> 31;
    const int masky = y >> 31;
    const int tmp0  = 32767 + maskx + masky;
    const int ymask = y ^ masky;
    const int tmp1  = tmp0 - (x ^ maskx);
    const int z     = tmp1 - ymask;
    float     zf;
    if(z < 0)
    {
      x  = (tmp0 - ymask) ^ maskx;
      y  = tmp1 ^ masky;
      zf = uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-z) << 8)) + 1.0f;
    }
    else
    {
      zf = uintBitsToFloat(0x3F800000u | (uint(z) << 8)) - 1.0f;
    }

    return normalize(vec3(short_to_floatm11(x), short_to_floatm11(y), zf));
  }
  else
  {
    return vec3(C_Stack_Max);
  }
}


#endif  // COMPRESS_GLSL

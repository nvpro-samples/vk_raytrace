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


#ifndef RANDOM_GLSL
#define RANDOM_GLSL 1

#define RAND_PCG 1
#define RAND_LCG 2
#define RAD_RADINV 3

#define RAND_METHOD RAND_LCG


//-----------------------------------------------------------------------
// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
//-----------------------------------------------------------------------
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

//-----------------------------------------------------------------------
// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
//-----------------------------------------------------------------------
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

//-----------------------------------------------------------------------
// https://www.pcg-random.org/
//-----------------------------------------------------------------------
uint pcg(inout uint state)
{
  uint prev = state * 747796405u + 2891336453u;
  uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
  state     = prev;
  return (word >> 22u) ^ word;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
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

uvec3 pcg3d(uvec3 v)
{
  v = v * 1664525u + uvec3(1013904223u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> uvec3(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}


//-----------------------------------------------------------------------
const float pcg_div = (1.0 / float(0xffffffffu));  // 4,294,967,295 max uint32
// Generate a random float in [0, 1) given the previous RNG state
//-----------------------------------------------------------------------
float rnd(inout uint seed)
{
#if(RAND_METHOD == RAND_PCG)
  seed = pcg(seed);
  return float(seed) * pcg_div;
#endif

#if(RAND_METHOD == RAND_LCG)
  return (float(lcg(seed)) / float(0x01000000u));
#endif
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec2 rnd2(inout uint prev)
{
#if(RAND_METHOD == RAND_PCG)
  return vec2(rnd(prev), rnd(prev));
#endif

#if(RAND_METHOD == RAND_LCG)
  return vec2(float(lcg(prev)) / float(0x01000000u), float(lcg(prev)) / float(0x01000000u));
#endif
}

#endif  // RANDOM_GLSL

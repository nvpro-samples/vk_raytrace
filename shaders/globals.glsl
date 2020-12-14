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

#ifndef GLOBALS_GLSL
#define GLOBALS_GLSL 1


// Math defines
const highp float M_PI   = 3.14159265358979323846;   // pi
const highp float M_PI_2 = 1.57079632679489661923;   // pi/2
const highp float M_PI_4 = 0.785398163397448309616;  // pi/4
const highp float M_1_PI = 0.318309886183790671538;  // 1/pi
const highp float M_2_PI = 0.636619772367581343076;  // 2/pi


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


#endif  // GLOBALS_GLSL

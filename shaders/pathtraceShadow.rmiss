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


#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "globals.glsl"

//-------------------------------------------------------------------------------------------------
// This will be executed when sending shadow rays and missing all geometries
// - There are no hit shader for the shadow ray, therefore
// - Before calling Trace, set isShadowed=true
// - The default anyhit, closesthit won't change isShadowed, but if nothing is hit, it will be
//   set to false.
//-------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------
// Miss shader for the shadow rayPayloadInEXT
//

layout(location = 1) rayPayloadInEXT ShadowHitPayload payload;

void main()
{
  payload.isHit = false;
}

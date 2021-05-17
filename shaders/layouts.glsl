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


#ifndef LAYOUTS_GLSL
#define LAYOUTS_GLSL 1


// C++ shared structures and binding
#include "../binding.h"
#include "../structures.h"


// Including structs used by the layouts
//#include "globals.glsl"


// Sun & Sky structure
#include "sun_and_sky.h"

//----------------------------------------------
// Descriptor Set Layout
//----------------------------------------------


// clang-format off
layout(set = S_ACCEL, binding = B_TLAS)						uniform accelerationStructureEXT topLevelAS;
//
layout(set = S_OUT,   binding = B_STORE)					uniform image2D			resultImage;
//
layout(set = S_SCENE, binding = B_INSTDATA            )     buffer _InstanceInfo	{ InstanceData geoInfo[]; };
layout(set = S_SCENE, binding = B_CAMERA,		scalar)		uniform _SceneCamera	{ SceneCamera sceneCamera; };
layout(set = S_SCENE, binding = B_VERTEX,		scalar)		buffer _VertexBuf		{ VertexAttributes v[]; } vertex[];
layout(set = S_SCENE, binding = B_INDICES,		scalar)		buffer _Indices			{ uvec3 i[];            } indices[];

layout(set = S_SCENE, binding = B_MATERIALS,	scalar)		buffer _MaterialBuffer	{ GltfShadeMaterial materials[]; };
layout(set = S_SCENE, binding = B_LIGHTS,		scalar)		buffer _Lights			{ Light lights[]; };
layout(set = S_SCENE, binding = B_TEXTURES			  )		uniform sampler2D		texturesMap[]; 
//
layout(set = S_ENV, binding = B_SUNANDSKY,		scalar)		uniform _SSBuffer		{ SunAndSky _sunAndSky; };
layout(set = S_ENV, binding = B_HDR)						uniform sampler2D		environmentTexture;
layout(set = S_ENV, binding = B_IMPORT_SMPL)				uniform sampler2D		environmentSamplingData;
// clang-format on


#endif  // LAYOUTS_GLSL

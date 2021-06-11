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


// Binding sets for the scene

// Sets
#define S_ACCEL 0  // Acceleration structure
#define S_OUT 1    // Offscreen output image
#define S_SCENE 2  // Scene data
#define S_ENV 3    // Environment / Sun & Sky
#define S_WF 4     // Wavefront extra data

// Acceleration Structure - Set 0
#define B_TLAS 0

// Output image - Set 1
#define B_SAMPLER 0  // As sampler
#define B_STORE 1    // As storage

// Scene Data - Set 2
#define B_CAMERA 0
#define B_MATERIALS 1
#define B_INSTDATA 2
#define B_LIGHTS 3
#define B_TEXTURES 4  // must be last elem

// Environment - Set 3
#define B_SUNANDSKY 0
#define B_HDR 1
#define B_IMPORT_SMPL 2

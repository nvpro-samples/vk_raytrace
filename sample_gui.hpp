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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nvvk/profiler_vk.hpp"


//--------------------------------------------------------------------------------------------------
// This implements all graphical user interface of SampleExample.
class SampleExample; // Forward declaration

class SampleGUI
{
public:
  SampleGUI(SampleExample* _s)
      : _se(_s)
  {
  }
  void render(nvvk::ProfilerVK& profiler);
  void titleBar();
  void menuBar();
  void showBusyWindow();

private:
  bool           guiCamera();
  bool           guiRayTracing();
  bool           guiTonemapper();
  bool           guiEnvironment();
  bool           guiStatistics();
  bool           guiProfiler(nvvk::ProfilerVK& profiler);
  bool           guiGpuMeasures();

  SampleExample* _se{nullptr};
};


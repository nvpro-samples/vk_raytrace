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


#pragma once

#ifndef TOOLS_H
#define TOOLS_H

// Utility to time the execution of something resetting the timer
// on each elapse call
// Usage:
// {
//   MilliTimer timer;
//   ... stuff ...
//   double time_elapse = timer.elapse();
// }
#include <chrono>

struct MilliTimer
{
  MilliTimer() { reset(); }
  void   reset() { startTime = std::chrono::high_resolution_clock::now(); }
  double elapse()
  {
    auto now  = std::chrono::high_resolution_clock::now();
    auto t    = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count() / 1000.0;
    startTime = now;
    return t;
  }
  void print() { LOGI(" --> (%5.3f ms)\n", elapse()); }

  std::chrono::high_resolution_clock::time_point startTime;
};


// Formating with local number representation
template <class T>
std::string FormatNumbers(T value)
{
  std::stringstream ss;
  ss.imbue(std::locale(""));
  ss << std::fixed << value;
  return ss.str();
}

#endif

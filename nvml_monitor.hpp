/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <iostream>
#include <nvml.h>
#include <string>
#include <vector>

#ifdef _WIN32
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#endif

/** 

Capture the GPU load and memory for all GPUs on the system.

Usage:
- There should be only one instance of NvmlMonitor
- call refresh() in each frame. It will not pull more measurement that the interval(ms)
- isValid() : return if it can be used
- nbGpu()   : return the number of GPU in the computer
- getMeasures() :  return the measurements for a GPU
- getInfo()     : return the info about the GPU

Measurements: 
- Uses a cycle buffer. 
- Offset is the last measurement
- Cycling is load[(offset + i) % load.size()]
- Values are in KB, not Bytes

*/


class NvmlMonitor
{
public:
  struct Measure
  {
    std::vector<float> memory;  // Memory measurement in KB
    std::vector<float> load;    // Load measurement [0, 100]
  };

  struct GpuInfo
  {
    uint32_t    max_mem;       // Max memory for each GPU
    uint32_t    driver_model;  // Driver model: WDDM/TCC
    std::string name;
  };

  struct SysInfo
  {
    std::vector<float> cpu;  // Load measurement [0, 100]
    char               driverVersion[80];
  };

  NvmlMonitor(uint32_t interval = 100, uint32_t limit = 100)
      : m_limit(limit)        // limit : number of measures
      , m_interval(interval)  // interval : ms between sampling
  {
#ifdef _WIN32
    auto    nvml_dll = R"(C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll)";
    HMODULE m;  //= LoadLibraryA(nvml_dll);
    m = (HMODULE)LoadNmvlLibrary();
    if(!m)
    {
      std::cerr << "Could not load NVML library - Please reinstall the NVIDIA Driver \n"
                   "Default location: "
                << nvml_dll;
      return;
    }

    nvmlReturn_t result;
    result = nvmlInit();
    if(result != NVML_SUCCESS)
      return;
    if(nvmlDeviceGetCount(&m_physicalGpuCount) != NVML_SUCCESS)
      return;
    m_measure.resize(m_physicalGpuCount);
    m_info.resize(m_physicalGpuCount);

    // System Info
    m_sysInfo.cpu.resize(m_limit);

    // Get driver version
    result = nvmlSystemGetDriverVersion(m_sysInfo.driverVersion, 80);

    // Loop over all GPUs
    for(int i = 0; i < (int)m_physicalGpuCount; i++)
    {
      // Sizing the data
      m_measure[i].memory.resize(m_limit);
      m_measure[i].load.resize(m_limit);

      // Retrieving general capabilities
      nvmlDevice_t      device;
      nvmlMemory_t      memory;
      nvmlDriverModel_t driver_model;
      nvmlDriverModel_t pdriver_model;

      // Find the memory of each cards
      result = nvmlDeviceGetHandleByIndex(i, &device);
      if(NVML_SUCCESS != result)
        return;
      nvmlDeviceGetMemoryInfo(device, &memory);
      m_info[i].max_mem = (uint32_t)(memory.total / (uint64_t)(1024));  // Convert to KB

      // name
      char name[80];
      result = nvmlDeviceGetName(device, name, 80);
      if(NVML_SUCCESS == result)
        m_info[i].name = name;

      // Find the model: TCC or WDDM
      result = nvmlDeviceGetDriverModel(device, &driver_model, &pdriver_model);
      if(NVML_SUCCESS != result)
        return;
      m_info[i].driver_model = driver_model;
    }
    startTime = std::chrono::high_resolution_clock::now();
    m_valid   = true;
#endif
  }
  ~NvmlMonitor() { nvmlShutdown(); }

  //--------------------------------------------------------------------------------------------------
  // Take measurement at each 'interval'
  //
  void refresh()
  {
    if(!m_valid)
      return;

    // Avoid refreshing too often
    auto now = std::chrono::high_resolution_clock::now();
    auto t   = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
    if(t < m_interval)
      return;
    startTime = now;

    // Increasing where to store the value
    m_offset = (m_offset + 1) % m_limit;

    // System
    m_sysInfo.cpu[m_offset] = getCpuLoad();

    // All GPUs
    for(unsigned int gpu_id = 0; gpu_id < m_physicalGpuCount; gpu_id++)
    {
      nvmlDevice_t device;
      nvmlDeviceGetHandleByIndex(gpu_id, &device);
      m_measure[gpu_id].memory[m_offset] = getMemory(device);
      m_measure[gpu_id].load[m_offset]   = getLoad(device);
    }
  }

  bool           isValid() { return m_valid; }
  uint32_t       nbGpu() { return m_physicalGpuCount; }
  const Measure& getMeasures(int gpu) { return m_measure[gpu]; }
  const GpuInfo& getInfo(int gpu) { return m_info[gpu]; }
  const SysInfo& getSysInfo() { return m_sysInfo; }
  int            getOffset() { return m_offset; }

private:
  float getMemory(nvmlDevice_t device)
  {
    nvmlMemory_t memory;
    nvmlDeviceGetMemoryInfo(device, &memory);
    return static_cast<float>(memory.used / (uint64_t)(1024));  // Convert to KB
  }

  float getLoad(nvmlDevice_t device)
  {
    nvmlUtilization_t utilization;
    nvmlDeviceGetUtilizationRates(device, &utilization);
    return static_cast<float>(utilization.gpu);
  }

  float getCpuLoad()
  {
  #ifdef _WIN32
    static uint64_t _previousTotalTicks = 0;
    static uint64_t _previousIdleTicks  = 0;

    FILETIME idleTime, kernelTime, userTime;
    GetSystemTimes(&idleTime, &kernelTime, &userTime);

    auto FileTimeToInt64 = [](const FILETIME& ft) {
      return (((uint64_t)(ft.dwHighDateTime)) << 32) | ((uint64_t)ft.dwLowDateTime);
    };

    auto totalTicks = FileTimeToInt64(kernelTime) + FileTimeToInt64(userTime);
    auto idleTicks  = FileTimeToInt64(idleTime);

    uint64_t totalTicksSinceLastTime = totalTicks - _previousTotalTicks;
    uint64_t idleTicksSinceLastTime  = idleTicks - _previousIdleTicks;

    float result = 1.0f - ((totalTicksSinceLastTime > 0) ? ((float)idleTicksSinceLastTime) / totalTicksSinceLastTime : 0);

    _previousTotalTicks = totalTicks;
    _previousIdleTicks  = idleTicks;

    return result * 100.f;
#else
    return 0;
#endif
  }

#ifdef _WIN32
  void* LoadNmvlLibrary()
  {
    const char* nvmlDllName = "nvml.dll";
    void*       handle      = nullptr;

    // We are going to look for the OpenGL driver which lives  next to nvoptix.dll and nvml.dll.
    // 0 (null) will be returned if any errors occurred.

    static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";  // Display Adapters
    const ULONG        flags                         = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    ULONG              deviceListSize                = 0;

    if(CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
    {
      return nullptr;
    }

    char* deviceNames = (char*)malloc(deviceListSize);

    if(CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
    {
      free(deviceNames);
      return nullptr;
    }

    DEVINST devID = 0;

    // Continue to the next device if errors are encountered.
    for(char* deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
    {
      if(CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
      {
        continue;
      }

      HKEY regKey = 0;
      if(CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
      {
        continue;
      }

      const char* valueName = "OpenGLDriverName";
      DWORD       valueSize = 0;

      LSTATUS ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
      if(ret != ERROR_SUCCESS)
      {
        RegCloseKey(regKey);
        continue;
      }

      char* regValue = (char*)malloc(valueSize);
      ret            = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE)regValue, &valueSize);
      if(ret != ERROR_SUCCESS)
      {
        free(regValue);
        RegCloseKey(regKey);
        continue;
      }

      // Strip the OpenGL driver dll name from the string then create a new string with
      // the path and the nvoptix.dll name
      for(int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
      {
        regValue[i] = '\0';
      }

      size_t newPathSize = strlen(regValue) + strlen(nvmlDllName) + 1;
      char*  dllPath     = (char*)malloc(newPathSize);
      strcpy(dllPath, regValue);
      strcat(dllPath, nvmlDllName);

      free(regValue);
      RegCloseKey(regKey);

      handle = LoadLibraryA((LPCSTR)dllPath);
      free(dllPath);

      if(handle)
      {
        free(deviceNames);
        return handle;
      }
    }

    // Did not find it, looking in the system directory
    free(deviceNames);

    // Get the size of the path first, then allocate
    unsigned int size = GetSystemDirectoryA(NULL, 0);
    if(size == 0)
    {
      // Couldn't get the system path size, so bail
      return nullptr;
    }

    // DAR DEBUG Which Windows versions have the DLL inside the Windows system directory (C:\Windows\System32)?
    // Under Windows 10 it's in the DriverStore.
    size_t pathSize   = size + 1 + strlen(nvmlDllName);
    char*  systemPath = (char*)malloc(pathSize);

    if(GetSystemDirectoryA(systemPath, size) != size - 1)
    {
      // Something went wrong
      free(systemPath);
      return nullptr;
    }

    strcat(systemPath, "\\");
    strcat(systemPath, nvmlDllName);

    handle = LoadLibraryA(systemPath);

    free(systemPath);

    if(handle)
    {
      return handle;
    }

    return handle;
  }
#endif

  bool                 m_valid{false};
  uint32_t             m_physicalGpuCount{0};
  std::vector<Measure> m_measure;
  std::vector<GpuInfo> m_info;  // Max memory for each GPU
  SysInfo              m_sysInfo;
  uint32_t             m_offset{0};

  uint32_t m_limit{100};
  uint32_t m_interval{100};  // ms

  std::chrono::high_resolution_clock::time_point startTime;
};

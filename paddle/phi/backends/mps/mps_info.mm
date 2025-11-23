/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/mps/mps_info.h"

#ifdef PADDLE_WITH_MPS

#include <cstdlib>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "paddle/common/exception.h"
#include "paddle/phi/core/enforce.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_MAC
#import <Metal/Metal.h>
#endif
#endif

namespace phi {
namespace backends {
namespace mps {

static int g_current_device_id = 0;
static bool g_mps_available = false;
static bool g_mps_initialized = false;

void InitMPSAvailability() {
  if (g_mps_initialized) {
    return;
  }
  
#ifdef __APPLE__
#if TARGET_OS_MAC
  // Step 1: Check macOS version (MPS requires macOS 12.3+)
  // Following PyTorch's approach: MPS backend only works on macOS 12.3+
  if (@available(macOS 12.3, *)) {
    // macOS version is sufficient, proceed with device check
  } else {
    g_mps_available = false;
    g_mps_initialized = true;
    return;
  }
  
  // Step 2: Check that a Metal GPU device exists and supports MPS
  // Following PyTorch's approach: try to create a Metal device and verify it's usable
  @autoreleasepool {
    // Try to get the default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device != nil) {
      // Verify the device is actually usable by trying to create a command queue
      // This ensures Metal is properly initialized and not just returning a nil device
      id<MTLCommandQueue> testQueue = [device newCommandQueue];
      if (testQueue != nil) {
        g_mps_available = true;
        VLOG(1) << "MPS device found and verified: " << [[device name] UTF8String];
        // ARC will automatically release testQueue when it goes out of scope
      } else {
        VLOG(1) << "MPS device found but cannot create command queue";
        g_mps_available = false;
      }
    } else {
      // If default device is nil, try to enumerate all devices as fallback
      // MTLCopyAllDevices is available on macOS 10.11+
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      if (devices != nil && [devices count] > 0) {
        // Verify at least one device is usable
        bool found_usable = false;
        for (id<MTLDevice> dev in devices) {
          id<MTLCommandQueue> testQueue = [dev newCommandQueue];
          if (testQueue != nil) {
              found_usable = true;
              g_mps_available = true;
              VLOG(1) << "MPS devices found via enumeration: " << [devices count]
                      << ", usable device: " << [[dev name] UTF8String];
              // ARC will automatically release testQueue when it goes out of scope
              break;
          }
        }
        if (!found_usable) {
          g_mps_available = false;
          VLOG(1) << "MPS devices found but none are usable";
        }
        // ARC will automatically release devices when it goes out of scope
        // MTLCopyAllDevices returns an autoreleased object in ARC mode
      } else {
        g_mps_available = false;
        VLOG(1) << "No MPS devices found via enumeration";
      }
    }
  }
#else
  g_mps_available = false;
  VLOG(1) << "Not on macOS, MPS not available";
#endif
#else
  g_mps_available = false;
  VLOG(1) << "Not on Apple platform, MPS not available";
#endif
  
  // Step 3: Cache the result (following PyTorch's approach)
  g_mps_initialized = true;
  VLOG(1) << "MPS availability initialized: " << (g_mps_available ? "available" : "not available");
}

bool IsMPSAvailable() {
  InitMPSAvailability();
  return g_mps_available;
}

int GetMPSDeviceCount() {
  if (!IsMPSAvailable()) {
    return 0;
  }
  // MPS typically has one device (the Apple Silicon GPU)
  return 1;
}

int GetCurrentDeviceId() {
  if (!IsMPSAvailable()) {
    return -1;
  }
  return g_current_device_id;
}

void SetDeviceId(int device_id) {
  if (!IsMPSAvailable()) {
    PADDLE_THROW(common::errors::Unavailable(
        "MPS is not available on this system."));
  }
  if (device_id < 0 || device_id >= GetMPSDeviceCount()) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Invalid MPS device id: %d. Available device count: %d",
        device_id,
        GetMPSDeviceCount()));
  }
  g_current_device_id = device_id;
}

size_t MPSAvailableMemToAlloc() {
  if (!IsMPSAvailable()) {
    return 0;
  }
  // MPS uses unified memory, so we return a large value
  // Actual memory management is handled by the system
  return 1024ULL * 1024 * 1024 * 1024;  // 1TB as placeholder
}

size_t MPSMinChunkSize() {
  return 256;  // 256 bytes minimum chunk
}

size_t MPSMaxChunkSize() {
  return 512 * 1024 * 1024;  // 512MB maximum chunk
}

size_t MPSInitAllocSize() {
  return 64 * 1024 * 1024;  // 64MB initial allocation
}

size_t MPSReallocSize() {
  return 32 * 1024 * 1024;  // 32MB reallocation increment
}

size_t MPSMaxAllocSize() {
  return 1024 * 1024 * 1024;  // 1GB maximum allocation
}

size_t MPSExtraPaddingSize() {
  return 0;  // No extra padding needed for MPS
}

std::vector<int> GetSelectedDevices() {
  std::vector<int> devices;
  if (!IsMPSAvailable()) {
    return devices;
  }
  
  const char* visible_devices = std::getenv("MPS_VISIBLE_DEVICES");
  if (visible_devices == nullptr || std::string(visible_devices) == "all") {
    for (int i = 0; i < GetMPSDeviceCount(); ++i) {
      devices.push_back(i);
    }
  } else {
    std::string devices_str(visible_devices);
    size_t pos = 0;
    while ((pos = devices_str.find(',')) != std::string::npos) {
      int device_id = std::stoi(devices_str.substr(0, pos));
      devices.push_back(device_id);
      devices_str.erase(0, pos + 1);
    }
    if (!devices_str.empty()) {
      int device_id = std::stoi(devices_str);
      devices.push_back(device_id);
    }
  }
  return devices;
}

}  // namespace mps
}  // namespace backends
}  // namespace phi

#endif  // PADDLE_WITH_MPS


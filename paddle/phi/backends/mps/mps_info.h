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

#pragma once

#ifdef PADDLE_WITH_MPS

#include <stddef.h>
#include <string>
#include <vector>

#include "paddle/common/macros.h"

namespace phi {
namespace backends {
namespace mps {

//! Get the total number of MPS devices in system.
PADDLE_API int GetMPSDeviceCount();

//! Get the current MPS device id in system.
PADDLE_API int GetCurrentDeviceId();

//! Set the MPS device id for next execution.
PADDLE_API void SetDeviceId(int device_id);

//! Get the available memory to allocate on MPS device.
size_t MPSAvailableMemToAlloc();

//! Get the minimum chunk size for MPS buddy allocator.
size_t MPSMinChunkSize();

//! Get the maximum chunk size for MPS buddy allocator.
size_t MPSMaxChunkSize();

//! Get the initial allocation size for MPS buddy allocator.
size_t MPSInitAllocSize();

//! Get the reallocation size for MPS buddy allocator.
size_t MPSReallocSize();

//! Get the maximum allocation size for MPS buddy allocator.
size_t MPSMaxAllocSize();

//! Get the extra padding size for MPS buddy allocator.
size_t MPSExtraPaddingSize();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices();

//! Check if MPS is available on the system.
PADDLE_API bool IsMPSAvailable();

}  // namespace mps
}  // namespace backends
}  // namespace phi

#endif  // PADDLE_WITH_MPS


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

// Import Metal headers FIRST before any C++ headers to avoid BOOL redefinition
#ifdef __APPLE__
#if TARGET_OS_MAC
#import <Metal/Metal.h>
#endif
#endif

#ifdef PADDLE_WITH_MPS

#include <cstring>

#include "glog/logging.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/backends/mps/mps_copy.h"

namespace phi {
namespace backends {
namespace mps {

// MPS copy implementation following PyTorch's approach.
// 
// PyTorch's mps_copy_ (in aten/src/ATen/native/mps/operations/Copy.mm) uses
// Metal's blit command encoder for CPU->MPS and MPS->MPS copies. However, at
// PyTorch's level, they have access to tensor storage objects that contain
// MTLBuffer references.
//
// At PaddlePaddle's low-level memcpy interface, we only have raw pointers
// (void*), not MTLBuffer objects. Since MPS uses unified memory
// (MTLResourceStorageModeShared), memcpy is efficient and correct:
// - Both CPU and GPU can access the same memory directly
// - No explicit synchronization needed for unified memory
// - PyTorch also uses memcpy for unified memory scenarios
//
// Future optimization: If we can track pointer->MTLBuffer mappings, we could
// use Metal blit for better performance on large copies.
void MpsCopy(void* dst,
             const void* src,
             size_t num,
             const MPSPlace& dst_place,
             const Place& src_place,
             void* stream) {
  if (num == 0) return;
  
  VLOG(4) << "MpsCopy " << num << " Bytes from " << src_place << " to " << dst_place;
  
  // Use memcpy for unified memory (matches PyTorch's behavior for unified memory)
  std::memcpy(dst, src, num);
}

}  // namespace mps
}  // namespace backends
}  // namespace phi

#endif  // PADDLE_WITH_MPS


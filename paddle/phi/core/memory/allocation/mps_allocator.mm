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

#include "paddle/phi/core/memory/allocation/mps_allocator.h"

#ifdef PADDLE_WITH_MPS

#import <Metal/Metal.h>

#include <mutex>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

COMMON_DECLARE_bool(init_allocated_mem);

namespace paddle {
namespace memory {
namespace allocation {

// MPSAllocation holds a Metal buffer reference to keep it alive
class MPSAllocation : public Allocation {
 public:
  MPSAllocation(void* ptr, size_t size, phi::Place place, id<MTLBuffer> buffer)
      : Allocation(ptr, size, place), buffer_(buffer) {
    // Retain the Metal buffer to keep it alive
    if (buffer_) {
      [buffer_ retain];
    }
  }

  ~MPSAllocation() {
    // Release the Metal buffer when allocation is destroyed
    if (buffer_) {
      [buffer_ release];
      buffer_ = nil;
    }
  }

  id<MTLBuffer> buffer() const { return buffer_; }

 private:
  id<MTLBuffer> buffer_;  // Metal buffer reference
};

// Similar to PyTorch's MPS allocator implementation
phi::Allocation* MPSAllocator::AllocateImpl(size_t size) {
  VLOG(10) << "Allocate " << size << " bytes on " << phi::Place(place_);

  @autoreleasepool {
    // Get the default Metal device (MPS backend)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      PADDLE_THROW(common::errors::Unavailable(
          "Failed to get MPS device for allocation. MPS is only available on Apple Silicon."));
    }

    // Create a Metal buffer with shared storage mode for unified memory
    // MTLResourceStorageModeShared allows both CPU and GPU access
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
    
    if (buffer == nil) {
    PADDLE_THROW(common::errors::ResourceExhausted(
        "Failed to allocate %zu bytes on MPS device.", size));
  }

    void* ptr = [buffer contents];

  if (FLAGS_init_allocated_mem) {
    memset(ptr, 0xEF, size);
  }

    VLOG(10) << "  pointer=" << ptr << ", buffer=" << buffer;
  
    // Create MPSAllocation that holds the Metal buffer reference
    // The buffer will be released when the allocation is destroyed
    return new MPSAllocation(ptr, size, place_, buffer);
  }
}

void MPSAllocator::FreeImpl(phi::Allocation* allocation) {
  VLOG(10) << "Free pointer=" << allocation->ptr() << " on " << allocation->place();
  
  // MPSAllocation destructor will automatically release the Metal buffer
  delete allocation;
}

uint64_t MPSAllocator::ReleaseImpl(const phi::Place& place) {
  // MPS uses unified memory managed by the system
  // We don't have explicit release mechanism like GPU
  return 0;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif  // PADDLE_WITH_MPS


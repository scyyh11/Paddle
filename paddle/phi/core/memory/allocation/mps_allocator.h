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

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

// Forward declarations - Metal types are only used in .mm implementation
// This allows the header to be included from C++ files without requiring Objective-C++
#ifdef __APPLE__
#if TARGET_OS_MAC
// Metal types are forward declared in the .mm file
#endif
#endif

namespace paddle {
namespace memory {
namespace allocation {

// MPS allocator for Apple Silicon using Metal Performance Shaders
// Similar to PyTorch's MPS allocator implementation
// Uses Metal buffers with unified memory (MTLResourceStorageModeShared)
class PADDLE_API MPSAllocator : public Allocator {
 public:
  explicit MPSAllocator(const phi::MPSPlace& place) : place_(place) {}

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation* allocation) override;
  uint64_t ReleaseImpl(const phi::Place& place) override;

 private:
  phi::MPSPlace place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif  // PADDLE_WITH_MPS


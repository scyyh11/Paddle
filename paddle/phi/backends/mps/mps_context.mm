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

#include "paddle/phi/backends/mps/mps_context.h"

#ifdef PADDLE_WITH_MPS

#include <memory>
#include <mutex>

#include "glog/logging.h"
#include "paddle/common/exception.h"
#include "paddle/phi/backends/mps/mps_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

#ifdef __APPLE__
#if TARGET_OS_MAC
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif
#endif

namespace phi {

#ifdef __APPLE__
#if TARGET_OS_MAC

struct MPSContext::Impl {
  Impl() : place_(MPSPlace()), device_(nil), command_queue_(nil), owned_(false) {}

  explicit Impl(const Place& place) : place_(place), device_(nil), command_queue_(nil), owned_(false) {}

  ~Impl() {
    if (owned_) {
      if (command_queue_) {
        // Command queue is automatically released when device is released
        command_queue_ = nil;
      }
      if (device_) {
        device_ = nil;
      }
    }
  }

  void Init() {
    @autoreleasepool {
      if (device_ == nil) {
        device_ = MTLCreateSystemDefaultDevice();
        PADDLE_ENFORCE_NOT_NULL(
            device_,
            common::errors::Unavailable("Failed to create MPS device."));
      }
      
      if (command_queue_ == nil) {
        command_queue_ = [device_ newCommandQueue];
        PADDLE_ENFORCE_NOT_NULL(
            command_queue_,
            common::errors::Unavailable("Failed to create MPS command queue."));
      }
      
      owned_ = true;
      VLOG(4) << "MPSContext initialized with device: " << [[device_ name] UTF8String];
    }
  }

  Place place_;
  id<MTLDevice> device_;
  id<MTLCommandQueue> command_queue_;
  bool owned_;
};

MPSContext::MPSContext(const MPSPlace& place)
    : DeviceContext(), impl_(std::make_unique<MPSContext::Impl>(place)) {
  impl_->Init();
}

MPSContext::MPSContext(MPSContext&&) = default;

MPSContext& MPSContext::operator=(MPSContext&&) = default;

MPSContext::~MPSContext() = default;

const Place& MPSContext::GetPlace() const {
  return impl_->place_;
}

void* MPSContext::command_queue() const {
  return (__bridge void*)impl_->command_queue_;
}

void* MPSContext::device() const {
  return (__bridge void*)impl_->device_;
}

void MPSContext::Wait() const {
  @autoreleasepool {
    if (impl_->command_queue_) {
      id<MTLCommandBuffer> command_buffer = [impl_->command_queue_ commandBuffer];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }
  }
}

void MPSContext::WaitCommandBuffer(void* command_buffer) const {
  @autoreleasepool {
    if (command_buffer) {
      id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
      [cb waitUntilCompleted];
    }
  }
}

void* MPSContext::CreateCommandBuffer() const {
  @autoreleasepool {
    if (impl_->command_queue_) {
      id<MTLCommandBuffer> command_buffer = [impl_->command_queue_ commandBuffer];
      return (__bridge_retained void*)command_buffer;
    }
    return nullptr;
  }
}

void MPSContext::CommitCommandBuffer(void* command_buffer) const {
  @autoreleasepool {
    if (command_buffer) {
      id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
      [cb commit];
    }
  }
}

void MPSContext::Init() {
  impl_->Init();
}

void MPSContext::SetCommandQueue(void* command_queue) {
  @autoreleasepool {
    impl_->command_queue_ = (__bridge id<MTLCommandQueue>)command_queue;
    impl_->owned_ = false;
  }
}

void MPSContext::SetDevice(void* device) {
  @autoreleasepool {
    impl_->device_ = (__bridge id<MTLDevice>)device;
    impl_->owned_ = false;
  }
}

#else  // TARGET_OS_MAC

// Stub implementation for non-Mac platforms
struct MPSContext::Impl {
  Impl() : place_(MPSPlace()) {}
  explicit Impl(const Place& place) : place_(place) {}
  Place place_;
};

MPSContext::MPSContext(const MPSPlace& place)
    : DeviceContext(), impl_(std::make_unique<MPSContext::Impl>(place)) {
  PADDLE_THROW(common::errors::Unavailable(
      "MPS is only available on macOS with Apple Silicon."));
}

MPSContext::MPSContext(MPSContext&&) = default;
MPSContext& MPSContext::operator=(MPSContext&&) = default;
MPSContext::~MPSContext() = default;

const Place& MPSContext::GetPlace() const { return impl_->place_; }
void* MPSContext::command_queue() const { return nullptr; }
void* MPSContext::device() const { return nullptr; }
void MPSContext::Wait() const {}
void MPSContext::WaitCommandBuffer(void* command_buffer) const {}
void* MPSContext::CreateCommandBuffer() const { return nullptr; }
void MPSContext::CommitCommandBuffer(void* command_buffer) const {}
void MPSContext::Init() {}
void MPSContext::SetCommandQueue(void* command_queue) {}
void MPSContext::SetDevice(void* device) {}

#endif  // TARGET_OS_MAC
#endif  // __APPLE__

}  // namespace phi

#endif  // PADDLE_WITH_MPS


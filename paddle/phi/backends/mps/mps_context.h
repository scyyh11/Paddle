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

#include <memory>
#include <mutex>

#include "paddle/phi/backends/mps/forwards.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

#ifdef __APPLE__
#if TARGET_OS_MAC
// Forward declarations for Metal types (C++ compatible)
// Use void* instead of Objective-C types in C++ headers
// The actual types will be used in .mm implementation files
#endif
#endif

class MPSContext : public DeviceContext,
                   public TypeInfoTraits<DeviceContext, MPSContext> {
 public:
  explicit MPSContext(const MPSPlace& place);

  MPSContext(MPSContext&&);
  MPSContext& operator=(MPSContext&&);

  virtual ~MPSContext();

  /*! \brief  Return place in the device context. */
  const Place& GetPlace() const override;

  /*! \brief  Return MPS command queue in the device context. */
  void* command_queue() const;

  /*! \brief  Return MPS device in the device context. */
  void* device() const;

  /*! \brief  Wait for all operations completion in the command queue. */
  void Wait() const override;

  /*! \brief  Wait for command buffer to complete. */
  void WaitCommandBuffer(void* command_buffer) const;

  /*! \brief  Create a new command buffer. */
  void* CreateCommandBuffer() const;

  /*! \brief  Commit command buffer. */
  void CommitCommandBuffer(void* command_buffer) const;

  static const char* name() { return "MPSContext"; }

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  void Init();

  // NOTE: External users manage resources. Used in inference scenarios.
  // The Set interface is for inference only, DeviceContext will mark the
  // resource as external, and will not delete any resource when destructing.
  void SetCommandQueue(void* command_queue);
  void SetDevice(void* device);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi

#endif  // PADDLE_WITH_MPS


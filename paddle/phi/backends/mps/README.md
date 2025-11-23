# MPS Backend Support for PaddlePaddle

## Overview

This directory contains the implementation of Metal Performance Shaders (MPS) backend support for PaddlePaddle. MPS is Apple's GPU acceleration framework available on macOS with Apple Silicon (M1, M2, M3, etc.).

## Architecture

The MPS backend follows the same design pattern as other device backends in PaddlePaddle:

1. **Place System**: `MPSPlace` represents the MPS device location
2. **Device Context**: `MPSContext` manages MPS device resources (command queues, devices)
3. **Device Info**: `mps_info` provides device query and management functions
4. **Memory Allocator**: `MPSAllocator` manages memory allocation using Metal buffers
5. **Memory Copy**: `mps_copy` provides efficient memory transfer operations
6. **Integration**: Integrated into the DeviceContextPool system and kernel fallback mechanism

## Key Components

### 1. MPSPlace (`paddle/phi/common/place.h`)

- Added `AllocationType::MPS` to the allocation type enum
- `MPSPlace` class similar to `GPUPlace` and `XPUPlace`
- Helper functions: `is_mps_place()`, `DefaultMPSPlace()`

### 2. MPSContext (`mps_context.h` / `mps_context.mm`)

- Implemented using Objective-C++ (`.mm` files) for Metal API interaction
- Manages Metal device (`id<MTLDevice>`) and command queue (`id<MTLCommandQueue>`)
- Provides interface for:
  - Command buffer creation and management
  - Synchronization (Wait operations)
  - Resource lifecycle management
- Uses Automatic Reference Counting (ARC) for memory management

### 3. MPS Info (`mps_info.h` / `mps_info.mm`)

- Implemented using Objective-C++ (`.mm` files)
- Robust device detection following PyTorch's approach:
  - Checks macOS version (12.3+ required)
  - Verifies Metal device existence via `MTLCreateSystemDefaultDevice()`
  - Falls back to `MTLCopyAllDevices()` if default device is unavailable
  - Validates device usability by creating a test command queue
  - Caches availability result for performance
- Device count and selection
- Memory management parameters (for future buddy allocator)
- Functions: `GetMPSDeviceCount()`, `GetCurrentDeviceId()`, `SetDeviceId()`, `IsMPSAvailable()`

### 4. MPS Allocator (`mps_allocator.h` / `mps_allocator.mm`)

- Implemented using Objective-C++ (`.mm` files)
- Uses Metal's `MTLBuffer` with `MTLResourceStorageModeShared` for unified memory
- Allocates memory accessible by both CPU and GPU
- Integrated into PaddlePaddle's allocator facade system
- Thread-safe allocation

### 5. MPS Copy (`mps_copy.h` / `mps_copy.mm`)

- Implemented using Objective-C++ (`.mm` files)
- Memory copy operations following PyTorch's approach
- Uses `std::memcpy` for unified memory (efficient for shared memory architecture)
- Handles CPU↔MPS and MPS↔MPS memory transfers
- Future optimization: Could use Metal blit command encoder if MTLBuffer tracking is added

### 6. Integration Points

- **DeviceContextPool**: MPS contexts are created and managed through the pool
- **EmplaceDeviceContexts**: MPS contexts are created on-demand
- **Operator Execution**: Operators can run on MPS devices with automatic fallback to CPU
- **Kernel Fallback**: Automatic fallback to CPU kernels when MPS-specific kernels are not available
- **Execution Modes**: 
  - **Dygraph/Eager Mode**: Fully supported via `eager.cc`, `imperative.cc`, `eager_math_op_patch.cc`
  - **Static Graph Mode**: Fully supported via `operator.cc` (old executor) and `interpreter_base_impl.h` (new executor)
- **Python API**: `is_compiled_with_mps()`, `paddle.device.mps.device_count()`, `paddle.device.mps.synchronize()`

## Compilation

To enable MPS support, configure CMake with:

```bash
cmake .. -DWITH_MPS=ON
```

**Note**: 
- MPS is only available on macOS with Apple Silicon (ARM architecture)
- When `-DWITH_MPS=ON` is set, the build system automatically sets `-DWITH_ARM=ON`
- The build system will automatically disable MPS on non-Apple platforms
- Objective-C++ language support is automatically enabled for `.mm` files

## Usage

### Python API

```python
import paddle

# Check if MPS is compiled and available
if paddle.device.is_compiled_with_mps():
    print(f"MPS devices: {paddle.device.mps.device_count()}")
    
    # Create tensor on MPS
    x = paddle.to_tensor([1, 2, 3], place=paddle.MPSPlace())
    print(f"Tensor on MPS: {x.is_mps()}")
    
    # Operations automatically fallback to CPU if MPS kernel not available
    y = paddle.sum(x)  # Falls back to CPU kernel
    
    # Synchronize MPS operations
    paddle.device.mps.synchronize()
```

### C++ API

```cpp
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/common/place.h"

// Create MPS place
phi::MPSPlace place(0);

// Get MPS context from pool
auto* ctx = phi::DeviceContextPool::Instance().Get(place);
auto* mps_ctx = static_cast<phi::MPSContext*>(ctx);

// Use MPS context
void* command_queue = mps_ctx->command_queue();
```

## Memory Management

MPS uses unified memory architecture (MTLResourceStorageModeShared), meaning:
- Memory allocated via `MPSAllocator` uses Metal's `MTLBuffer` with shared storage mode
- Both CPU and GPU can access the same memory directly without explicit transfers
- No separate pinned memory allocator needed
- Memory is automatically managed by ARC (Automatic Reference Counting)
- No explicit stream needed for allocator (unlike CUDA)
- Memory copies use `std::memcpy` which is efficient for unified memory

## Limitations and Notes

1. **Platform**: Only available on macOS 12.3+ with Apple Silicon (M1, M2, M3, etc.)
2. **Device Count**: Typically one MPS device per system
3. **Memory**: Uses unified memory (MTLResourceStorageModeShared), different from discrete GPU architectures
4. **Streams**: MPS uses command buffers instead of CUDA-style streams
5. **Kernel Coverage**: Currently, operations fallback to CPU kernels. MPS-specific kernels need to be implemented for optimal performance
6. **Implementation Language**: Uses Objective-C++ (`.mm` files) for Metal API interaction

## Future Enhancements

- [ ] MPS-specific kernel implementations using Metal Performance Shaders
  - Elementwise operations (add, subtract, multiply, divide)
  - Reduction operations (sum, mean, max, min)
  - Linear algebra (matmul, transpose)
  - Neural network operations (conv2d, batch_norm, softmax)
- [ ] Metal blit command encoder for optimized large memory copies (if MTLBuffer tracking is added)
- [ ] Collective communication support (if needed for distributed training)
- [ ] Profiling and performance tools integration
- [ ] Buddy allocator implementation for better memory management

## Implementation Files

### Core Backend Files
- `paddle/phi/backends/mps/mps_context.h` / `mps_context.mm` - Device context implementation
- `paddle/phi/backends/mps/mps_info.h` / `mps_info.mm` - Device information and detection
- `paddle/phi/backends/mps/mps_copy.h` / `mps_copy.mm` - Memory copy operations

### Memory Management
- `paddle/phi/core/memory/allocation/mps_allocator.h` / `mps_allocator.mm` - Memory allocator

### Integration Files
- `paddle/phi/common/place.h` - Place definitions (MPSPlace, AllocationType::MPS)
- `paddle/phi/backends/context_pool.h` / `context_pool.cc` - Context pool integration
- `paddle/phi/core/platform/device_context.cc` - Context creation
- `paddle/phi/core/memory/memcpy.cc` - Memory copy integration
- `paddle/fluid/framework/operator.cc` - Operator execution with MPS fallback (old executor)
- `paddle/fluid/framework/new_executor/interpreter_base_impl.h` - New executor device management
- `paddle/fluid/framework/new_executor/interpreter/execution_config.cc` - New executor thread pool config
- `paddle/fluid/framework/phi_utils.cc` - Kernel fallback mechanism
- `paddle/phi/core/kernel_factory.cc` - Kernel selection with MPS fallback
- `paddle/fluid/pybind/pybind.cc` - Python bindings
- `paddle/fluid/pybind/eager.cc` - Eager/dygraph execution bindings
- `paddle/fluid/pybind/imperative.cc` - Imperative mode bindings
- `paddle/fluid/pybind/eager_math_op_patch.cc` - Eager math operations
- `python/paddle/device/mps/__init__.py` - Python device API

### Build System
- `CMakeLists.txt` - Main CMake configuration
- `cmake/configure.cmake` - Platform-specific configuration
- `paddle/phi/backends/CMakeLists.txt` - Backend build configuration
- `paddle/phi/core/memory/allocation/CMakeLists.txt` - Allocator build configuration


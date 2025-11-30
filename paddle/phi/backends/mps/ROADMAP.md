# MPS Backend Development Roadmap

## Overview

This document outlines the development roadmap for Metal Performance Shaders (MPS) backend support in PaddlePaddle. It provides a structured checklist for implementing MPS-specific kernels and features.

## Current Status ✅

### Infrastructure (Completed)
- [x] **Place System**: `MPSPlace` and `AllocationType::MPS` implemented
- [x] **Device Context**: `MPSContext` with Metal device and command queue management
- [x] **Device Info**: Robust MPS device detection and query functions
- [x] **Memory Allocator**: `MPSAllocator` using Metal `MTLBuffer` with unified memory
- [x] **Memory Copy**: Basic `memcpy` implementation for CPU↔MPS and MPS↔MPS transfers
- [x] **Kernel Fallback**: Automatic fallback to CPU kernels when MPS kernels are unavailable
- [x] **Execution Modes**: Both dygraph and static graph modes supported
- [x] **Python API**: `is_compiled_with_mps()`, `device_count()`, `synchronize()`
- [x] **Build System**: CMake integration with automatic ARM detection
- [x] **Integration**: DeviceContextPool, operator execution, kernel registry

### What Works Now
- ✅ Creating tensors on MPS device
- ✅ Memory allocation on MPS
- ✅ Basic memory copy operations
- ✅ Automatic CPU fallback for all operations
- ✅ Both dygraph and static graph execution

### What Doesn't Work Yet
- ❌ MPS-specific kernel implementations (all operations fallback to CPU)
- ❌ Optimized memory transfers using Metal blit
- ❌ Performance optimizations

---

## Development Phases

### Phase 1: Foundation & Basic Kernels (Current Priority)

**Goal**: Implement essential kernels with core dtypes to establish the pattern and validate the infrastructure.

#### 1.1 Setup Kernel Infrastructure
- [ ] Create `paddle/phi/kernels/mps/` directory structure
- [ ] Update `paddle/phi/kernels/CMakeLists.txt` to include MPS kernels
- [ ] Ensure `.mm` files are compiled with Objective-C++ language
- [ ] Create helper utilities for MPS kernel development
  - [ ] MPSGraph wrapper utilities
  - [ ] Tensor to MPSGraph tensor conversion helpers
  - [ ] Data type conversion utilities

#### 1.2 Implement First Kernel: Elementwise Add
**Priority**: Highest - This establishes the pattern for all future kernels

- [ ] Create `paddle/phi/kernels/mps/elementwise_add_kernel.h`
- [ ] Create `paddle/phi/kernels/mps/elementwise_add_kernel.mm`
- [ ] Implement using MPSGraph API
- [ ] Register kernel with dtype support:
  - [ ] `float` (Phase 1.2.1 - Start here)
  - [ ] `phi::float16` (Phase 1.2.2)
  - [ ] `int` (Phase 1.2.3)
  - [ ] `int64_t` (Phase 1.2.4)
- [ ] Write unit tests
- [ ] Verify correctness against CPU implementation
- [ ] Performance benchmarking

#### 1.3 Implement Other Elementwise Operations
**Priority**: High - These follow the same pattern as Add

- [ ] `elementwise_subtract_kernel.mm`
  - [ ] float, float16, int, int64_t
- [ ] `elementwise_multiply_kernel.mm`
  - [ ] float, float16, int, int64_t
- [ ] `elementwise_divide_kernel.mm`
  - [ ] float, float16, int, int64_t
- [ ] `elementwise_max_kernel.mm`
  - [ ] float, float16, int, int64_t
- [ ] `elementwise_min_kernel.mm`
  - [ ] float, float16, int, int64_t

#### 1.4 Implement Reduction Operations
**Priority**: High - Commonly used, relatively straightforward

- [ ] `reduce_sum_kernel.mm`
  - [ ] float, float16, int, int64_t
- [ ] `reduce_mean_kernel.mm`
  - [ ] float, float16
- [ ] `reduce_max_kernel.mm`
  - [ ] float, float16, int, int64_t
- [ ] `reduce_min_kernel.mm`
  - [ ] float, float16, int, int64_t

#### 1.5 Implement Basic Tensor Operations
**Priority**: Medium - Needed for tensor manipulation

- [ ] `fill_kernel.mm`
  - [ ] float, float16, int, int64_t, bool
- [ ] `cast_kernel.mm`
  - [ ] Core type conversions (float ↔ float16, int ↔ int64_t)
- [ ] `reshape_kernel.mm` (if needed, may use view)
- [ ] `transpose_kernel.mm`
  - [ ] float, float16

**Phase 1 Success Criteria**:
- ✅ At least 3 elementwise operations working with float and float16
- ✅ At least 2 reduction operations working
- ✅ All implementations pass unit tests
- ✅ Performance is better than CPU fallback (even if not optimal)

---

### Phase 2: Linear Algebra & Advanced Operations

**Goal**: Implement matrix operations and more complex kernels.

#### 2.1 Matrix Operations
**Priority**: High - Essential for neural networks

- [ ] `matmul_kernel.mm`
  - [ ] float, float16
  - [ ] Support for different matrix shapes
  - [ ] Optional: Transpose support
- [ ] `batch_matmul_kernel.mm` (if different from matmul)
  - [ ] float, float16

#### 2.2 Activation Functions
**Priority**: Medium - Needed for neural networks

- [ ] `relu_kernel.mm`
  - [ ] float, float16
- [ ] `sigmoid_kernel.mm`
  - [ ] float, float16
- [ ] `tanh_kernel.mm`
  - [ ] float, float16
- [ ] `gelu_kernel.mm`
  - [ ] float, float16
- [ ] `softmax_kernel.mm`
  - [ ] float, float16

#### 2.3 Normalization Operations
**Priority**: Medium - Important for training

- [ ] `layer_norm_kernel.mm`
  - [ ] float, float16
- [ ] `batch_norm_kernel.mm`
  - [ ] float, float16
- [ ] `group_norm_kernel.mm` (if needed)
  - [ ] float, float16

**Phase 2 Success Criteria**:
- ✅ Matrix multiplication working with competitive performance
- ✅ At least 3 activation functions implemented
- ✅ At least 1 normalization operation working

---

### Phase 3: Neural Network Operations

**Goal**: Implement convolution and other deep learning primitives.

#### 3.1 Convolution Operations
**Priority**: High - Core of CNNs

- [ ] `conv2d_kernel.mm`
  - [ ] float, float16
  - [ ] Support common padding modes
  - [ ] Support common stride values
- [ ] `conv2d_transpose_kernel.mm` (if needed)
  - [ ] float, float16
- [ ] `depthwise_conv2d_kernel.mm` (if different implementation)
  - [ ] float, float16

#### 3.2 Pooling Operations
**Priority**: Medium - Common in CNNs

- [ ] `max_pool2d_kernel.mm`
  - [ ] float, float16
- [ ] `avg_pool2d_kernel.mm`
  - [ ] float, float16
- [ ] `adaptive_pool2d_kernel.mm` (if needed)
  - [ ] float, float16

#### 3.3 Advanced Operations
**Priority**: Low-Medium - Based on user needs

- [ ] `dropout_kernel.mm`
  - [ ] float, float16
- [ ] `embedding_kernel.mm`
  - [ ] float, float16
- [ ] `gather_kernel.mm`
  - [ ] float, float16, int, int64_t
- [ ] `scatter_kernel.mm`
  - [ ] float, float16

**Phase 3 Success Criteria**:
- ✅ Convolution working with reasonable performance
- ✅ At least 1 pooling operation implemented
- ✅ Can run simple CNN models end-to-end on MPS

---

### Phase 4: Optimization & Advanced Features

**Goal**: Optimize performance and add advanced features.

#### 4.1 Memory Optimization
**Priority**: Medium - Improve performance

- [ ] Implement Metal blit command encoder for large memory copies
  - [ ] Requires MTLBuffer tracking system
  - [ ] Optimize CPU↔MPS transfers
  - [ ] Optimize MPS↔MPS transfers
- [ ] Implement buddy allocator for better memory management
  - [ ] Reduce allocation overhead
  - [ ] Better memory reuse

#### 4.2 Performance Optimizations
**Priority**: Medium - Make MPS competitive

- [ ] Kernel fusion opportunities
  - [ ] Elementwise + activation fusion
  - [ ] BatchNorm + activation fusion
- [ ] Graph optimization
  - [ ] MPSGraph optimization passes
  - [ ] Operation scheduling
- [ ] Profiling and performance tools
  - [ ] Metal System Trace integration
  - [ ] Performance counters

#### 4.3 Additional Dtype Support
**Priority**: Low - Based on user needs

- [ ] `int8_t`, `uint8_t` (quantization support)
- [ ] `bool` (logical operations)
- [ ] `double` (if needed, may have limited MPSGraph support)
- [ ] `bfloat16` (via conversion or CPU fallback)
- [ ] Complex types (likely CPU fallback)

#### 4.4 Advanced Features
**Priority**: Low - Nice to have

- [ ] Distributed training support (if needed)
- [ ] Mixed precision training
- [ ] Automatic mixed precision (AMP)
- [ ] Memory pool optimization

**Phase 4 Success Criteria**:
- ✅ Memory transfers optimized
- ✅ Performance competitive with CPU for common operations
- ✅ Can run real-world models efficiently

---

## Dtype Implementation Strategy

### Priority Order for Each Kernel

When implementing a new kernel, add dtype support in this order:

1. **Phase 1 (Essential)**:
   - `float` - Start here, most common
   - `phi::float16` - Important for inference and mixed precision
   - `int` - Needed for indices and integer operations
   - `int64_t` - Needed for large indices

2. **Phase 2 (Additional)**:
   - `int8_t`, `uint8_t` - Quantization support
   - `bool` - Logical operations

3. **Phase 3 (Advanced - Optional)**:
   - `double` - Limited Metal support, may need fallback
   - `bfloat16` - Not natively supported, use CPU fallback
   - Complex types - Use CPU fallback

### Dtype Support Matrix

| Dtype | Metal Support | MPSGraph Support | Priority | Implementation |
|-------|---------------|------------------|----------|----------------|
| `float` | ✅ Native | ✅ Full | **Highest** | Implement first |
| `phi::float16` | ✅ Native | ✅ Full | **High** | Implement second |
| `int` | ✅ Native | ✅ Full | **High** | Implement third |
| `int64_t` | ✅ Native | ⚠️ Limited | Medium | Implement fourth |
| `int8_t` | ✅ Native | ✅ Full | Medium | Add later |
| `uint8_t` | ✅ Native | ✅ Full | Medium | Add later |
| `bool` | ✅ Native | ✅ Full | Medium | Add later |
| `double` | ✅ Native | ⚠️ Limited | Low | Add if needed |
| `phi::bfloat16` | ❌ No | ❌ No | Low | CPU fallback |
| `phi::complex64` | ❌ No | ❌ No | Very Low | CPU fallback |

---

## Testing Strategy

### Unit Tests
For each kernel implementation:
- [ ] Create test file: `test/kernels/mps/{kernel_name}_test.cc`
- [ ] Test with different tensor shapes
- [ ] Test with different dtypes
- [ ] Compare results with CPU implementation (within tolerance)
- [ ] Test edge cases (empty tensors, single element, etc.)

### Integration Tests
- [ ] Test with real models (simple CNN, MLP)
- [ ] Test end-to-end training (if applicable)
- [ ] Test inference workloads
- [ ] Performance benchmarking vs CPU

### Test Coverage Goals
- [ ] At least 80% code coverage for each kernel
- [ ] All dtype combinations tested
- [ ] Common shape combinations tested
- [ ] Edge cases covered

---

## Implementation Guidelines

### Code Structure

Each kernel should follow this structure:

```
paddle/phi/kernels/mps/
├── {kernel_name}_kernel.h          # Kernel declaration
├── {kernel_name}_kernel.mm         # MPS implementation
└── (if needed)
    └── {kernel_name}_kernel_impl.h # Shared implementation logic
```

### MPSGraph Usage Pattern

```objective-c++
// 1. Get MPSContext
auto* mps_ctx = static_cast<phi::MPSContext*>(&dev_ctx);

// 2. Create MPSGraph
MPSGraph* graph = [[MPSGraph alloc] init];

// 3. Create input tensors
MPSGraphTensor* inputTensor = [graph placeholderWithShape:... dataType:...];

// 4. Build computation graph
MPSGraphTensor* outputTensor = [graph operationWithInputs:...];

// 5. Execute
MPSGraphExecutionDescriptor* execDesc = [[MPSGraphExecutionDescriptor alloc] init];
[graph encodeToCommandBuffer:commandBuffer 
                  inputTensors:inputDict 
         inputMPSStates:nil 
                outputTensors:outputDict 
               executionDescriptor:execDesc];
```

### Best Practices

1. **Always check MPS availability** before using MPS-specific code
2. **Use unified memory** - MPS uses shared memory, no explicit transfers needed
3. **Reuse MPSGraph instances** when possible (cache them)
4. **Handle errors gracefully** - Fallback to CPU if MPS operation fails
5. **Profile performance** - Use Metal System Trace to identify bottlenecks
6. **Test on multiple devices** - M1, M2, M3 may have different performance characteristics

---

## Performance Targets

### Phase 1 Targets
- [ ] Elementwise operations: 2-5x faster than CPU
- [ ] Reduction operations: 1.5-3x faster than CPU
- [ ] Memory operations: Comparable to CPU (unified memory)

### Phase 2 Targets
- [ ] Matrix multiplication: 3-10x faster than CPU
- [ ] Activation functions: 2-5x faster than CPU

### Phase 3 Targets
- [ ] Convolution: 2-5x faster than CPU (depends on kernel size)
- [ ] End-to-end model inference: 1.5-3x faster than CPU

### Phase 4 Targets
- [ ] Optimized memory transfers: Minimal overhead
- [ ] Overall model performance: Competitive with CPU for inference

---

## Known Limitations

### Current Limitations
1. **No MPS kernels yet** - All operations fallback to CPU
2. **Memory transfers** - Using basic memcpy, not optimized Metal blit
3. **No kernel fusion** - Each operation is separate
4. **Limited dtype support** - Only basic types when implemented

### Metal/MPS Limitations
1. **Platform**: Only macOS 12.3+ with Apple Silicon
2. **Device count**: Typically one MPS device per system
3. **Memory model**: Unified memory (different from discrete GPUs)
4. **Streams**: Uses command buffers, not CUDA-style streams
5. **Dtype support**: Limited support for double, no bfloat16/complex

---

## Contributing Guidelines

### Before Starting Work
1. Check this roadmap to see if the feature is planned
2. Create an issue or discuss in PR if adding new major features
3. Follow the dtype implementation strategy
4. Ensure tests are written before implementation

### Code Review Checklist
- [ ] Follows existing code style
- [ ] Has unit tests
- [ ] Handles errors gracefully
- [ ] Includes performance benchmarks (for new kernels)
- [ ] Documentation updated
- [ ] No memory leaks (ARC handles this, but verify)

### PR Requirements
- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Performance benchmarks included (for performance-critical changes)
- [ ] Documentation updated (README, code comments)
- [ ] No breaking changes (or clearly documented)

---

## Resources

### Apple Documentation
- [Metal Performance Shaders Framework](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSGraph](https://developer.apple.com/documentation/metalperformanceshaders/mpsgraph)
- [Metal Shading Language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

### Reference Implementations
- PyTorch MPS backend: Good reference for MPSGraph usage patterns
- Metal Performance Shaders examples: Apple's official examples

### Tools
- **Metal System Trace**: For profiling Metal performance
- **Instruments**: For memory and performance analysis
- **Xcode**: For debugging Objective-C++ code

---

## Timeline Estimate

### Phase 1: Foundation (2-4 weeks)
- Week 1-2: Setup infrastructure, implement first kernel (elementwise_add)
- Week 3-4: Implement other elementwise and reduction operations

### Phase 2: Linear Algebra (2-3 weeks)
- Week 1-2: Matrix operations, activation functions
- Week 3: Normalization operations

### Phase 3: Neural Networks (3-4 weeks)
- Week 1-2: Convolution operations
- Week 3: Pooling operations
- Week 4: Advanced operations and testing

### Phase 4: Optimization (Ongoing)
- Continuous improvement based on user feedback and profiling

**Total Estimated Time**: 2-3 months for basic functionality, ongoing for optimization

---

## Success Metrics

### Short-term (Phase 1-2)
- [ ] Can run simple ML models on MPS
- [ ] Performance is better than CPU for supported operations
- [ ] All tests pass

### Medium-term (Phase 3)
- [ ] Can run CNN models on MPS
- [ ] Performance is competitive with CPU
- [ ] Good test coverage

### Long-term (Phase 4)
- [ ] Can run production models efficiently
- [ ] Performance is significantly better than CPU
- [ ] Comprehensive feature set

---

## Notes

- This roadmap is a living document and should be updated as development progresses
- Priorities may shift based on user needs and feedback
- Some features may be implemented in parallel by different contributors
- Performance targets are estimates and may vary by device and workload

---

**Last Updated**: 2024
**Status**: Phase 1 - Foundation & Basic Kernels (In Progress)


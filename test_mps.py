#!/usr/bin/env python3
"""
Test script for MPS backend functionality
"""

import paddle
import numpy as np

print("=" * 60)
print("Testing MPS Backend")
print("=" * 60)

# Test 1: Check if MPS is compiled
print("\n1. Checking if MPS is compiled...")
try:
    is_compiled = paddle.device.is_compiled_with_mps()
    print(f"   paddle.device.is_compiled_with_mps(): {is_compiled}")
except AttributeError:
    # Fallback to core
    try:
        is_compiled = paddle.base.core.is_compiled_with_mps()
        print(f"   paddle.base.core.is_compiled_with_mps(): {is_compiled}")
    except AttributeError:
        is_compiled = paddle.fluid.core.is_compiled_with_mps()
        print(f"   paddle.fluid.core.is_compiled_with_mps(): {is_compiled}")

if not is_compiled:
    print("   ERROR: MPS is not compiled. Please rebuild with -DWITH_MPS=ON")
    exit(1)

# Test 2: Check device count
print("\n2. Checking MPS device count...")
try:
    device_count = paddle.device.device_count()
    print(f"   device_count(): {device_count}")
    if device_count == 0:
        print("   WARNING: No MPS devices found. Make sure you're on an Apple Silicon Mac.")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: Check MPS device count specifically
print("\n3. Checking MPS device count (specific)...")
try:
    mps_count = paddle.device.mps.device_count()
    print(f"   paddle.device.mps.device_count(): {mps_count}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Create MPSPlace
print("\n4. Creating MPSPlace...")
try:
    mps_place = paddle.MPSPlace(0)
    print(f"   MPSPlace(0): {mps_place}")
    print(f"   MPSPlace device_id: {mps_place.get_device_id()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Create tensor on MPS using place string
print("\n5. Creating tensor on MPS using place='mps'...")
try:
    x = paddle.to_tensor([1.0, 2.0, 3.0], place='mps')
    print(f"   Tensor: {x}")
    print(f"   Tensor place: {x.place}")
    print(f"   Tensor data: {x.numpy()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 6: Create tensor on MPS using MPSPlace object
print("\n6. Creating tensor on MPS using MPSPlace object...")
try:
    mps_place = paddle.MPSPlace(0)
    y = paddle.to_tensor([4.0, 5.0, 6.0], place=mps_place)
    print(f"   Tensor: {y}")
    print(f"   Tensor place: {y.place}")
    print(f"   Tensor data: {y.numpy()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 7: Basic operations on MPS tensors
print("\n7. Testing basic operations on MPS tensors...")
try:
    a = paddle.to_tensor([1.0, 2.0, 3.0], place='mps')
    b = paddle.to_tensor([4.0, 5.0, 6.0], place='mps')
    c = a + b
    print(f"   a + b = {c.numpy()}")
    
    d = a * 2.0
    print(f"   a * 2.0 = {d.numpy()}")
    
    e = paddle.sum(a)
    print(f"   sum(a) = {e.numpy()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 8: Copy from CPU to MPS
print("\n8. Testing copy from CPU to MPS...")
try:
    cpu_tensor = paddle.to_tensor([1.0, 2.0, 3.0], place='cpu')
    mps_tensor = cpu_tensor.mps() if hasattr(cpu_tensor, 'mps') else None
    if mps_tensor is None:
        # Manual copy
        mps_tensor = paddle.to_tensor(cpu_tensor.numpy(), place='mps')
    print(f"   CPU tensor: {cpu_tensor.place}")
    print(f"   MPS tensor: {mps_tensor.place}")
    print(f"   Data matches: {np.allclose(cpu_tensor.numpy(), mps_tensor.numpy())}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 9: Check place type
print("\n9. Testing place type checks...")
try:
    mps_place = paddle.MPSPlace(0)
    # Try different core imports
    try:
        core = paddle.base.core
    except AttributeError:
        core = paddle.fluid.core
    print(f"   is_mps_place: {core.is_mps_place(mps_place)}")
    
    # Test with Place object
    place = core.Place(mps_place)
    print(f"   Place.is_mps_place(): {place.is_mps_place()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 10: Set device to MPS
print("\n10. Testing set_device('mps')...")
try:
    paddle.device.set_device('mps')
    current_place = paddle.device.get_device()
    print(f"   Current device: {current_place}")
    
    # Create tensor without specifying place (should use current device)
    z = paddle.to_tensor([7.0, 8.0, 9.0])
    print(f"   Tensor place (should be MPS): {z.place}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)


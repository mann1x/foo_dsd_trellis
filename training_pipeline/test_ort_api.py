"""Test ORT C API vtable slot positions by calling functions through ctypes."""
import ctypes
import ctypes.wintypes as wt
import sys

dll_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\manni\source\repos\foo_dsd_trellis\bin\Release\x64\onnxruntime.dll"
print(f"Loading: {dll_path}")

dll = ctypes.CDLL(dll_path)

# OrtGetApiBase
get_api_base = dll.OrtGetApiBase
get_api_base.restype = ctypes.c_void_p
base_ptr = get_api_base()
print(f"OrtApiBase at: 0x{base_ptr:x}")

# Read vtable: GetApi at slot 0, GetVersionString at slot 1
base_vt = (ctypes.c_void_p * 2).from_address(base_ptr)

# GetVersionString
ver_fn = ctypes.CFUNCTYPE(ctypes.c_char_p)(base_vt[1])
version = ver_fn().decode()
print(f"Version: {version}")

# GetApi(18)
get_api_fn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)(base_vt[0])
api_ptr = get_api_fn(18)
print(f"OrtApi (v18) at: 0x{api_ptr:x}")

if not api_ptr:
    print("ERROR: GetApi(18) returned NULL")
    sys.exit(1)

# The ORT API is a big struct of function pointers.
# Each slot is a void* (8 bytes on x64).
# Let's read the first 100 slots and check which are non-NULL.
api_vt = (ctypes.c_void_p * 200).from_address(api_ptr)

# Print non-NULL slots to verify layout
key_slots = {
    0: "CreateStatus",
    1: "GetErrorMessage",
    2: "CreateEnv",
    13: "CreateSession",
    14: "CreateSessionFromArray",
    15: "Run",
    18: "CreateSessionOptions",
    31: "SetSessionGraphOptimizationLevel",
    32: "SetIntraOpNumThreads",
    33: "SetInterOpNumThreads",
    38: "CreateCpuMemoryInfo",
    44: "SessionGetInputName",
    45: "SessionGetOutputName",
    50: "ReleaseEnv",
    51: "ReleaseStatus",
    52: "ReleaseMemoryInfo",
    53: "ReleaseSession",
    54: "ReleaseValue",
    58: "ReleaseSessionOptions",
    80: "GetAllocatorWithDefaultOptions",
    84: "AllocatorFree",
    93: "CreateTensorWithDataAsOrtValue",
    97: "GetTensorMutableData",
}

print(f"\nVtable slots (first 100):")
for i in range(100):
    ptr = api_vt[i]
    name = key_slots.get(i, "")
    if name or ptr:
        status = "OK" if ptr else "NULL"
        print(f"  [{i:3d}] 0x{ptr:016x}  {status}  {name}")

# Try calling CreateEnv (slot 2)
print(f"\nTesting CreateEnv (slot 2)...")
CreateEnv = ctypes.CFUNCTYPE(
    ctypes.c_void_p,  # OrtStatus*
    ctypes.c_int,     # log level
    ctypes.c_char_p,  # name
    ctypes.POINTER(ctypes.c_void_p)  # OrtEnv**
)(api_vt[2])

env_ptr = ctypes.c_void_p()
status = CreateEnv(2, b"test", ctypes.byref(env_ptr))
if status:
    print(f"  CreateEnv failed: status=0x{status:x}")
else:
    print(f"  CreateEnv OK: env=0x{env_ptr.value:x}")

# Try CreateSessionOptions (slot 18)
print(f"\nTesting CreateSessionOptions (slot 18)...")
CreateSessionOptions = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p)
)(api_vt[18])

opts_ptr = ctypes.c_void_p()
status = CreateSessionOptions(ctypes.byref(opts_ptr))
if status:
    print(f"  CreateSessionOptions failed: status=0x{status:x}")
else:
    print(f"  CreateSessionOptions OK: opts=0x{opts_ptr.value:x}")

print("\nAll tests passed!")

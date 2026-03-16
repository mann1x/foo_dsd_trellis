"""Test ORT C API - detailed error handling."""
import ctypes
import sys

dll_path = r"C:\Users\manni\source\repos\foo_dsd_trellis\bin\Release\x64\onnxruntime.dll"
dll = ctypes.CDLL(dll_path)

get_api_base = dll.OrtGetApiBase
get_api_base.restype = ctypes.c_void_p
base_ptr = get_api_base()
base_vt = (ctypes.c_void_p * 2).from_address(base_ptr)

get_api_fn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)(base_vt[0])
api_ptr = get_api_fn(18)
api_vt = (ctypes.c_void_p * 200).from_address(api_ptr)

print(f"ORT API at 0x{api_ptr:x}")

# GetErrorMessage (slot 1) — returns const char*
GetErrorMessage = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)(api_vt[1])

# ReleaseStatus (slot 51)
ReleaseStatus = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(api_vt[51])

# CreateEnv (slot 2)
CreateEnv = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p)
)(api_vt[2])

env_ptr = ctypes.c_void_p()
status = CreateEnv(2, b"test", ctypes.byref(env_ptr))
print(f"CreateEnv status: 0x{status:x}" if status else "CreateEnv: OK")

if status:
    try:
        msg = GetErrorMessage(status)
        print(f"  Error: {msg}")
        ReleaseStatus(status)
    except Exception as e:
        print(f"  Cannot read error: {e}")
    sys.exit(1)

print(f"  env=0x{env_ptr.value:x}")

# CreateSessionOptions (slot 18)
CreateSessionOptions = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)(api_vt[18])

opts_ptr = ctypes.c_void_p()
status = CreateSessionOptions(ctypes.byref(opts_ptr))
print(f"CreateSessionOptions: {'OK' if not status else f'FAILED 0x{status:x}'}")
if not status and opts_ptr.value:
    print(f"  opts=0x{opts_ptr.value:x}")

# CreateSession with nonexistent model (slot 13)
CreateSession = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,   # OrtEnv*
    ctypes.c_wchar_p,  # model path
    ctypes.c_void_p,   # OrtSessionOptions*
    ctypes.POINTER(ctypes.c_void_p)  # OrtSession**
)(api_vt[13])

session_ptr = ctypes.c_void_p()
status = CreateSession(env_ptr, "nonexistent.onnx", opts_ptr, ctypes.byref(session_ptr))
if status:
    try:
        msg = GetErrorMessage(status)
        print(f"CreateSession (nonexistent): Expected failure - {msg}")
        ReleaseStatus(status)
    except Exception as e:
        print(f"CreateSession error read failed: {e}")
else:
    print("CreateSession succeeded (unexpected)")

# Try with real model
model_path = r"C:\Users\manni\source\repos\foo_dsd_trellis\bin\Release\x64\foo_dsd_trellis_ml.onnx"
status = CreateSession(env_ptr, model_path, opts_ptr, ctypes.byref(session_ptr))
if status:
    try:
        msg = GetErrorMessage(status)
        print(f"CreateSession (real model): FAILED - {msg}")
        ReleaseStatus(status)
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"CreateSession (real model): OK, session=0x{session_ptr.value:x}")

# Clean up
ReleaseSessionOptions = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(api_vt[58])
if opts_ptr.value:
    ReleaseSessionOptions(opts_ptr)
ReleaseEnv = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(api_vt[50])
if env_ptr.value:
    ReleaseEnv(env_ptr)

print("\nDone!")

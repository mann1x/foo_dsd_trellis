"""Check ONNX Runtime DLL version."""
import ctypes
import sys

dll_path = sys.argv[1] if len(sys.argv) > 1 else "onnxruntime.dll"
try:
    dll = ctypes.CDLL(dll_path)
    get_api_base = dll.OrtGetApiBase
    get_api_base.restype = ctypes.c_void_p
    base_ptr = get_api_base()

    # OrtApiBase layout: GetApi at offset 0, GetVersionString at offset 8
    vt = (ctypes.c_void_p * 2).from_address(base_ptr)
    ver_fn = ctypes.CFUNCTYPE(ctypes.c_char_p)(vt[1])
    version = ver_fn().decode()
    print(f"ONNX Runtime version: {version}")
    print(f"DLL: {dll_path}")
except Exception as e:
    print(f"Error: {e}")

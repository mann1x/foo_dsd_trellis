"""Convert PTX file to C header with binary array (avoids string concat OOM)."""
import sys, os

ptx_file = sys.argv[1] if len(sys.argv) > 1 else "sdm_parallel.ptx"
out_file = sys.argv[2] if len(sys.argv) > 2 else "sdm_parallel_ptx.h"

with open(ptx_file, "rb") as f:
    data = f.read()

base = os.path.splitext(os.path.basename(ptx_file))[0]
var_name = "g_ptx_" + base

with open(out_file, "w") as out:
    out.write("/* Auto-generated from {} — do not edit */\n".format(ptx_file))
    out.write("static const char {}[] = {{\n".format(var_name))
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_vals = ", ".join("0x{:02x}".format(b) for b in chunk)
        out.write("    {},\n".format(hex_vals))
    out.write("    0x00  /* null terminator */\n")
    out.write("};\n")

print(f"Embedded {len(data)} bytes from {ptx_file} -> {out_file} (binary)")

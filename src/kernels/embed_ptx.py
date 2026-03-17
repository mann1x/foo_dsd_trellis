"""Convert PTX file to C header with embedded string constant."""
import sys

ptx_file = sys.argv[1] if len(sys.argv) > 1 else "fir_kernels.ptx"
out_file = sys.argv[2] if len(sys.argv) > 2 else "fir_kernels_ptx.h"

with open(ptx_file, "r") as f:
    lines = f.readlines()

with open(out_file, "w") as out:
    out.write("/* Auto-generated from {} — do not edit */\n".format(ptx_file))
    out.write("static const char g_ptx_fir_kernels[] =\n")
    for line in lines:
        clean = line.rstrip("\n").replace("\\", "\\\\").replace('"', '\\"')
        out.write('    "{}\\n"\n'.format(clean))
    out.write("    ;\n")

print(f"Embedded {len(lines)} lines from {ptx_file} -> {out_file}")

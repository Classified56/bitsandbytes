"""Build and bind the native helpers; own page-aligned activation buffers."""

import ctypes as ct
import mmap
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from flush_reload_test import FlushReload


class Native(FlushReload):
    """Reuse the verified timer API, with additional prefill kernels."""

    def __init__(self):
        # Compile both sources together so timing uses the existing primitive.
        if os.uname().machine not in ("x86_64", "AMD64"):
            raise RuntimeError("This experiment requires x86-64 with RDTSCP")
        self._build = tempfile.TemporaryDirectory(prefix="prefill-flush-reload-")
        root = Path(__file__).resolve().parent
        library = Path(self._build.name) / "prefill.so"
        command = [*shlex.split(os.environ.get("CC", "cc")), "-O3", "-mf16c", "-shared", "-fPIC",
                   str(root / "kernels.c"), str(root.parent / "flush_reload_native.c"),
                   "-lm", "-o", str(library)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        self._lib = ct.CDLL(str(library))
        self._lib.platform_supported.argtypes = []
        self._lib.platform_supported.restype = ct.c_int
        if not self._lib.platform_supported():
            self._build.cleanup()
            raise RuntimeError("Requires F16C, RDTSCP, and 64-byte cache lines")
        self.timer, self.unit, self.threshold = "native", "TSC ticks", None
        ptr, size, fp = ct.c_void_p, ct.c_size_t, ct.c_float
        signatures = {
            "fr_flush": ([ptr], None), "fr_touch": ([ptr], None),
            "fr_reload": ([ptr], ct.c_uint64),
            "fill_half": ([ptr, size, ct.c_uint32, fp], None),
            "inject": ([ptr, size, size, ptr, size, fp], None),
            "quantize": ([ptr, ptr, ptr, ptr, ptr, size, size, fp], size),
            "gather": ([ptr, ptr, ptr, size, size, size], None),
            "flush_lines": ([ptr, size], None), "probe_lines": ([ptr, ptr, ptr, size], None),
            "expand_weights": ([ptr, ptr, ptr, ptr, ptr, size, size], None),
            "references": ([ptr] * 5 + [size] * 3, None),
        }
        # down_project has six pointers, the selected-column count, three
        # more pointers, and the three matrix dimensions.
        signatures["down_project"] = ([ptr] * 6 + [size] + [ptr] * 3 + [size] * 3, None)
        for name, (arguments, result) in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = arguments, result

    def __getattr__(self, name):
        return getattr(self._lib, name)


class HalfMatrix:
    """Keep storage and its ctypes view alive until all native calls finish."""

    def __init__(self, rows, columns):
        self.mapping = mmap.mmap(-1, rows * columns * 2)
        self.data = (ct.c_uint16 * (rows * columns)).from_buffer(self.mapping)
        self.address = ct.addressof(self.data)
        assert self.address % 64 == 0

    def close(self):
        del self.data
        self.mapping.close()

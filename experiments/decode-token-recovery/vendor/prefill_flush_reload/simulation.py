"""Create synthetic prefill tensors and run the quantized down projection.

Nothing in this module infers outliers from timings: exact indices are known
simulation ground truth and are kept separate from cache predictions.
"""

import ctypes as ct
import math
import random
import struct

from .native import HalfMatrix


def half_value(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


class Simulation:
    def __init__(self, native, tokens, width, hidden, seed, threshold, outliers):
        self.native = native
        self.tokens, self.width, self.hidden = tokens, width, hidden
        self.seed, self.threshold = seed, threshold
        self.activations = HalfMatrix(tokens, width)
        self.weights = HalfMatrix(hidden, width)
        self.qa = (ct.c_int8 * (tokens * width))()
        self.qw = (ct.c_int8 * (hidden * width))()
        self.sa, self.sw = (ct.c_float * tokens)(), (ct.c_float * hidden)()
        self.columns, self.mask = (ct.c_size_t * width)(), (ct.c_uint8 * width)()
        # Reserve and pre-fault the largest possible selected-column copy.
        self.sub = (ct.c_uint16 * (tokens * max(1, outliers)))()
        rng = random.Random(seed)
        self.injected = sorted(rng.sample(range(width), outliers))
        self.inject_array = (ct.c_size_t * outliers)(*self.injected)
        native.fill_half(self.weights.data, hidden * width, seed + 1, .02)
        scratch_columns, scratch_mask = (ct.c_size_t * width)(), (ct.c_uint8 * width)()
        native.quantize(self.weights.data, self.qw, self.sw, scratch_columns,
                        scratch_mask, hidden, width, 0)

    def prepare(self, injected):
        """Perform detection/quantization completely before flushing."""
        n = self.native
        n.fill_half(self.activations.data, self.tokens * self.width, self.seed, 1)
        if injected:
            n.inject(self.activations.data, self.tokens, self.width, self.inject_array,
                     len(self.injected), max(8, self.threshold * 1.5))
        self.count = n.quantize(self.activations.data, self.qa, self.sa, self.columns,
                               self.mask, self.tokens, self.width, self.threshold)
        expected = self.injected if injected else []
        assert list(self.columns)[:self.count] == expected, "Outlier detection mismatch"
        # Confirm selected columns are completely absent from the INT8 path.
        for column in expected:
            assert all(self.qa[row * self.width + column] == 0 for row in range(self.tokens))
        return expected

    def gather(self):
        self.native.gather(self.activations.data, self.sub, self.columns, self.count,
                           self.tokens, self.width)

    def touched_lines(self, columns):
        return {(row * self.width + column) // 32
                for row in range(self.tokens) for column in columns}

    def numerical_check(self):
        """Run computation after the probe, then independently check samples."""
        n = self.native
        weight_count, output_count = self.width * self.hidden, self.tokens * self.hidden
        original_w, dequant_w = (ct.c_float * weight_count)(), (ct.c_float * weight_count)()
        normal, mixed = (ct.c_float * output_count)(), (ct.c_float * output_count)()
        original_ref, dequant_ref = (ct.c_float * output_count)(), (ct.c_float * output_count)()
        n.expand_weights(self.weights.data, self.qw, self.sw, original_w, dequant_w,
                         self.hidden, self.width)
        n.down_project(self.qa, self.qw, self.sa, self.sw, self.sub, self.columns,
                       self.count, dequant_w, normal, mixed, self.tokens, self.width, self.hidden)
        n.references(self.activations.data, original_w, dequant_w, original_ref, dequant_ref,
                     self.tokens, self.width, self.hidden)
        # Independent Python scalar reference for selected outputs catches layout,
        # column-union, gather, and scale mistakes without mirroring the C loop.
        for row in sorted({0, self.tokens // 2, self.tokens - 1}):
            for j in range(self.count):
                assert self.sub[row * self.count + j] == self.activations.data[
                    row * self.width + self.columns[j]]
            for h in sorted({0, self.hidden // 2, self.hidden - 1}):
                offset = row * self.hidden + h
                ref = math.fsum(half_value(self.activations.data[row * self.width + k]) *
                                dequant_w[h * self.width + k] for k in range(self.width))
                assert abs(ref - dequant_ref[offset]) < 1e-4 * max(1, abs(ref))
                integer_sum = sum(self.qa[row * self.width + k] * self.qw[h * self.width + k]
                                  for k in range(self.width))
                base = half_value(struct.unpack("<H", struct.pack("<e",
                                  integer_sum * self.sa[row] * self.sw[h] / 127**2))[0])
                correction = math.fsum(half_value(self.sub[row * self.count + j]) *
                                       dequant_w[h * self.width + self.columns[j]]
                                       for j in range(self.count))
                expected = struct.unpack("<e", struct.pack("<e", base + correction))[0]
                # C uses FP32 accumulation; allow two FP16 ulps near a rounding boundary.
                assert abs(mixed[offset] - expected) <= .002 * max(1, abs(expected))
        def error(output, reference):
            differences = [float(a) - float(b) for a, b in zip(output, reference)]
            denominator = math.fsum(float(value)**2 for value in reference)
            return {"max_absolute": max(map(abs, differences)),
                    "relative_l2": math.sqrt(math.fsum(value**2 for value in differences) /
                                              max(denominator, 1e-30))}
        return {"normal_vs_reconstructed_weights": error(normal, dequant_ref),
                "mixed_vs_reconstructed_weights": error(mixed, dequant_ref),
                "mixed_vs_original_weights": error(mixed, original_ref),
                "scalar_checks": "passed"}

    def close(self):
        self.activations.close()
        self.weights.close()

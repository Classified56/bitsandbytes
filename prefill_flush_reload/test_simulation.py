"""Deterministic semantic checks; cache latency is deliberately not asserted."""

import ctypes as ct
import struct
import unittest

from .measurement import classification
from .native import Native
from .simulation import Simulation, half_value


def half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native()

    @classmethod
    def tearDownClass(cls):
        cls.native.close()

    def test_prefill_column_union_and_threshold_boundary(self):
        """One crossing selects that feature for every token, including negatives."""
        sim = Simulation(self.native, 4, 64, 8, 42, 6, 3)
        try:
            for index in range(4 * 64):
                sim.activations.data[index] = half_bits(.5)
            sim.activations.data[1] = half_bits(6)
            sim.activations.data[64 + 33] = half_bits(-6)
            sim.activations.data[2 * 64 + 2] = half_bits(5.5)
            before = list(sim.activations.data)
            sim.count = self.native.quantize(sim.activations.data, sim.qa, sim.sa, sim.columns,
                                             sim.mask, 4, 64, 6)
            self.assertEqual(list(sim.columns)[:sim.count], [1, 33])
            self.assertEqual(list(sim.activations.data), before)
            for row in range(4):
                self.assertEqual(sim.qa[row * 64 + 1], 0)
                self.assertEqual(sim.qa[row * 64 + 33], 0)
            self.assertEqual(sim.sa[0], .5)
            self.assertEqual(sim.sa[2], 5.5)
            sim.gather()
            for row in range(4):
                for j, column in enumerate((1, 33)):
                    self.assertEqual(sim.sub[row * 2 + j], before[row * 64 + column])
            self.assertEqual(sim.touched_lines([1, 2]), {0, 2, 4, 6})
            self.assertEqual(sim.numerical_check()["scalar_checks"], "passed")
        finally:
            sim.close()

    def test_no_outliers_and_injected_numerical_path(self):
        sim = Simulation(self.native, 8, 128, 16, 9, 6, 2)
        try:
            self.assertEqual(sim.prepare(False), [])
            self.assertEqual(sim.count, 0)
            sim.gather()
            self.assertEqual(sim.numerical_check()["scalar_checks"], "passed")
            self.assertEqual(sim.prepare(True), sim.injected)
            sim.gather()
            result = sim.numerical_check()
            self.assertLess(result["mixed_vs_reconstructed_weights"]["relative_l2"],
                            result["normal_vs_reconstructed_weights"]["relative_l2"])
        finally:
            sim.close()

    def test_zero_rows_and_disabled_threshold(self):
        source = (ct.c_uint16 * 128)()
        source[0] = half_bits(8)
        q, scales = (ct.c_int8 * 128)(), (ct.c_float * 2)()
        columns, mask = (ct.c_size_t * 64)(), (ct.c_uint8 * 64)()
        count = self.native.quantize(source, q, scales, columns, mask, 2, 64, 0)
        self.assertEqual(count, 0)
        self.assertEqual(q[0], 127)
        self.assertEqual(list(q)[64:], [0] * 64)
        self.assertEqual(scales[1], 0)
        self.assertEqual(half_value(source[0]), 8)

    def test_sparse_metrics_do_not_reward_all_miss_predictions(self):
        result = classification([(True, 200)] + [(False, 200)] * 99, 100)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["false_positive_rate"], 0)
        self.assertIsNone(result["precision"])


if __name__ == "__main__":
    unittest.main()

"""Check row-limited acquisition and analysis without asserting cache timings."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .analyze import analyze_csv, row_ranges, write_analysis
from .measurement import observe, summarize


class FakeNative:
    def __init__(self):
        self.flushes = []
        self.probes = []

    def flush_lines(self, address, count):
        self.flushes.append((address, count))

    def probe_lines(self, data, order, times, count):
        self.probes.append(list(order))
        for line in order:
            times[line] = 30 if line % 4 == 0 else 200


class ReportingTests(unittest.TestCase):
    def simulation(self):
        native = FakeNative()
        sim = SimpleNamespace(native=native, tokens=8, width=128,
                              activations=SimpleNamespace(address=4096, data=None), gathers=0)
        def gather():
            sim.gathers += 1
        sim.gather = gather
        return sim

    def test_subset_only_probes_requested_rows_and_preserves_full_gather(self):
        sim, trace = self.simulation(), []
        timings = observe(sim, True, 2, 42, probe_stride=3, probe_rows=2,
                          row_start=3, trace=trace)
        self.assertTrue(all(set(trial) == set(range(12, 20)) for trial in timings))
        self.assertEqual(sim.gathers, 6)
        self.assertTrue(all(count == 32 for _, count in sim.native.flushes))
        self.assertEqual(len(trace), 16)
        result = summarize(sim, timings, {12, 16}, {12, 16}, 100)
        self.assertEqual(result['observed_rows'], [3, 4])
        self.assertEqual(result['tp'], 4)
        self.assertEqual(result['tn'], 12)

    def test_single_probe_windows_and_observed_flush_bounds(self):
        sim, trace = self.simulation(), []
        timings = observe(sim, True, 1, 42, probe_rows=1, row_start=4,
                          probe_mode='single', flush_scope='observed', trace=trace)
        self.assertEqual(set(timings[0]), set(range(16, 20)))
        self.assertEqual(sim.gathers, 4)
        self.assertEqual(sim.native.flushes, [(4096 + 16 * 64, 4)] * 4)
        self.assertTrue(all(len(order) == 1 for order in sim.native.probes))
        self.assertTrue(all(index == 0 and previous is None for _, _, _, index, previous in trace))
        with self.assertRaises(ValueError):
            observe(sim, False, 1, 42, probe_rows=2, row_start=7)

    def test_probe_order_diagnostic_uses_temporal_not_column_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lines.csv"
            with path.open("w", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(("case", "trial", "row", "column_start", "predicted_hit",
                                 "outlier_line", "expected_access", "probe_index", "previous_probe_line"))
                # Probe column group 64, then 0, then 32. The halo diagnostic
                # uses spatial coordinates; this diagnostic uses actual order.
                writer.writerow(("injected_gather", 0, 0, 64, 1, 1, 1, 0, ""))
                writer.writerow(("injected_gather", 0, 0, 0, 1, 0, 0, 1, 2))
                writer.writerow(("injected_gather", 0, 0, 32, 0, 0, 0, 2, 0))
            analysis = analyze_csv(path)
            entry = analysis["probe_order_diagnostic"][0]
            self.assertEqual(entry["condition"], "after_nearby_hit")
            self.assertEqual(entry["false_hits"], 1)
            self.assertEqual(entry["observations"], 2)
            self.assertEqual(entry["false_positive_rate"], .5)
            self.assertEqual(analysis["control_adjusted_groups"], [])

    def test_consensus_control_adjustment_and_row_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lines.csv'
            with path.open('w', newline='') as target:
                writer = csv.writer(target)
                writer.writerow(('case', 'trial', 'row', 'column_start', 'column_end_exclusive',
                                 'predicted_hit', 'outlier_line'))
                for case in ('no_outliers', 'injected_gather', 'injected_no_gather'):
                    for trial in range(5):
                        for row in range(3):
                            for start in (0, 32, 64):
                                hit = start == 64  # Repeatable background, removable by controls.
                                if case == 'injected_gather':
                                    hit |= (start == 0 or (start == 32 and trial < (1, 3, 0)[row]))
                                writer.writerow((case, trial, row, start, start + 32, int(hit),
                                                 int(case == 'injected_gather' and start == 0)))
            analysis = analyze_csv(path)
            self.assertEqual(analysis['detected_groups'], [0, 64])
            self.assertEqual(analysis['control_adjusted_groups'], [0])
            group = next(item for item in analysis['groups'] if item['column_start'] == 32)
            self.assertEqual(group['any_hit_rows'], [0, 1])
            self.assertEqual(group['consistent_hit_rows'], [1])
            self.assertEqual(group['rows'][1]['cases']['injected_gather']['hits'], 3)
            self.assertEqual(analyze_csv(path, row_start=1, probe_rows=1)['detected_groups'], [0, 32, 64])
            self.assertIsNone(analysis['spatial_offsets'][-1]['injected_gather']['hit_fraction'])
            self.assertEqual(analysis['probe_order_diagnostic'], [])
            write_analysis(analysis, folder)
            self.assertIn('0:32', (Path(folder) / 'report.md').read_text())
            self.assertTrue((Path(folder) / 'row_sections.csv').exists())
            self.assertTrue((Path(folder) / 'group_scores.csv').exists())
            with (Path(folder) / 'group_scores.csv').open(newline='') as source:
                scores = list(csv.DictReader(source))
            self.assertEqual(len(scores), 4)
            leading = next(row for row in scores if row['selector'] == 'control_adjusted'
                           and row['processing'] == 'leading_group')
            self.assertEqual(leading['exact_match'], 'True')
            self.assertEqual(leading['row_count'], '3')
            with self.assertRaises(ValueError):
                analyze_csv(path, row_start=9)
            self.assertEqual(row_ranges([0, 1, 3, 5, 6, 7]), '0-1,3,5-7')


if __name__ == '__main__':
    unittest.main()

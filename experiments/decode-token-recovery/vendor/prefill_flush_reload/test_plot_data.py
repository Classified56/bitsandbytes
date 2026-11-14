"""Check figure provenance, truth agreement, and comparable-run grouping."""

import json
from pathlib import Path
import tempfile
import unittest

from .plot_data import compatible_cohorts, discover_results, load_result


class PlotDataTests(unittest.TestCase):
    def make_result(self, folder, rows=(0, 1), seed=42, scope='observed'):
        folder.mkdir(parents=True)
        config = dict(tokens=8, intermediate=64, hidden=16, seed=seed, row_start=0,
                      probe_rows=len(rows), probe_mode='sweep', probe_stride=1,
                      flush_scope=scope, threshold=6, outliers=1, repeats=5)
        summary = {'configuration': config, 'cases': {'injected_gather': {'outlier_columns': [33]}}}
        groups = []
        for start in (0, 32):
            groups.append({'column_start': start, 'contains_outlier': start == 32,
                           'detected': True, 'control_adjusted_detected': start == 32,
                           'rows': [{'row': row, 'cases': {'injected_gather':
                                     {'hits': 5 if start == 32 else 2, 'observations': 5}}} for row in rows]})
        analysis = {'groups': groups, 'observed_rows': list(rows),
                    'rules': {'row_hit_fraction': .6, 'min_row_fraction': .5, 'min_control_delta': .5}}
        (folder / 'summary.json').write_text(json.dumps(summary))
        (folder / 'analysis.json').write_text(json.dumps(analysis))
        return folder

    def test_load_preserves_original_candidates_and_does_not_modify_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.make_result(Path(directory) / 'run')
            before = {path.name: path.read_bytes() for path in folder.iterdir()}
            result = load_result(folder)
            self.assertEqual(result['hit_fractions'], [[.4, 1], [.4, 1]])
            # Leading noise is not removed using ground truth during plotting.
            score = result['scores']['methods']['consensus']['leading_group']
            self.assertEqual(score['predicted_groups'], [0])
            self.assertEqual((score['tp'], score['fp'], score['fn']), (0, 1, 1))
            self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir()})

    def test_different_seeds_protocols_or_thresholds_never_share_a_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [load_result(self.make_result(root / 'one', rows=(0,))),
                       load_result(self.make_result(root / 'two')),
                       load_result(self.make_result(root / 'seed-change', seed=420)),
                       load_result(self.make_result(root / 'scope-change', scope='all'))]
            cohorts = compatible_cohorts(records)
            self.assertEqual(sorted(len(cohort) for cohort in cohorts), [1, 1, 2])
            matched = next(cohort for cohort in cohorts if len(cohort) == 2)
            self.assertEqual([record['row_count'] for record in matched], [1, 2])
            records[1]['metadata']['row_hit_fraction'] = .8
            self.assertEqual(len(compatible_cohorts(records)), 4)

    def test_truth_mismatch_and_missing_row_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.make_result(Path(directory) / 'run')
            summary = json.loads((folder / 'summary.json').read_text())
            summary['cases']['injected_gather']['outlier_columns'] = [0]
            (folder / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_result(folder)
            summary['cases']['injected_gather']['outlier_columns'] = [33]
            (folder / 'summary.json').write_text(json.dumps(summary))
            analysis = json.loads((folder / 'analysis.json').read_text())
            analysis['groups'][0]['rows'] = []
            (folder / 'analysis.json').write_text(json.dumps(analysis))
            with self.assertRaisesRegex(ValueError, 'Missing per-row'):
                load_result(folder)

    def test_discovery_deduplicates_inputs_and_excludes_figure_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_result(root / 'run')
            self.make_result(root / 'figures' / 'unrelated')
            found = discover_results([root, run, run / 'summary.json'], excluded=root / 'figures')
            self.assertEqual(found, [run.resolve()])


if __name__ == '__main__':
    unittest.main()

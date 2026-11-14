"""Check that the halo heuristic cannot use truth to hide wrong guesses."""

import unittest

from .scoring import candidate_runs, detection_scores, score_groups


class ScoringTests(unittest.TestCase):
    def test_consecutive_runs_keep_followers_and_do_not_bridge_gaps(self):
        runs = candidate_runs([160, 0, 32, 160, 96, 192])
        self.assertEqual([run['guessed_group'] for run in runs], [0, 96, 160])
        self.assertEqual([run['following_groups'] for run in runs], [[32], [], [192]])
        self.assertEqual(runs[-1]['column_end_exclusive'], 224)
        self.assertEqual(candidate_runs([]), [])

    def test_adjacent_real_outliers_are_scored_as_suppressed_misses(self):
        groups = [{'column_start': start, 'contains_outlier': start in (0, 32),
                   'detected': start in (0, 32, 64), 'control_adjusted_detected': start in (32, 64)}
                  for start in (0, 32, 64, 96)]
        result = detection_scores(groups)
        consensus = result['methods']['consensus']
        self.assertEqual(consensus['raw']['recall'], 1)
        self.assertEqual(consensus['leading_group']['predicted_groups'], [0])
        self.assertEqual(consensus['leading_group']['missed_groups'], [32])
        self.assertEqual(consensus['leading_group']['recall'], .5)
        self.assertFalse(consensus['leading_group']['exact_match'])
        self.assertEqual(consensus['candidate_runs'][0]['following_actual_groups'], [32])
        self.assertEqual(result['methods']['control_adjusted']['leading_group']['predicted_groups'], [32])

    def test_noise_before_true_group_does_not_get_corrected_using_truth(self):
        groups = [{'column_start': start, 'contains_outlier': start == 32,
                   'detected': start in (0, 32), 'control_adjusted_detected': False}
                  for start in (0, 32, 64)]
        score = detection_scores(groups)['methods']['consensus']['leading_group']
        self.assertEqual(score['predicted_groups'], [0])
        self.assertEqual((score['tp'], score['fp'], score['fn']), (0, 1, 1))
        self.assertEqual(score['f1'], 0)

    def test_set_metrics_and_empty_predictions(self):
        result = score_groups([0, 32, 64], [0, 64, 96], [0, 32, 64, 96, 128])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['tn']), (2, 1, 1, 1))
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertAlmostEqual(result['recall'], 2 / 3)
        self.assertEqual(result['jaccard'], .5)
        empty = score_groups([], [0], [0, 32])
        self.assertIsNone(empty['precision'])
        self.assertEqual(empty['recall'], 0)
        self.assertEqual(empty['f1'], 0)
        with self.assertRaises(ValueError):
            score_groups([64], [0], [0, 32])


if __name__ == '__main__':
    unittest.main()

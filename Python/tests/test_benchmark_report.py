"""The sweep report's reduction of results.csv rows to one summary per method."""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import report  # noqa: E402


def row(method, error, timed_out=False, speed='1', run_error=''):
    return {'method': method, 'meanTrajectoryError': error, 'meanActualSpeed': speed,
            'timedOut': 'True' if timed_out else 'False', 'error': run_error}


class SummarizeTests(unittest.TestCase):
    def test_groups_by_method_in_order_of_appearance(self):
        rows = [row('Pfnn', '0.1'), row('Lmm', '0.4'), row('Pfnn', '0.3')]
        summaries = report.summarize(rows, 'sweep')
        self.assertEqual([s.method for s in summaries], ['Pfnn', 'Lmm'])
        self.assertAlmostEqual(summaries[0].values['meanTrajectoryError'], 0.2)
        self.assertEqual(summaries[0].runs, 2)

    def test_median(self):
        rows = [row('Mm', '0.1'), row('Mm', '0.2'), row('Mm', '9')]
        summary, = report.summarize(rows, 'sweep', stat='median')
        self.assertAlmostEqual(summary.values['meanTrajectoryError'], 0.2)

    def test_blank_and_nan_cells_are_skipped(self):
        rows = [row('Mm', '0.5', speed=''), row('Mm', 'NaN', speed='2')]
        summary, = report.summarize(rows, 'sweep')
        self.assertAlmostEqual(summary.values['meanTrajectoryError'], 0.5)
        self.assertAlmostEqual(summary.values['meanActualSpeed'], 2.0)
        self.assertTrue(math.isnan(summary.values['footskatePerMeter']))

    def test_timeouts_are_counted_either_way_and_excluded_on_request(self):
        rows = [row('Lmm', '0.2'), row('Lmm', '1.0', timed_out=True)]
        kept, = report.summarize(rows, 'sweep')
        dropped, = report.summarize(rows, 'sweep', exclude_timeouts=True)
        self.assertAlmostEqual(kept.values['meanTrajectoryError'], 0.6)
        self.assertAlmostEqual(dropped.values['meanTrajectoryError'], 0.2)
        self.assertEqual((kept.timeouts, dropped.timeouts), (1, 1))
        self.assertEqual(dropped.runs, 2)

    def test_stopped_short_runs_are_counted_and_kept(self):
        rows = [row('Lmm', '0.2'), dict(row('Lmm', '0.4'), stoppedShort='true', remainingDistance='1.5')]
        summary, = report.summarize(rows, 'sweep', exclude_timeouts=True)
        self.assertEqual((summary.stopped, summary.timeouts), (1, 0))
        self.assertAlmostEqual(summary.values['meanTrajectoryError'], 0.3)
        self.assertAlmostEqual(summary.values['remainingDistance'], 1.5)

    def test_errors_are_counted(self):
        rows = [row('Mm', '0.1', run_error='spawn failed'), row('Mm', '0.1')]
        summary, = report.summarize(rows, 'sweep')
        self.assertEqual(summary.errors, 1)


def seeded(method, seed, error):
    return dict(row(method, error), seed=str(seed))


class SeedTests(unittest.TestCase):
    def test_paths_are_reduced_per_seed_then_seeds_are_averaged(self):
        # Seed 1 has three paths and seed 2 one, so a plain mean over rows would give 0.4.
        rows = [seeded('Pfnn', 1, '0.1'), seeded('Pfnn', 1, '0.1'), seeded('Pfnn', 1, '0.1'),
                seeded('Pfnn', 2, '1.3')]
        summary, = report.summarize(rows, 'sweep')
        self.assertEqual(summary.seeds, 2)
        self.assertAlmostEqual(summary.values['meanTrajectoryError'], 0.7)

    def test_interval_is_t_over_per_seed_values(self):
        rows = [seeded('Lmm', 42, '1'), seeded('Lmm', 43, '2'), seeded('Lmm', 44, '3')]
        summary, = report.summarize(rows, 'sweep')
        # stdev 1 over three seeds: t(0.975, 2) / sqrt(3).
        self.assertAlmostEqual(summary.values['meanTrajectoryError'], 2.0)
        self.assertAlmostEqual(summary.ci['meanTrajectoryError'], 4.303 / math.sqrt(3), places=6)

    def test_one_seed_has_no_interval(self):
        summary, = report.summarize([seeded('Lmm', 42, '1'), seeded('Lmm', 42, '3')], 'sweep')
        self.assertEqual(summary.seeds, 1)
        self.assertAlmostEqual(summary.values['meanTrajectoryError'], 2.0)
        self.assertTrue(math.isnan(summary.ci['meanTrajectoryError']))

    def test_unseeded_methods_in_a_seeded_sweep_reduce_as_before(self):
        rows = [row('Mm', '0.2'), row('Mm', '0.4'), seeded('Pfnn', 1, '1'), seeded('Pfnn', 2, '3')]
        mm, pfnn = report.summarize(rows, 'sweep')
        self.assertEqual((mm.seeds, mm.ci), (0, {}))
        self.assertAlmostEqual(mm.values['meanTrajectoryError'], 0.3)
        self.assertEqual(pfnn.seeds, 2)

    def test_rendering_adds_seeds_and_intervals_only_when_seeded(self):
        plain = report.summarize([row('Mm', '0.5')], 'sweep')
        self.assertNotIn('Seeds', report.render(plain, 'markdown'))

        rows = [row('Mm', '0.5'), seeded('Lmm', 42, '1'), seeded('Lmm', 43, '3')]
        summaries = report.summarize(rows, 'sweep')
        markdown = report.render(summaries, 'markdown')
        self.assertIn('| Seeds |', markdown)
        self.assertIn('| sweep | Lmm | 2 | 2.000 ± 12.7 |', markdown)
        self.assertIn('| sweep | Mm | 0 | 0.500 |', markdown)

        header, *_ = report.render(summaries, 'csv').split('\n')
        self.assertIn('Path error (m),Path error (m) ±95%', header)


class LabelTests(unittest.TestCase):
    def test_timestamp_is_stripped(self):
        path = os.path.join('Benchmarks', 'MixamoHoldenBenchmark_20260929_081158', 'results.csv')
        self.assertEqual(report.sweep_label(path), 'MixamoHoldenBenchmark')

    def test_bare_timestamp_folder_keeps_its_name(self):
        path = os.path.join('Benchmarks', '20260929_013734', 'results.csv')
        self.assertEqual(report.sweep_label(path), '20260929_013734')


class LoadTests(unittest.TestCase):
    def test_folder_or_csv_and_markdown_render(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, 'Demo_20260101_000000')
            os.makedirs(folder)
            with open(os.path.join(folder, 'results.csv'), 'w', newline='') as f:
                f.write('method,path,meanTrajectoryError,timedOut,error\n'
                        'Mm,Circle,0.25,False,\nMm,Oval,0.75,True,\n')
            from_folder = report.load(folder)
            from_csv = report.load(os.path.join(folder, 'results.csv'))
            self.assertEqual(from_folder, from_csv)

            text = report.render(from_folder, 'markdown')
            self.assertIn('| Demo | Mm | 0.500 |', text)
            self.assertIn('1/2', text)


if __name__ == '__main__':
    unittest.main()

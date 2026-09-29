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

    def test_errors_are_counted(self):
        rows = [row('Mm', '0.1', run_error='spawn failed'), row('Mm', '0.1')]
        summary, = report.summarize(rows, 'sweep')
        self.assertEqual(summary.errors, 1)


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

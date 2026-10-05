"""
Summarizes one or more benchmark sweeps as a table of per-method statistics.

A sweep (``MoSynth/Benchmark/Run Sweep``) writes one ``results.csv`` row per method and path. This
reduces those rows to one per method, so sweeps over different datasets or configs can be compared
side by side. Each sweep is labelled by its folder name with the timestamp stripped, e.g.
``MixamoHoldenBenchmark_20260929_081158`` reads as ``MixamoHoldenBenchmark``.

Usage, from the ``Python/`` folder::

    python -m benchmark.report ../Benchmarks/MixamoHoldenBenchmark_20260929_081158 \\
        ../Benchmarks/MixamoEdinburghLocoBenchmark_20260929_090448 --format markdown

Every path is weighted equally, whatever its length. Timed-out runs are kept unless
``--exclude-timeouts`` is given: they are runs the method could not finish, so dropping them
flatters it, but keeping them mixes partial laps into the averages. The timeout count is always
reported so the reader can tell which applies. A run that came to rest short of the end of an open
path is not a timeout: it is kept, counted under Stopped, and how far short it stopped is averaged
over the open paths as ``Short of end (m)``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from dataclasses import dataclass

# (results.csv column, table heading). The order is the table's column order.
METRICS = [
    ('meanTrajectoryError', 'Path error (m)'),
    ('meanHeadingErrorDeg', 'Heading error (deg)'),
    ('meanActualSpeed', 'Speed (m/s)'),
    ('meanVelocityError', 'Velocity error (m/s)'),
    ('footskatePerMeter', 'Footskate /m'),
    ('contactFraction', 'Contact'),
    ('rootJerkMean', 'Root jerk'),
    ('discontinuitiesPerSecond', 'Disc. /s'),
    ('applyMsMean', 'Cost (ms/tick)'),
    ('applyMsP95', 'Cost P95 (ms)'),
    ('remainingDistance', 'Short of end (m)'),
]

_TIMESTAMP_SUFFIX = re.compile(r'_?\d{8}_\d{6}$')


@dataclass
class MethodSummary:
    sweep: str
    method: str
    runs: int
    timeouts: int
    stopped: int
    errors: int
    values: dict[str, float]


def sweep_label(results_path: str) -> str:
    """The sweep's folder name without its timestamp; a bare timestamp folder keeps its name."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(results_path)))
    return _TIMESTAMP_SUFFIX.sub('', folder) or folder


def resolve_results(path: str) -> str:
    """Accepts either a sweep folder or its results.csv."""
    return os.path.join(path, 'results.csv') if os.path.isdir(path) else path


def _number(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _flag(row: dict, column: str) -> bool:
    return row.get(column, '').strip().lower() == 'true'


def _timed_out(row: dict) -> bool:
    return _flag(row, 'timedOut')


def summarize(rows: list[dict], sweep: str, stat: str = 'mean',
              exclude_timeouts: bool = False) -> list[MethodSummary]:
    """One summary per method, in the order methods first appear in ``rows``."""
    reduce = statistics.fmean if stat == 'mean' else statistics.median
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row['method'], []).append(row)

    summaries = []
    for method, method_rows in by_method.items():
        kept = [r for r in method_rows if not (exclude_timeouts and _timed_out(r))]
        values = {}
        for column, _ in METRICS:
            numbers = [v for v in (_number(r.get(column)) for r in kept) if not math.isnan(v)]
            values[column] = reduce(numbers) if numbers else math.nan
        summaries.append(MethodSummary(
            sweep=sweep, method=method, runs=len(method_rows),
            timeouts=sum(_timed_out(r) for r in method_rows),
            stopped=sum(_flag(r, 'stoppedShort') for r in method_rows),
            errors=sum(bool(r.get('error', '').strip()) for r in method_rows),
            values=values))
    return summaries


def load(path: str, stat: str = 'mean', exclude_timeouts: bool = False) -> list[MethodSummary]:
    results_path = resolve_results(path)
    with open(results_path, newline='') as f:
        rows = list(csv.DictReader(f))
    return summarize(rows, sweep_label(results_path), stat, exclude_timeouts)


def _format_value(value: float) -> str:
    if math.isnan(value):
        return '-'
    return f'{value:.3f}' if abs(value) < 10 else f'{value:.1f}'


def table(summaries: list[MethodSummary]) -> tuple[list[str], list[list[str]]]:
    header = ['Sweep', 'Method'] + [title for _, title in METRICS] + ['Timeouts', 'Stopped', 'Errors']
    body = [[s.sweep, s.method]
            + [_format_value(s.values[column]) for column, _ in METRICS]
            + [f'{s.timeouts}/{s.runs}', f'{s.stopped}/{s.runs}', str(s.errors)]
            for s in summaries]
    return header, body


def render(summaries: list[MethodSummary], fmt: str) -> str:
    header, body = table(summaries)
    if fmt == 'markdown':
        lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        lines += ['| ' + ' | '.join(row) + ' |' for row in body]
        return '\n'.join(lines)
    if fmt == 'csv':
        return '\n'.join(','.join(row) for row in [header] + body)

    widths = [max(len(row[i]) for row in [header] + body) for i in range(len(header))]
    return '\n'.join('  '.join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip()
                     for row in [header] + body)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    parser.add_argument('sweeps', nargs='+', help='sweep folders, or their results.csv files')
    parser.add_argument('--stat', choices=['mean', 'median'], default='mean',
                        help='how each metric is reduced over a method\'s paths')
    parser.add_argument('--exclude-timeouts', action='store_true',
                        help='leave timed-out runs out of the statistics (they are still counted)')
    parser.add_argument('--format', choices=['text', 'markdown', 'csv'], default='text')
    args = parser.parse_args(argv)

    summaries = []
    for sweep in args.sweeps:
        summaries += load(sweep, args.stat, args.exclude_timeouts)
    print(render(summaries, args.format))


if __name__ == '__main__':
    main()

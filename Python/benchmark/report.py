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

**Seeds.** A sweep that ran a learned method once per training seed writes a ``seed`` column. Such a
method's runs are reduced over paths separately for each seed (``--stat`` applies here), and those
per-seed values are then averaged, so every seed weighs the same. The spread is a 95% confidence
interval from Student's t over the per-seed values, ``t(0.975, n - 1) * s / sqrt(n)`` with ``s`` the
sample standard deviation over ``n`` seeds: the seeds are the independent draws, while the paths of
one seed share a trained model and are not. It is shown as ``mean ± half-width`` and needs at least
two seeds. A method with no seed (motion matching), and any sweep without the column, is reduced
exactly as before.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from dataclasses import dataclass, field

# (results.csv column, table heading). The order is the table's column order.
METRICS = [
    ('meanTrajectoryError', 'Path error (m)'),
    ('meanHeadingErrorDeg', 'Heading error (deg)'),
    ('meanActualSpeed', 'Speed (m/s)'),
    ('meanVelocityError', 'Velocity error (m/s)'),
    ('footskatePerMeter', 'Footskate /m'),
    ('contactFraction', 'Contact'),
    ('flagFootskatePerMeter', 'Flag footskate /m'),
    ('contactFlagAgreement', 'Flag agreement'),
    ('flagOnsetHeightError', 'Flag onset height (m)'),
    ('rootJerkMean', 'Root jerk'),
    ('rootVerticalJerkMean', 'Vert. jerk'),
    ('discontinuitiesPerSecond', 'Disc. /s'),
    ('applyMsMean', 'Cost (ms/tick)'),
    ('applyMsP95', 'Cost P95 (ms)'),
    ('remainingDistance', 'Short of end (m)'),
]

_TIMESTAMP_SUFFIX = re.compile(r'_?\d{8}_\d{6}$')

# Two-sided 95% quantiles of Student's t, by degrees of freedom; past the table the normal 1.96.
_T_975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
          2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
          2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


@dataclass
class MethodSummary:
    sweep: str
    method: str
    runs: int
    timeouts: int
    stopped: int
    errors: int
    values: dict[str, float]
    # Distinct training seeds the runs came from; 0 when the method was not run per seed.
    seeds: int = 0
    # 95% confidence half-width across seeds, per metric; NaN below two seeds.
    ci: dict[str, float] = field(default_factory=dict)


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


def t_quantile_975(degrees_of_freedom: int) -> float:
    """The two-sided 95% critical value of Student's t."""
    if degrees_of_freedom < 1:
        return math.nan
    return _T_975[degrees_of_freedom - 1] if degrees_of_freedom <= len(_T_975) else 1.96


def confidence_interval(values: list[float]) -> tuple[float, float]:
    """``(mean, 95% half-width)`` of independent draws; the half-width is NaN below two."""
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan
    half_width = t_quantile_975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half_width


def _seed(row: dict) -> str:
    return (row.get('seed') or '').strip()


def _reduce_metrics(rows: list[dict], reduce) -> dict[str, float]:
    values = {}
    for column, _ in METRICS:
        numbers = [v for v in (_number(r.get(column)) for r in rows) if not math.isnan(v)]
        values[column] = reduce(numbers) if numbers else math.nan
    return values


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
        seeds = {_seed(r) for r in method_rows} - {''}
        ci = {}
        if seeds:
            by_seed: dict[str, list[dict]] = {}
            for r in kept:
                by_seed.setdefault(_seed(r), []).append(r)
            per_seed = [_reduce_metrics(seed_rows, reduce) for seed_rows in by_seed.values()]
            values = {}
            for column, _ in METRICS:
                draws = [v[column] for v in per_seed if not math.isnan(v[column])]
                values[column], ci[column] = confidence_interval(draws)
        else:
            values = _reduce_metrics(kept, reduce)
        summaries.append(MethodSummary(
            sweep=sweep, method=method, runs=len(method_rows),
            timeouts=sum(_timed_out(r) for r in method_rows),
            stopped=sum(_flag(r, 'stoppedShort') for r in method_rows),
            errors=sum(bool(r.get('error', '').strip()) for r in method_rows),
            values=values, seeds=len(seeds), ci=ci))
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


def _format_with_ci(summary: MethodSummary, column: str) -> str:
    value = _format_value(summary.values[column])
    half_width = summary.ci.get(column, math.nan)
    return value if math.isnan(half_width) else f'{value} ± {_format_value(half_width)}'


def table(summaries: list[MethodSummary],
          separate_ci: bool = False) -> tuple[list[str], list[list[str]]]:
    """
    Header and rows. A Seeds column appears only when some method was run per seed; its intervals
    are written into the metric's cell, or into a column of their own with ``separate_ci``.
    """
    seeded = any(s.seeds for s in summaries)
    header = ['Sweep', 'Method'] + (['Seeds'] if seeded else [])
    for _, title in METRICS:
        header += [title, f'{title} ±95%'] if seeded and separate_ci else [title]
    header += ['Timeouts', 'Stopped', 'Errors']

    body = []
    for s in summaries:
        cells = [s.sweep, s.method] + ([str(s.seeds)] if seeded else [])
        for column, _ in METRICS:
            if seeded and separate_ci:
                cells += [_format_value(s.values[column]), _format_value(s.ci.get(column, math.nan))]
            else:
                cells.append(_format_with_ci(s, column))
        cells += [f'{s.timeouts}/{s.runs}', f'{s.stopped}/{s.runs}', str(s.errors)]
        body.append(cells)
    return header, body


def render(summaries: list[MethodSummary], fmt: str) -> str:
    header, body = table(summaries, separate_ci=fmt == 'csv')
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
                        help='how each metric is reduced over a method\'s paths (per seed, when it has seeds)')
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

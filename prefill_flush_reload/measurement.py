"""Control the observation window and report cache-line classification.

Execution is intentionally staged in one process: returning from each native
call is the synchronization point. There are no independently drifting slots.
The probe runs before matrix multiplication or any numerical verification.
"""

import ctypes as ct
import mmap
import random
import statistics

from flush_reload_test import balanced_bits, choose_threshold, primitive_samples


def calibrate(native, seed, samples=2000):
    """Use a separate page and independent holdout samples, never case labels."""
    rng = random.Random(seed)
    with mmap.mmap(-1, mmap.PAGESIZE) as mapping:
        mapping[0] = 1
        address = ct.addressof(ct.c_uint8.from_buffer(mapping))
        primitive_samples(native, address, balanced_bits(100, rng))
        training = primitive_samples(native, address, balanced_bits(samples, rng))
        validation = primitive_samples(native, address, balanced_bits(samples, rng))
    threshold = choose_threshold(training)
    return threshold, classification(validation, threshold)


def classification(samples, threshold):
    """Expose errors explicitly; sparse all-zero predictions must not look good."""
    positives = sum(bool(label) for label, _ in samples)
    negatives = len(samples) - positives
    tp = sum(bool(label) and time < threshold for label, time in samples)
    fp = sum(not label and time < threshold for label, time in samples)
    fn, tn = positives - tp, negatives - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / positives if positives else None,
            "false_positive_rate": fp / negatives if negatives else None,
            "balanced_accuracy": .5 * (tp / positives + tn / negatives)
            if positives and negatives else None}


def observe(simulation, gather, repeats, seed, probe_stride=4, probe_rows=None,
            row_start=0, probe_mode="sweep", flush_scope="all", trace=None):
    """Keep the full prefill/gather, observing only the chosen token rows.

    Sweeps space their probes to reduce interference. Single mode performs a
    fresh flush/gather for EVERY target line, so no earlier reload in that
    window can fetch its neighbors. It is slower and still does not prevent
    prefetch triggered by the gather itself or learned across repetitions.

    Non-observed rows have no latency entry: they must not be treated as misses.
    Optional trace records preserve actual probe order for later diagnostics.
    """
    native = simulation.native
    lines_per_row = simulation.width // 32
    probe_rows = simulation.tokens - row_start if probe_rows is None else probe_rows
    if not 0 <= row_start < simulation.tokens or not 1 <= probe_rows <= simulation.tokens - row_start:
        raise ValueError("Observed row range must be inside the prefill matrix")
    if probe_mode not in ("sweep", "single") or flush_scope not in ("all", "observed"):
        raise ValueError("Unknown probe mode or flush scope")
    if probe_stride < 1:
        raise ValueError("Probe stride must be positive")
    first, last = row_start * lines_per_row, (row_start + probe_rows) * lines_per_row
    lines = simulation.tokens * lines_per_row
    times = (ct.c_uint64 * lines)()
    rng = random.Random(seed)
    results = []
    for trial in range(repeats):
        combined = {}
        if probe_mode == "single":
            order = list(range(first, last))
            rng.shuffle(order)
            sweeps = [[line] for line in order]
        else:
            sweeps = [list(range(first + (residue - first) % probe_stride, last, probe_stride))
                      for residue in range(probe_stride)]
            for order in sweeps:
                rng.shuffle(order)
        for sweep, order in enumerate(sweeps):
            if not order:
                continue
            order_array = (ct.c_size_t * len(order))(*order)
            ############# FLUSH + RELOAD #############
            address = simulation.activations.address
            if flush_scope == "observed":
                native.flush_lines(address + first * 64, last - first)
            else:
                native.flush_lines(address, lines)
            if gather:
                simulation.gather()
            native.probe_lines(simulation.activations.data, order_array, times, len(order))
            for index, line in enumerate(order):
                combined[line] = times[line]
                if trace is not None:
                    trace.append((trial, line, sweep, index, order[index - 1] if index else None))
        results.append(combined)
    return results


def summarize(simulation, timings, accessed_lines, outlier_lines, threshold):
    """Report per-line errors and per-feature-section hit fractions across rows."""
    samples = [(line in accessed_lines, latency)
               for trial in timings for line, latency in trial.items()]
    result = classification(samples, threshold)
    distributions = {}
    for label, name in ((True, "expected_access"), (False, "expected_no_access")):
        values = sorted(time for truth, time in samples if truth == label)
        if values:
            distributions[name] = {"n": len(values), "p05": values[round(.05 * (len(values) - 1))],
                                   "median": statistics.median(values),
                                   "p95": values[round(.95 * (len(values) - 1))]}
    lines_per_row = simulation.width // 32
    sections = []
    observed_rows = sorted({line // lines_per_row for trial in timings for line in trial})
    for section in range(lines_per_row):
        hits = sum(trial[row * lines_per_row + section] < threshold
                   for trial in timings for row in observed_rows)
        sections.append({"column_start": section * 32, "column_end_exclusive": (section + 1) * 32,
                         "contains_outlier": section in {line % lines_per_row for line in outlier_lines},
                         "hit_fraction": hits / (len(timings) * len(observed_rows))})
    result.update({"latencies_tsc_ticks": distributions, "sections": sections,
                   "observed_rows": observed_rows})
    return result

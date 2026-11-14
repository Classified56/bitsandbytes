#!/usr/bin/env python3
"""Verify Flush+Reload on a private test mapping with a cooperative sender.

Requires x86-64 Linux/macOS and a C compiler (CC, or cc). The helper is built
in a temporary directory; nothing is installed. Native reloads measure TSC
ticks, not nanoseconds or CPU core cycles. --timer python demonstrates the
additional noise from timing a ctypes load in Python (nanoseconds).

The primitive check and handshake check each have disjoint calibration and
validation samples. The handshake is correctness instrumentation, not a
measurement of an uncooperative application or channel throughput.
"""

import argparse
import csv
import ctypes
import mmap
import os
from pathlib import Path
import queue
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time


class FlushReload:
    def __init__(self, timer="native"):
        if sys.platform not in ("linux", "darwin") or os.uname().machine not in ("x86_64", "AMD64"):
            raise RuntimeError("This test requires x86-64 Linux or macOS with RDTSCP support")
        self.timer = timer
        self.unit = "TSC ticks" if timer == "native" else "ns"
        self.threshold = None
        self._build = tempfile.TemporaryDirectory(prefix="flush-reload-")
        library = Path(self._build.name) / "helper.so"
        command = [*shlex.split(os.environ.get("CC", "cc")), "-O2", "-shared", "-fPIC", str(Path(__file__).with_name("flush_reload_native.c")), "-o", str(library)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        self._lib = ctypes.CDLL(str(library))
        for name in ("fr_flush", "fr_touch", "fr_reload"):
            function = getattr(self._lib, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_uint64 if name == "fr_reload" else None

    def flush(self, addr):
        self._lib.fr_flush(addr)

    def touch(self, addr):
        self._lib.fr_touch(addr)

    def reload(self, addr):
        if self.timer == "native":
            return self._lib.fr_reload(addr)
        start = time.perf_counter_ns()
        _ = ctypes.c_uint8.from_address(addr).value
        return time.perf_counter_ns() - start

    def probe(self, addr):
        if self.threshold is None:
            raise RuntimeError("Calibrate and set threshold before probing")
        return self.reload(addr) < self.threshold

    def close(self):
        self._build.cleanup()


def balanced_bits(count, rng):
    bits = [i % 2 for i in range(count)]
    rng.shuffle(bits)
    return bits


def primitive_samples(fr, addr, bits):
    """Both labels begin with a flush; a hit adds exactly one explicit read."""
    samples = []
    for bit in bits:
        fr.flush(addr)
        if bit:
            fr.touch(addr)
        samples.append((bit, fr.reload(addr)))
    return samples


def handshake_samples(fr, addr, bits, timeout):
    """READY -> optional read -> DONE -> reload, once per bit.

    Sender signals DONE for both labels. It cannot access the next bit until
    receiver has reloaded and flushed again. Events are outside the probe line.
    """
    ready = threading.Event()
    done = threading.Event()
    stop = threading.Event()
    errors = queue.Queue()

    def sender():
        try:
            for bit in bits:
                if not ready.wait(timeout):
                    raise TimeoutError("Sender timed out waiting for READY")
                ready.clear()
                if stop.is_set():
                    return
                if bit:
                    fr.touch(addr)
                done.set()
        except Exception as error:
            errors.put(error)
            stop.set()
            done.set()  # Wake receiver so it can propagate the actual error.

    worker = threading.Thread(target=sender, name="flush-reload-sender", daemon=True)
    samples = []
    worker.start()
    try:
        for bit in bits:
            fr.flush(addr)
            ready.set()
            if not done.wait(timeout):
                raise TimeoutError("Receiver timed out waiting for DONE")
            done.clear()
            if stop.is_set():
                raise errors.get_nowait()
            samples.append((bit, fr.reload(addr)))
    finally:
        stop.set()
        ready.set()
        worker.join(timeout)
        if worker.is_alive():
            raise TimeoutError("Sender did not shut down")
    if not errors.empty():
        raise errors.get_nowait()
    return samples


def choose_threshold(samples):
    """Minimize balanced error on calibration data only; hit iff time < cutoff."""
    hits = sum(bit for bit, _ in samples)
    misses = len(samples) - hits
    if not hits or not misses:
        raise ValueError("Calibration requires both hit and miss samples")
    # Start below every observation: all hits are false negatives.
    fn, fp = hits, 0
    best_error = 1.0
    best_threshold = 0
    best_gap = -1
    ordered = sorted(samples, key=lambda item: item[1])
    index = 0
    while index < len(ordered):
        latency = ordered[index][1]
        while index < len(ordered) and ordered[index][1] == latency:
            bit = ordered[index][0]
            fn -= bit
            fp += 1 - bit
            index += 1
        error = 0.5 * (fn / hits + fp / misses)
        # Prefer the middle of the widest equally good gap, rather than
        # hugging the slowest training hit and failing after a small drift.
        next_latency = ordered[index][1] if index < len(ordered) else latency + 1
        gap = next_latency - latency
        if error < best_error or (error == best_error and gap > best_gap):
            best_error = error
            best_gap = gap
            best_threshold = (latency + next_latency) // 2 + 1
    return best_threshold


def report(name, samples, threshold, unit):
    print(f"\n{name}: threshold {threshold} {unit} (hit if strictly below)")
    for bit, label in ((1, "access / hit"), (0, "no access / miss")):
        values = sorted(latency for truth, latency in samples if truth == bit)
        def percentile(fraction):
            return values[round((len(values) - 1) * fraction)]
        print(f"  {label}: n={len(values)}, p05={percentile(.05)}, "
              f"median={statistics.median(values):g}, p95={percentile(.95)} {unit}")
    fp = sum(not bit and latency < threshold for bit, latency in samples)
    fn = sum(bit and latency >= threshold for bit, latency in samples)
    positives = sum(bit for bit, _ in samples)
    negatives = len(samples) - positives
    balanced_accuracy = 1 - .5 * (fp / negatives + fn / positives)
    print(f"  False positives: {fp}/{negatives}; false negatives: {fn}/{positives}")
    print(f"  Accuracy: {(len(samples) - fp - fn) / len(samples):.2%}; "
          f"balanced accuracy: {balanced_accuracy:.2%}")
    return balanced_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000, help="samples per primitive phase")
    parser.add_argument("--bits", type=int, default=100, help="bits per handshake phase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=5, help="seconds per handshake wait")
    parser.add_argument("--timer", choices=("native", "python"), default="native")
    parser.add_argument("--min-accuracy", type=float, default=.90, help="validation pass criterion")
    parser.add_argument("--csv", type=Path, help="save all labeled raw timings")
    args = parser.parse_args()
    if args.samples < 20 or args.bits < 20 or args.timeout <= 0 or not 0 < args.min_accuracy <= 1:
        parser.error("samples/bits must be >=20, timeout positive, min-accuracy in (0, 1]")
    rng = random.Random(args.seed)
    fr = FlushReload(args.timer)
    rows = []
    try:
        # Dedicated, page-aligned allocation: unrelated Python objects cannot
        # accidentally occupy the target cache line. Pre-fault before calibration.
        with mmap.mmap(-1, mmap.PAGESIZE) as target:
            target[0] = 1
            addr = ctypes.addressof(ctypes.c_uint8.from_buffer(target))
            print(f"Timer: {args.timer}; units: {fr.unit}; seed: {args.seed}")
            print("Protocol: flush -> READY -> optional read -> DONE -> reload")
            primitive_samples(fr, addr, balanced_bits(100, rng))
            passed = True
            for mode, count in (("primitive", args.samples), ("handshake", args.bits)):
                phases = {}
                for phase in ("calibration", "validation"):
                    bits = balanced_bits(count, rng)
                    if mode == "primitive":
                        phases[phase] = primitive_samples(fr, addr, bits)
                    else:
                        phases[phase] = handshake_samples(fr, addr, bits, args.timeout)
                threshold = choose_threshold(phases["calibration"])
                for phase, samples in phases.items():
                    accuracy = report(f"{mode} {phase}", samples, threshold, fr.unit)
                    rows.extend((mode, phase, index, bit, latency, threshold,
                                 int(latency < threshold), fr.unit)
                                for index, (bit, latency) in enumerate(samples))
                    if phase == "validation":
                        passed = passed and accuracy >= args.min_accuracy
            if args.csv:
                with args.csv.open("w", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(("mode", "phase", "index", "sent_bit", "latency",
                                     "threshold", "received_bit", "unit"))
                    writer.writerows(rows)
                print(f"\nRaw samples: {args.csv}")
            print(f"\n{'PASS' if passed else 'FAIL'}: both held-out checks must reach "
                  f"{args.min_accuracy:.0%} balanced accuracy.")
            if not passed:
                print("Timing distributions overlap or drifted. Inspect the CSV; "
                      "a completed handshake alone does not prove reliable cache classification.")
            return 0 if passed else 1
    finally:
        fr.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        sys.exit(2)

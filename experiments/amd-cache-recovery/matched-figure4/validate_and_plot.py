"""Verify the preserved Figure 4 records, then plot without rerunning probes.

Requires matplotlib; the unmodified upstream analyzer is imported from --source.
All rates in JSON/CSV are fractions, and all group bounds are zero-based.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
PINNED = "58cc1cc78f708c3a70c2f6cd969baa5f835ef957"
CASES = ("no_outliers", "injected_gather", "injected_no_gather")
TRUTH = [192, 896, 1984, 2240]
SCHEDULE = ("case", "trial", "row", "column_start", "column_end_exclusive",
            "expected_access", "outlier_line", "sweep", "probe_index", "previous_probe_line")


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def score(tp, fp, fn, tn):
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None)


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values),
                sample_stdev=statistics.stdev(values) if len(values) > 1 else None)


def validate(name, directory, config, analyze_csv, reference=None):
    summary = read_json(directory / "summary.json")
    assert {k: v for k, v in summary["configuration"].items() if k != "output"} == config, name
    with (directory / "lines.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9120, (name, len(rows))
    expected_keys = {(case, t, r, c) for case in CASES for t in range(5)
                     for r in range(4) for c in range(0, 4864, 32)}
    keys = [(r["case"], int(r["trial"]), int(r["row"]), int(r["column_start"])) for r in rows]
    assert len(set(keys)) == len(keys) and set(keys) == expected_keys, name
    schedule = [[r[k] for k in SCHEDULE] for r in rows]
    schedule_hash = digest(json.dumps(schedule, separators=(",", ":")).encode())
    if reference is not None:
        assert schedule == reference["schedule"], name + ": probe schedule/labels differ"
        for case in CASES:
            for key in ("outlier_columns", "outlier_count", "observed_rows", "numerics"):
                assert summary["cases"][case].get(key) == reference["summary"]["cases"][case].get(key), (name, case, key)
    cutoff = summary["calibration"]["reload_threshold_tsc_ticks"]
    counts = {case: Counter() for case in CASES}
    for r in rows:
        case, start = r["case"], int(r["column_start"])
        pred, expected = int(r["predicted_hit"]), int(r["expected_access"])
        assert pred == int(int(r["latency_tsc_ticks"]) < cutoff), (name, r)
        assert expected == int(case == "injected_gather" and start in TRUTH), (name, r)
        assert int(r["outlier_line"]) == int(case != "no_outliers" and start in TRUTH), (name, r)
        counts[case][("tp" if expected else "fp") if pred else ("fn" if expected else "tn")] += 1
    raw = {}
    for case in CASES:
        raw[case] = score(*(counts[case][k] for k in ("tp", "fp", "fn", "tn")))
        for k, v in raw[case].items():
            assert v == summary["cases"][case][k], (name, case, k)
    analysis = analyze_csv(directory / "lines.csv", row_start=config["row_start"],
        probe_rows=config["probe_rows"], row_hit_fraction=config["row_hit_fraction"],
        min_row_fraction=config["min_row_fraction"], min_control_delta=config["min_control_delta"])
    saved = read_json(directory / "analysis.json")
    for field in ("observed_rows", "rules", "detected_groups", "control_adjusted_groups", "groups"):
        assert analysis[field] == saved[field], (name, field)
    if "group_detection" in saved:
        assert analysis["group_detection"] == saved["group_detection"], name
    # Independently recompute consensus/filter from raw CSV hits.
    hits = Counter((int(r["row"]), int(r["column_start"])) for r in rows
                   if r["case"] == "injected_gather" and int(r["predicted_hit"]))
    detected = [c for c in range(0, 4864, 32) if sum(hits[r, c] >= 3 for r in range(4)) >= 2]
    filtered = [c for c in detected if c - 32 not in detected]
    method = analysis["group_detection"]["methods"]["consensus"]
    for values, stage in ((detected, "raw"), (filtered, "leading_group")):
        result = method[stage]
        assert values == result["predicted_groups"], (name, stage)
        truth, pred = set(TRUTH), set(values)
        independent = score(len(truth & pred), len(pred - truth), len(truth - pred),
                            152 - len(truth | pred))
        for key in ("tp", "fp", "fn", "tn", "precision", "recall"):
            assert independent[key] == result[key], (name, stage, key)
    output = dict(name=name, raw_probe=raw, calibration=summary["calibration"],
        consensus=method["raw"], filtered=method["leading_group"],
        control_adjusted=analysis["group_detection"]["methods"]["control_adjusted"],
        spatial_offsets=analysis["spatial_offsets"],
        validation=dict(records=len(rows), configuration_match=True,
            probe_schedule_and_labels_match_intel=True, schedule_sha256=schedule_hash,
            classification_recomputed=True, consensus_and_filter_independently_recomputed=True,
            numerical_checks_match_intel=True,
            input_sha256={p.name: digest(p.read_bytes()) for p in directory.iterdir()
                          if p.is_file() and p.suffix in (".json", ".csv")}),
        cache_check_passed=summary["cache_check_passed"])
    return output, dict(summary=summary, schedule=schedule), analysis


def plot_comparison(data, plt):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.6), sharex=True)
    fig.subplots_adjust(left=.19, right=.97, top=.84, bottom=.15, hspace=.55)
    fig.text(.04, .96, "Figure 4 replication: Intel and AMD", fontsize=24, weight="bold", va="top")
    fig.text(.04, .91, "64 × 4864 FP16  |  observed rows 0–3  |  seed 42  |  5 repeats  |  sweep / observed  |  stride 5",
             fontsize=11, color="#536177")
    for ax, record, title in zip(axes, [data["intel"], data["amd"][0]],
            ["Intel Core i7-10700 — original Figure 4 record", "AMD Ryzen 7 8700G — first acquisition (run-01)"]):
        sets = [TRUTH, record["consensus"]["predicted_groups"], record["filtered"]["predicted_groups"]]
        colors = ["#df7200", "#007db5", "#009d79"]
        for y, groups, color in zip([2, 1, 0], sets, colors):
            ax.broken_barh([(0, 4864)], (y - .24, .48), facecolors="#eff1f4")
            ax.broken_barh([(g, 32) for g in groups], (y - .24, .48), facecolors=color)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=25)
        ax.text(0, 1.04, f"Consensus: TP {record['consensus']['tp']}, FP {record['consensus']['fp']}  ·  "
                f"Filtered: TP {record['filtered']['tp']}, FP {record['filtered']['fp']}",
                fontsize=11, color="#536177", transform=ax.transAxes)
        ax.set_yticks([2, 1, 0], ["Ground truth", "Raw groups\n(after consensus)", "Filtered groups"], fontsize=12)
        ax.set_ylim(-.45, 2.45)
        ax.set_xlim(0, 4864)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 4864])
        ax.tick_params(axis="y", length=0, pad=12)
        ax.tick_params(axis="x", labelsize=11, length=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Activation column (32 FP16 columns per cache-line group)", fontsize=12, labelpad=10)
    fig.text(.04, .065, "Filter: retain the first group in each strictly consecutive candidate run. Each colored mark spans 32 columns.",
             fontsize=10, color="#536177")
    fig.text(.04, .035, "AMD: 10/10 acquisitions recover the exact group set after filtering; 5/10 have one false group before filtering.",
             fontsize=10, color="#536177")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(BASE / f"figure4-intel-amd.{suffix}", dpi=200, facecolor="white")
        if suffix == "svg":
            path = BASE / f"figure4-intel-amd.{suffix}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_repeats(data, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=.07, right=.98, top=.77, bottom=.22, wspace=.28)
    fig.text(.035, .96, "Same settings, ten AMD acquisitions", fontsize=21, weight="bold", va="top")
    fig.text(.035, .875, "Each acquisition includes five probe repeats. Run 01 was designated primary before execution.",
             fontsize=11, color="#536177")
    x = list(range(1, 11))
    raw = [100 * r["raw_probe"]["injected_gather"]["false_positive_rate"] for r in data["amd"]]
    intel = 100 * data["intel"]["raw_probe"]["injected_gather"]["false_positive_rate"]
    axes[0].plot(x, raw, "o-", color="#007db5", label="AMD measured")
    axes[0].axhline(intel, color="#df7200", ls="--", label="Intel original record")
    axes[0].set_ylabel("Probe false-positive rate (%)")
    axes[0].set_ylim(0, 11)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1].plot(x, [r["consensus"]["fp"] for r in data["amd"]], "o-", color="#007db5", label="AMD after consensus")
    axes[1].plot(x, [r["filtered"]["fp"] for r in data["amd"]], "s-", color="#009d79", label="AMD after filtering")
    axes[1].axhline(data["intel"]["consensus"]["fp"], color="#df7200", ls="--", label="Intel after consensus")
    axes[1].set_ylabel("False-positive groups")
    axes[1].set_ylim(-.3, 5.6)
    axes[1].set_yticks(range(6))
    axes[1].legend(frameon=False, fontsize=9, loc="center right")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xlabel("AMD acquisition")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.035, .07, "Recall is 100% at every stage in both datasets. Intel also has zero false groups after filtering.",
             fontsize=10, color="#536177")
    fig.text(.035, .025, "These are repeated runs of one fixed input/seed, not ten different inputs or a matched ten-run Intel sample.",
             fontsize=10, color="#536177")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(BASE / f"figure4-amd-repeatability.{suffix}", dpi=200, facecolor="white")
        if suffix == "svg":
            path = BASE / f"figure4-amd-repeatability.{suffix}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def write_report(data):
    lines = [
        "图 4 在 AMD 上的同设置复现实验与 Intel 原始记录比较",
        "日期：2026-11-09",
        "",
        "结论：AMD Ryzen 7 8700G 的 10 次采集中，全部在相同过滤后准确恢复 4 个真实缓存行分组。",
        "其中 5 次仅用行间投票即可恢复，另 5 次仍有 1 个相邻误报；相同过滤均消除了该误报。",
        "AMD 第一次采集的逐次探测 FPR 高于 Intel 原记录，但投票后误报分组更少。",
        "这验证了本合成实验在 AMD 上可以复现，不能证明 AMD 总体更好或预取器是唯一成因。",
        "",
        "一、基线与代码",
        f"上游：https://github.com/Classified56/bitsandbytes/tree/{PINNED}",
        "分支：shadow_outlier；固定提交见上述链接。AMD 测量代码未修改，17 个源文件逐一 SHA256 核对。",
        "Intel 基线：prefill_flush_reload/prefill-four-rows（英文 four，seed 42）。",
        "不是 prefill-4-rows（数字 4，seed 420），也不是论文表 III 的结果。",
        "论文图 4：四行、seed 42、sweep / observed；分组位置与上述记录吻合。",
        "Intel 原始文件逐字节核对上游保存文件。原始文件保存在 intel-original/；重新计分另存 intel-reanalysis/。",
        "Intel 平台信息来自论文：i7-10700，Ubuntu 22.04.5，16 GB DDR4。没有重新采集 Intel 数据。",
        "AMD 实测：Ryzen 7 8700G，Ubuntu 24.04.4，Linux 7.0.0-38-generic，Python 3.12.3，GCC 13.3.0。",
        "AMD 环境、亲和性、编译器、负载、每次运行命令、开始时间、退出码均在 amd/manifest.json。",
        "",
        "二、相同参数与测量过程",
        json.dumps(data["configuration"], indent=2, ensure_ascii=False),
        "",
        "64 × 4864 FP16 激活；4 个异常列为 204、912、2006、2253；隐藏维度为 896。",
        "对应 32 列分组的起点为 192、896、1984、2240；每行 152 个候选分组。",
        "每次仍执行全部 64 行的 gather，只 flush / reload 第 0–3 行。",
        "每种 case 有 5 次 trial；每次 trial 按 stride 5 分成 5 个交错 sweep，组内按固定 seed 打乱。",
        "case 顺序：no_outliers、injected_gather、injected_no_gather。",
        "每个 case 3040 条记录，每次采集共 9120 条；这不是单次 gather 的同时快照。",
        "flush、gather、reload 在同一进程中按阶段执行；是 CPU 上的受控合成实验。",
        "计时区间在矩阵乘法和数值校验之前，与原代码一致；这里没有执行 NPU / 实际 LLM 推理。",
        "native 编译参数：cc -O3 -mf16c -shared -fPIC kernels.c flush_reload_native.c -lm。",
        "",
        "异常激活阈值 threshold=32 固定。缓存命中计时阈值是另一个参数，由原代码在独立页上标定：",
        "100 次预热、2000 个训练样本选阈值、2000 个独立验证样本检查分类；不使用本实验异常列标签调阈值。",
        "命中判据为 latency_tsc_ticks < reload_threshold_tsc_ticks。",
        "Intel 阈值 119 TSC ticks；AMD 主采集阈值 253；AMD 十次范围 106–253。",
        "不同数值是相同标定算法的输出，不是修改实验设置；TSC ticks 不等同于 CPU 核心周期或纳秒。",
        "没有额外固定核心、调整 CPU governor、关闭预取器或修改系统设置。",
        "",
        "三、三个统计层级（不要把图中 Raw groups 当成逐次探测结果）",
        "1. Raw probe：每次 reload 的二分类，尚未投票；precision = TP/(TP+FP)，FPR = FP/(FP+TN)。",
        "2. Consensus：同一行 5 次中至少 3 次命中；4 行中至少 2 行满足，判该分组为候选。",
        "3. Filtered：每段严格按 32 列相邻的候选分组，只保留起点最小的一个。",
        "主对比使用 consensus selector，与论文图 4 一致；不是 control_adjusted selector。",
        "对照调整结果仍完整保留在 JSON 和原始分析文件中。",
        "标签仅用于评分，不用于过滤决策。该启发式可能删掉真实相邻异常分组，本输入没有这种情况。",
        "",
        "四、主对比：Intel 原记录 vs 事先指定的 AMD 第一次采集",
        "平台   阶段          TP    FP    FN    TN    Precision    Recall    FPR",
    ]
    for name, record in (("Intel", data["intel"]), ("AMD", data["amd"][0])):
        for label, s in (("Raw probe", record["raw_probe"]["injected_gather"]),
                         ("Consensus", record["consensus"]), ("Filtered", record["filtered"])):
            fpr = s["fp"] / (s["fp"] + s["tn"])
            lines.append(f"{name:<6} {label:<12} {s['tp']:>4} {s['fp']:>5} {s['fn']:>5} {s['tn']:>5} "
                         f"{s['precision']:>11.4%} {s['recall']:>9.4%} {fpr:>9.4%}")
    lines += ["", "分组起点："]
    for name, record in (("Intel", data["intel"]), ("AMD run-01", data["amd"][0])):
        lines += [f"{name} consensus: {record['consensus']['predicted_groups']}",
                  f"{name} filtered:  {record['filtered']['predicted_groups']}"]
    lines += ["", "对照 case 的逐次探测 FPR：", "平台   no_outliers   injected_no_gather"]
    for name, record in (("Intel", data["intel"]), ("AMD", data["amd"][0])):
        lines.append(f"{name:<6} {record['raw_probe']['no_outliers']['false_positive_rate']:.4%}       "
                     f"{record['raw_probe']['injected_no_gather']['false_positive_rate']:.4%}")
    lines += ["", "五、AMD 十次采集（全部保留，无挑选或替换）",
              "采集     阈值(ticks)  Probe precision  Probe FPR   Consensus FP  Filtered FP"]
    for r in data["amd"]:
        s = r["raw_probe"]["injected_gather"]
        lines.append(f"{r['name']:<8} {r['calibration']['reload_threshold_tsc_ticks']:>10} "
                     f"{s['precision']:>16.4%} {s['false_positive_rate']:>10.4%} "
                     f"{r['consensus']['fp']:>14} {r['filtered']['fp']:>12}")
    a = data["amd_aggregate"]
    lines += ["", "每次每个阶段 recall 均为 100%；过滤后 precision 均为 100%。",
              f"Probe FPR：均值 {a['false_positive_rate']['mean']:.4%}；"
              f"中位数 {a['false_positive_rate']['median']:.4%}；"
              f"范围 {a['false_positive_rate']['minimum']:.4%}–{a['false_positive_rate']['maximum']:.4%}。",
              "这些是短时间内对同一 seed / 输入的重复测量，不是十种输入，也没有对称的 Intel 十次测量。",
              "与 Intel 原记录相比，AMD 多数采集的逐次误报更多，但通过行间投票的误报分组更少。",
              "因此不能只用一个层级的指标概括平台优劣；也没有因果实验证明差异来自特定预取器。",
              "", "六、已完成的核对与限制",
              "共检查 100320 条记录：9120 条 Intel + 91200 条 AMD。",
              "每次的配置、case/trial/row/group 覆盖、完整探测顺序、期望标签均与 Intel 基线一致。",
              "从延迟重新计算命中；从 CSV 重新计算混淆矩阵；独立重算投票和过滤；均与保存结果吻合。",
              "AMD 的 outlier 列、数值验证指标与 Intel 记录一致；全部缓存检测和数值检测通过。",
              "证据保存在 comparison.json 的 validation 和 amd/manifest.json 中。",
              "历史 Intel 的编译器版本、原生二进制、系统调优细节没有记录；不能保证二进制及全部系统条件相同。",
              "因此准确说法是：固定现有上游实现、匹配原记录全部实验参数和探测顺序的跨平台复现。",
              "不能声称完成了只更换 CPU、其余系统因素完全受控的实验。",
              "Overleaf 没有修改。", "", "七、文件与复现",
              "figure4-intel-amd.{png,pdf,svg}：与论文图 4 相同含义的三条分组带，Intel / AMD 对照。",
              "figure4-amd-repeatability.{png,pdf,svg}：十次波动和投票前后误报。",
              "comparison.csv：55 行指标记录；比例用 0–1 表示。comparison.json：完整计分与验证。",
              "amd/：原始 CSV、JSON、日志及环境；intel-original/：原始 Intel 基线；",
              "intel-reanalysis/：用同一上游分析函数对 Intel CSV 重新计分的结果。",
              "本归档不包含论文源文件；原始论文没有修改。",
              "", "在 Linux x86-64 上重新采集（需 F16C、RDTSCP、64-byte cache line；输出目录必须不存在）：",
              "  git clone https://github.com/Classified56/bitsandbytes.git upstream",
              f"  git -C upstream checkout --detach {PINNED}",
              "  python3 run_matched.py --source upstream --output amd-new --acquisitions 10",
              "", "仅重算本次归档数据并重画图（不产生新的缓存测量）：",
              "  python3 -m pip install matplotlib",
              "  python3 -B validate_and_plot.py --source upstream",
              "分析脚本读取同目录 amd/ 和 intel-original/；若使用新采集数据，请在新目录内组织同样结构。",
              "重新运行会重写派生图表和 comparison 文件，但不会改写原始 amd/、intel-original/ 或上游代码。"]
    (BASE / "REPORT.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    hashes = read_json(BASE / "source-sha256.json")
    for name, sha in hashes.items():
        assert digest((source / name).read_bytes()) == sha, name
    for path in (BASE / "intel-original").iterdir():
        if path.is_file():
            assert path.read_bytes() == (source / "prefill_flush_reload" / "prefill-four-rows" / path.name).read_bytes(), path.name
    sys.path.insert(0, str(source))
    from prefill_flush_reload.analyze import analyze_csv, write_analysis
    config = read_json(BASE / "matched-config.json")
    intel, reference, analysis = validate("intel-original", BASE / "intel-original", config, analyze_csv)
    assert intel["consensus"]["predicted_groups"] == [192, 224, 896, 928, 960, 1984, 2240, 2272, 2304]
    write_analysis(analysis, BASE / "intel-reanalysis")
    manifest = read_json(BASE / "amd" / "manifest.json")
    assert manifest["complete"] and manifest["source_commit"] == PINNED
    assert manifest["primary_acquisition"] == "run-01" and manifest["planned_acquisitions"] == 10
    assert len(manifest["runs"]) == 10 and manifest["configuration"] == config
    assert manifest["source_hashes"] == hashes
    amd = []
    for index, run in enumerate(manifest["runs"], 1):
        assert run["run"] == f"run-{index:02d}"
        assert run["returncode"] == 0
        record, _, _ = validate(run["run"], BASE / "amd" / run["run"], config, analyze_csv, reference)
        assert record["cache_check_passed"]
        amd.append(record)
    aggregates = {key: stats([r["raw_probe"]["injected_gather"][key] for r in amd])
                  for key in ("precision", "recall", "false_positive_rate")}
    aggregates.update(consensus_exact_count=sum(r["consensus"]["exact_match"] for r in amd),
                      filtered_exact_count=sum(r["filtered"]["exact_match"] for r in amd),
                      consensus_precision=stats([r["consensus"]["precision"] for r in amd]),
                      calibrated_threshold_tsc_ticks=stats([r["calibration"]["reload_threshold_tsc_ticks"] for r in amd]))
    data = dict(source_commit=PINNED, configuration=config, primary_amd="run-01",
        intel_baseline="prefill_flush_reload/prefill-four-rows (seed 42; original recorded data)",
        intel=intel, amd=amd, amd_aggregate=aggregates,
        limitations=["Intel historical compiler/binary/OS tuning metadata are unavailable; cannot claim identical binaries or isolate CPU architecture causally.",
            "Intel has one historical acquisition; AMD repeats one fixed seed/input ten times in a short session.",
            "The code stages flush, gather and reload in one process; this is a controlled CPU synthetic footprint experiment, not an NPU or live-model attack.",
            "Reload cutoff is calibrated independently per acquisition by unchanged code; it is not forced to the same numeric value across platforms.",
            "The first-group heuristic can erase truly adjacent outlier groups; these four true groups are separated."])
    save_json(BASE / "comparison.json", data)
    with (BASE / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["platform", "acquisition", "stage", "case", "tp", "fp", "fn", "tn", "precision", "recall", "false_positive_rate", "predicted_groups"])
        for record in [intel, *amd]:
            platform = "Intel" if record is intel else "AMD"
            for case in CASES:
                values = record["raw_probe"][case]
                writer.writerow([platform, record["name"], "raw_probe", case,
                    *[values[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "false_positive_rate")], ""])
            for stage in ("consensus", "filtered"):
                values = record[stage]
                writer.writerow([platform, record["name"], stage, "injected_gather",
                    *[values[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall")],
                    values["fp"] / (values["fp"] + values["tn"]), json.dumps(values["predicted_groups"])])
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(BASE / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    plot_comparison(data, plt)
    plot_repeats(data, plt)
    write_report(data)
    print(json.dumps({"validated_acquisitions": 11, "validated_records": 11 * 9120,
                      "amd_aggregate": aggregates}, indent=2))


if __name__ == "__main__":
    main()

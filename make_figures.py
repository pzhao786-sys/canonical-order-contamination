"""Figures and LaTeX-ready tables for the report.

  python analyze.py --results results --out results/summary      (first)
  python make_figures.py --summary results/summary --out report/figs

Reads runs.csv / files.csv / mmlu_fragments.csv and writes
  fig_main.pdf (+ .png)   panel (a) reversal diagnostic, panel (b) repaired MMLU boundaries
  tables.txt              every number quoted in the report, in plain text
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# categorical slots 1-3 of the documented palette (validated all-pairs) + neutral ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID, AXIS = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

CLEAN = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/pythia-1.4b"]
SHORT = {"gpt2": "GPT-2", "gpt2-medium": "GPT-2 M", "gpt2-large": "GPT-2 L", "gpt2-xl": "GPT-2 XL",
         "EleutherAI/pythia-1.4b": "Pythia-1.4B", "yonatano/contam-1.4b-dupcount-higher": "contam-1.4B"}
CONTAM = "yonatano/contam-1.4b-dupcount-higher"
CANARY_DUP = {"boolq": 1, "hellaswag": 1, "openbookqa": 1, "mnli": 10, "truthfulqa": 10, "naturalquestions": 10,
              "piqa": 50, "mmlu_professional_psychology": 50, "mmlu_professional_law": 50,
              "mmlu_high_school_psychology": 100}
PAPER_T1 = {"boolq": "0.156", "hellaswag": "0.478", "openbookqa": "0.462", "mnli": "1.96e-11",
            "truthfulqa": "3.43e-13", "naturalquestions": "1e-38", "piqa": "1e-38",
            "mmlu_professional_psychology": "1e-38", "mmlu_professional_law": "1e-38",
            "mmlu_high_school_psychology": "1e-38"}
MAIN_FRAG = ["high_school_us_history", "high_school_world_history", "high_school_european_history",
             "professional_law"]
SUBJ_SHORT = {"high_school_us_history": "US hist.", "high_school_world_history": "World hist.",
              "high_school_european_history": "Eur. hist.", "professional_law": "Prof. law"}


def fisher(p):
    p = np.clip(np.asarray(list(p), dtype=np.float64), 1e-300, 1.0)
    return float(chi2.sf(-2 * np.log(p).sum(), df=2 * len(p))) if len(p) else float("nan")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt_p(p):
    if p != p:
        return "--"
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=6.5, width=0.5, length=2.5, color=AXIS)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="report/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = read(os.path.join(args.summary, "files.csv"))
    runs = read(os.path.join(args.summary, "runs.csv"))
    frag = {r["subject"]: r for r in read(os.path.join(args.summary, "mmlu_fragments.csv"))}
    F = {(r["model"], r["tag"], r["file"]): r for r in files}
    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    # ------------------------------------------------------------------ T1: reproduction of Table 1
    say("== Table 1 reproduction: contaminated model, sharded test (50 shards, 50 perms, 3 shuffle seeds)")
    say("file | dup | paper p | ours median [min, max] | reversed p (median) | eff_canon +- sd(seeds) | eff_rev | strict fp32 p")
    for f, dup in sorted(CANARY_DUP.items(), key=lambda kv: kv[1]):
        r = F.get((CONTAM, "paper50", f))
        if not r:
            continue
        r32 = F.get((CONTAM, "paper50_fp32", f))
        say(f"{f:30s} | {dup:3d} | {PAPER_T1[f]:>8s} | {fmt_p(fnum(r['p_canon_median']))} "
            f"[{fmt_p(fnum(r['p_canon_min']))}, {fmt_p(fnum(r['p_canon_max']))}] | {fmt_p(fnum(r['p_rev_median']))} | "
            f"{fnum(r['eff_canon_mean']):+.3f} +- {fnum(r['eff_canon_sd_seeds']):.3f} | {fnum(r['eff_rev_mean']):+.3f} | "
            f"{fmt_p(fnum(r32['p_canon_median'])) if r32 else '--'}")
    say("clean models on the same canary files (dup count 0 row), p_canon / p_rev:")
    for m in ("gpt2-xl", "EleutherAI/pythia-1.4b"):
        pc = [fnum(F[(m, 'paper50', f)]['p_canon_median']) for f in CANARY_DUP if (m, 'paper50', f) in F]
        pr = [fnum(F[(m, 'paper50', f)]['p_rev_median']) for f in CANARY_DUP if (m, 'paper50', f) in F]
        if pc:
            say(f"  {SHORT[m]:12s} n={len(pc)} min p_canon={min(pc):.3f} #<0.05={sum(p < .05 for p in pc)} | "
                f"min p_rev={min(pr):.4f} #<0.05={sum(p < .05 for p in pr)}")

    # ------------------------------------------------------------------ T2: controls with fresh orderings
    say()
    say("== Fresh-ordering controls (tag ctrl; 2 datasets x 5 orderings per model; 50 shards of 20)")
    say("model | kind | #p_canon<0.05 | #p_rev<0.05 | median p_canon | median p_rev | eff_canon mean+-sd | eff_rev mean+-sd | Fisher canon | Fisher rev")
    ctrl_rows = [r for r in files if r["tag"] == "ctrl"]
    for m in CLEAN:
        for kind in ("sorted", "shuffled"):
            g = [r for r in ctrl_rows if r["model"] == m and f".{kind}" in r["file"]]
            if not g:
                continue
            pc = np.array([fnum(r["p_canon_median"]) for r in g]); pr = np.array([fnum(r["p_rev_median"]) for r in g])
            ec = np.array([fnum(r["eff_canon_mean"]) for r in g]); er = np.array([fnum(r["eff_rev_mean"]) for r in g])
            say(f"{SHORT[m]:12s} | {kind:8s} | {int((pc < .05).sum())}/{len(g)} | {int((pr < .05).sum())}/{len(g)} | "
                f"{np.median(pc):.2e} | {np.median(pr):.2e} | {ec.mean():+.3f}+-{ec.std(ddof=1):.3f} | "
                f"{er.mean():+.3f}+-{er.std(ddof=1):.3f} | {fisher(pc):.2e} | {fisher(pr):.2e}")
    # file-level view of the null control (models are correlated: the same 10 files)
    by_file = defaultdict(list)
    for r in ctrl_rows:
        if ".shuffled" in r["file"]:
            by_file[r["file"]].append(fnum(r["z_canon_mean"]))
    if by_file:
        zbar = np.array([np.mean(v) for v in by_file.values()])
        from scipy.stats import ttest_1samp
        say(f"null control, file-level mean z_canon over models: mean={zbar.mean():+.3f} sd={zbar.std(ddof=1):.3f} "
            f"n={len(zbar)} two-sided t-test p={ttest_1samp(zbar, 0).pvalue:.3f}")

    # ------------------------------------------------------------------ T3: benchmark files, canonical order
    say()
    say("== Off-the-shelf benchmark files, canonical order, clean models (tag paper50): p_canon / p_rev")
    bench = {"ARC-Easy-Test": "ARC-Easy", "tests": "PIQA", "openbookqa_test": "OpenBookQA", "test": "GSM8K",
             "NQ-open.efficientqa.test.1.1": "NaturalQA"}
    for f, name in bench.items():
        cells = []
        for m in CLEAN:
            r = F.get((m, "paper50", f))
            cells.append(f"{SHORT[m]} {fnum(r['p_canon_median']):.3f}/{fnum(r['p_rev_median']):.3f}" if r else f"{SHORT[m]} --")
        say(f"{name:11s} | " + " | ".join(cells))

    # ------------------------------------------------------------------ T4: MMLU fragmentation
    say()
    say("== MMLU: record fragments and the test in clean models")
    tot = sum(int(r["n_elements"]) for r in frag.values()); nfr = sum(int(r["n_fragment_elements"]) for r in frag.values())
    say(f"{len(frag)} files, {tot} list elements, {nfr} fragments ({nfr / tot:.1%}), "
        f"{sum(int(r['n_fragment_elements']) > 0 for r in frag.values())} files affected")
    for tag in ("k10", "paper50"):
        say(f"-- tag {tag}: subject | frag% | " + " | ".join(SHORT[m] for m in CLEAN) + "   (p_canon; eff_canon/eff_rev nats per element)")
        for s in MAIN_FRAG:
            cells = []
            for m in CLEAN:
                r = F.get((m, tag, s))
                cells.append(f"{fmt_p(fnum(r['p_canon_median']))} ({fnum(r['eff_canon_mean']):+.1f}/{fnum(r['eff_rev_mean']):+.1f})" if r else "--")
            say(f"{s:30s} | {100 * fnum(frag[s]['frac_fragment']):4.1f} | " + " | ".join(cells))
    for tag in ("k10_fixed", "paper50_fixed"):
        if not any(r["tag"] == tag for r in files):
            say(f"-- tag {tag}: no results yet")
            continue
        say(f"-- tag {tag} (same text, repaired record boundaries): subject | " + " | ".join(SHORT[m] for m in CLEAN))
        for s in MAIN_FRAG:
            cells = []
            for m in CLEAN:
                r = F.get((m, tag, s))
                cells.append(f"{fmt_p(fnum(r['p_canon_median']))} [{fmt_p(fnum(r['p_canon_min']))},{fmt_p(fnum(r['p_canon_max']))}] "
                             f"({fnum(r['eff_canon_mean']):+.2f}/{fnum(r['eff_rev_mean']):+.2f}; rev p {fmt_p(fnum(r['p_rev_median']))}; n_seeds {r['n_seeds']})" if r else "--")
            say(f"{s:30s} | " + " | ".join(cells))
    # lightly fragmented files: Fisher omnibus, original vs repaired
    minor = [s for s, r in frag.items() if 0 < fnum(r["frac_fragment"]) < 0.3]
    nofrag = [s for s, r in frag.items() if fnum(r["frac_fragment"]) == 0]
    say(f"-- Fisher omnibus over the {len(minor)} lightly fragmented files (<30%) and the {len(nofrag)} fragment-free files, k10")
    for m in CLEAN:
        o = [fnum(F[(m, 'k10', s)]['p_canon_median']) for s in minor if (m, 'k10', s) in F]
        fx = [fnum(F[(m, 'k10_fixed', s)]['p_canon_median']) for s in minor if (m, 'k10_fixed', s) in F]
        c = [fnum(F[(m, 'k10', s)]['p_canon_median']) for s in nofrag if (m, 'k10', s) in F]
        orev = [fnum(F[(m, 'k10', s)]['p_rev_median']) for s in minor if (m, 'k10', s) in F]
        say(f"{SHORT[m]:12s} lightly fragmented original: {fisher(o):.2e} (n={len(o)}; reversed {fisher(orev):.2e}) | "
            f"repaired: {fisher(fx):.2e} (n={len(fx)}) | fragment-free: {fisher(c):.2e} (n={len(c)})")

    # ------------------------------------------------------------------ T5: capability ladder
    say()
    say("== Capability ladder: mean canonical effect (nats/example) by model")
    for m in CLEAN:
        srt = [fnum(r["eff_canon_mean"]) for r in ctrl_rows if r["model"] == m and ".sorted" in r["file"]]
        law = F.get((m, "k10", "professional_law"))
        nll = [fnum(r["nll_per_token_shuf"]) for r in ctrl_rows if r["model"] == m]
        say(f"{SHORT[m]:12s} topic-sorted: {np.mean(srt) if srt else float('nan'):+.3f} | prof. law (fragmented): "
            f"{fnum(law['eff_canon_mean']) if law else float('nan'):+.2f} | NLL/token on controls: {np.mean(nll) if nll else float('nan'):.3f}")

    # ------------------------------------------------------------------ null calibration
    say()
    say("== Calibration: fake-canonical tests (each shuffle vs the other shuffles), fraction with p<0.05")
    for r in read(os.path.join(args.summary, "null_calib.csv")):
        say(f"{SHORT.get(r['model'], r['model']):12s} {r['tag']:14s} runs={r['runs']:>3s} tests={r['null_tests']:>5s} frac={fnum(r['frac_p_below_0.05']):.3f}")

    with open(os.path.join(args.out, "tables.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ================================================================== figure
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.labelsize": 7,
                         "axes.titlesize": 7.5, "pdf.fonttype": 42, "axes.labelcolor": INK2,
                         "text.color": INK, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(5.5, 2.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.085, right=0.995, top=0.91, bottom=0.285, wspace=0.27)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- panel (a): canonical vs reversed gain, one point per (model, file)
    groups = [
        ("exchangeable controls", GRAY, "o", 8,  # legend labels are kept short; the caption has the details
         [r for r in files if (r["tag"] == "ctrl" and ".shuffled" in r["file"])
          or (r["tag"] == "paper50" and r["file"] in CANARY_DUP and r["model"] in CLEAN)]),
        ("topic-sorted (clean models)", ORANGE, "^", 17,
         [r for r in files if r["tag"] == "ctrl" and ".sorted" in r["file"]]),
        ("mis-split MMLU (clean models)", AQUA, "s", 15,
         [r for r in files if r["tag"] == "k10" and r["file"] in MAIN_FRAG and r["model"] in CLEAN]),
        ("memorised (contaminated model)", BLUE, "o", 22,
         [r for r in files if r["model"] == CONTAM and r["tag"] == "paper50" and CANARY_DUP.get(r["file"], 0) >= 10]),
    ]
    lim = (-13, 42)
    ax.plot(lim, lim, color=AXIS, linewidth=0.7, zorder=1)
    ax.axhline(0, color=AXIS, linewidth=0.7, zorder=1)
    ax.axvline(0, color=AXIS, linewidth=0.7, zorder=1)
    handles = []
    for label, color, marker, size, rows in groups:
        x = [fnum(r["eff_canon_mean"]) for r in rows]; y = [fnum(r["eff_rev_mean"]) for r in rows]
        ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors="white", linewidths=0.5, zorder=3,
                   alpha=0.95 if color != GRAY else 0.75, clip_on=True)
        handles.append(Line2D([], [], marker=marker, linestyle="none", markerfacecolor=color, markeredgecolor="white",
                              markeredgewidth=0.5, markersize=np.sqrt(size) * 1.1, label=label))
        print(f"panel (a) group '{label}': n={len(rows)}")
    ax.set_xscale("symlog", linthresh=1, linscale=0.9)
    ax.set_yscale("symlog", linthresh=1, linscale=0.9)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ticks = [-10, -1, 0, 1, 10]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks]); ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    style_axes(ax)
    ax.set_xlabel("canonical-order gain (nats / element)")
    ax.set_ylabel("reversed-order gain (nats / element)")
    ax.set_title("(a) Reversal diagnostic", loc="left", color=INK)
    ax.text(-11.5, -4.2, "reversed =\ncanonical", fontsize=5.5, color=INK2, ha="left", va="top", linespacing=0.95)

    # ---- panel (b): authors' file vs repaired boundaries (same text), 5 clean models per subject
    have_fixed = any(r["tag"] == "k10_fixed" for r in files)
    xs = np.arange(len(MAIN_FRAG))
    off = np.linspace(-0.28, 0.28, len(CLEAN))
    for j, m in enumerate(CLEAN):
        for i, s in enumerate(MAIN_FRAG):
            o = F.get((m, "k10", s)); fx = F.get((m, "k10_fixed", s))
            if o:
                bx.errorbar(xs[i] + off[j], fnum(o["eff_canon_mean"]), yerr=1.96 * fnum(o["eff_canon_se_shards"]),
                            fmt="s", color=AQUA, markersize=3.6, markeredgecolor="white", markeredgewidth=0.5,
                            ecolor=AQUA, elinewidth=0.7, capsize=0, zorder=3)
            if fx:
                bx.errorbar(xs[i] + off[j], fnum(fx["eff_canon_mean"]), yerr=1.96 * fnum(fx["eff_canon_se_shards"]),
                            fmt="D", color=INK2, markersize=3.0, markeredgecolor="white", markeredgewidth=0.5,
                            ecolor=INK2, elinewidth=0.7, capsize=0, zorder=4)
    bx.axhline(0, color=AXIS, linewidth=0.7, zorder=1)
    bx.set_xticks(xs)
    bx.set_xticklabels([f"{SUBJ_SHORT[s]}\n({100 * fnum(frag[s]['frac_fragment']):.0f}%)" for s in MAIN_FRAG])
    bx.set_xlim(-0.55, len(MAIN_FRAG) - 0.45)
    bx.set_ylabel("canonical-order gain (nats / element)")
    bx.set_title("(b) Element boundaries repaired", loc="left", color=INK)
    style_axes(bx)
    bx.grid(axis="x", visible=False)
    handles_b = [Line2D([], [], marker="s", linestyle="none", markerfacecolor=AQUA, markeredgecolor="white",
                        markeredgewidth=0.5, markersize=4.2, label="authors' MMLU file"),
                 Line2D([], [], marker="D", linestyle="none", markerfacecolor=INK2, markeredgecolor="white",
                        markeredgewidth=0.5, markersize=3.6,
                        label="repaired boundaries" + ("" if have_fixed else " (pending)"))]
    kw = dict(frameon=False, fontsize=6, handletextpad=0.2, labelcolor=INK2, borderaxespad=0.0, labelspacing=0.35)
    pa, pb = ax.get_position(), bx.get_position()
    fig.legend(handles=handles[::-1], loc="lower left", ncol=2, columnspacing=0.8,
               bbox_to_anchor=(pa.x0 - 0.07, 0.005), **kw)
    fig.legend(handles=handles_b, loc="lower left", ncol=1, bbox_to_anchor=(pb.x0 + 0.02, 0.005), **kw)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"fig_main.{ext}"), dpi=300)
    print("wrote", os.path.join(args.out, "fig_main.pdf"))


if __name__ == "__main__":
    main()


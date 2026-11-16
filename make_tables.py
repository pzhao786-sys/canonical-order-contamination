"""LaTeX tables for the report, generated from the result CSVs (no hand-copied numbers).

  python analyze.py --results results --out results/summary
  python make_tables.py --summary results/summary --out ../report/tables

Writes one .tex file per table (tabular environments only; captions live in main.tex)
and numbers.tex with \\newcommand macros for values quoted in the running text.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.stats import chi2, spearmanr

from reanalyze_table4 import table4, subject_sets

CLEAN = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/pythia-1.4b"]
SHORT = {"gpt2": "GPT-2", "gpt2-medium": "GPT-2 M", "gpt2-large": "GPT-2 L", "gpt2-xl": "GPT-2 XL",
         "EleutherAI/pythia-1.4b": "Pythia-1.4B", "mistralai/Mistral-7B-v0.1": "Mistral-7B"}
CONTAM = "yonatano/contam-1.4b-dupcount-higher"
MISTRAL = "mistralai/Mistral-7B-v0.1"
CANARY = [("boolq", "BoolQ", 1, "0.156"), ("hellaswag", "HellaSwag", 1, "0.478"),
          ("openbookqa", "OpenBookQA", 1, "0.462"), ("mnli", "MNLI", 10, r"$2.0{\cdot}10^{-11}$"),
          ("truthfulqa", "TruthfulQA", 10, r"$3.4{\cdot}10^{-13}$"),
          ("naturalquestions", "Natural Questions", 10, r"$<10^{-38}$"), ("piqa", "PIQA", 50, r"$<10^{-38}$"),
          ("mmlu_professional_psychology", "MMLU Prof. Psychology", 50, r"$<10^{-38}$"),
          ("mmlu_professional_law", "MMLU Prof. Law", 50, r"$<10^{-38}$"),
          ("mmlu_high_school_psychology", "MMLU H.S. Psychology", 100, r"$<10^{-38}$")]
MAIN_FRAG = [("high_school_us_history", "H.S. US history"), ("high_school_world_history", "H.S. world history"),
             ("high_school_european_history", "H.S. European history"), ("professional_law", "Professional law")]
# paper, Table 2 (Pythia-1.4B, GPT-2 XL) for the single-file benchmarks
PAPER_T2 = {"ARC-Easy": (0.686, 0.929), "GSM8K": (0.619, 0.770), "NaturalQA": (0.948, 0.463),
            "OpenBookQA": (0.364, 0.902), "PIQA": (0.956, 0.959)}
BENCH = [("ARC-Easy-Test", "ARC-Easy"), ("test", "GSM8K"), ("NQ-open.efficientqa.test.1.1", "NaturalQA"),
         ("openbookqa_test", "OpenBookQA"), ("tests", "PIQA")]


def fisher(p):
    p = np.clip(np.asarray([x for x in p if x == x], dtype=np.float64), 1e-300, 1.0)
    return float(chi2.sf(-2 * np.log(p).sum(), df=2 * len(p))) if len(p) else float("nan")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def texp(p, digits=1):
    """p-value for a table cell."""
    if p != p:
        return "--"
    if p >= 0.0995:
        return f"{p:.2f}"
    if p >= 0.00095:
        return f"{p:.3f}"
    mant, exp = f"{p:.{digits}e}".split("e")
    mant = mant.rstrip("0").rstrip(".") if digits else mant
    if mant in ("1", "1.0"):
        return rf"$10^{{{int(exp)}}}$"
    return rf"${mant}{{\cdot}}10^{{{int(exp)}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="../report/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = read(os.path.join(args.summary, "files.csv"))
    runs = read(os.path.join(args.summary, "runs.csv"))
    fragcsv = os.path.join(args.summary, "mmlu_fragments.csv")
    frag = {r["subject"]: r for r in read(fragcsv)}
    F = {(r["model"], r["tag"], r["file"]): r for r in files}
    t4 = table4()
    sets = subject_sets(fragcsv)
    macros = {}

    def write(name, body):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write(body + "\n")
        print("wrote", name)

    def P(m, tag, s, key="p_canon_median"):
        r = F.get((m, tag, s))
        return fnum(r[key]) if r else float("nan")

    def cell_range(m, tag, s, which="canon"):
        r = F.get((m, tag, s))
        if not r:
            return "--"
        med = texp(fnum(r[f"p_{which}_median"]))
        if int(r["n_seeds"]) > 1:
            return f"{med} [{texp(fnum(r[f'p_{which}_min']))}, {texp(fnum(r[f'p_{which}_max']))}]"
        return med

    # ---------------------------------------------------------------- appendix: Table 1 reproduction
    rows = []
    for f, name, dup, paper in CANARY:
        r = F.get((CONTAM, "paper50", f))
        if not r:
            continue
        r32 = F.get((CONTAM, "paper50_fp32", f))
        rows.append(f"{name} & {dup} & {paper} & {texp(fnum(r['p_canon_median']))} "
                    f"[{texp(fnum(r['p_canon_min']))}, {texp(fnum(r['p_canon_max']))}] & "
                    f"{texp(fnum(r32['p_canon_median'])) if r32 else '--'} & "
                    f"${fnum(r['eff_canon_mean']):+.2f}\\pm{fnum(r['eff_canon_se_shards']):.2f}$ & "
                    f"{texp(fnum(r['p_rev_median']))} & ${fnum(r['eff_rev_mean']):+.2f}\\pm{fnum(r['eff_rev_se_shards']):.2f}$ \\\\")
    write("tab_repro.tex", "\n".join([
        r"\begin{tabular}{@{}lrllllll@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{canonical order} & \multicolumn{2}{c}{reversed order} \\",
        r"\cmidrule(lr){3-6}\cmidrule(l){7-8}",
        r"File & dup. & paper $p$ & our $p$: median [min, max] & fp32 & gain & $p$ & gain \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]))

    # ---------------------------------------------------------------- appendix: calibration
    rows = []
    for r in read(os.path.join(args.summary, "null_calib.csv")):
        if r["tag"] in ("smoke",):
            continue
        rows.append(f"{SHORT.get(r['model'], 'contam-1.4B')} & \\texttt{{{r['tag'].replace('_', chr(92) + '_')}}} & {r['runs']} & {r['null_tests']} & "
                    f"{100 * fnum(r['frac_p_below_0.05']):.1f}\\% \\\\")
    write("tab_calib.tex", "\n".join([r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
                                      r"Model & protocol & runs & null tests & rejected at 0.05 \\", r"\midrule",
                                      *rows, r"\bottomrule", r"\end{tabular}"]))
    allc = [r for r in read(os.path.join(args.summary, "null_calib.csv"))]
    fr = [fnum(r["frac_p_below_0.05"]) for r in allc if int(r["null_tests"]) >= 1000]
    macros["calibmin"] = f"{100 * min(fr):.1f}"
    macros["calibmax"] = f"{100 * max(fr):.1f}"

    # ---------------------------------------------------------------- appendix: fresh-ordering controls
    rows = []
    tot = {"sorted": [0, 0, 0], "shuffled": [0, 0, 0]}
    gains = {"sorted": [], "shuffled": []}
    for kind in ("sorted", "shuffled"):
        for m in CLEAN:
            g = [r for r in files if r["tag"] == "ctrl" and r["model"] == m and f".{kind}" in r["file"]]
            if not g:
                continue
            pc = np.array([fnum(r["p_canon_median"]) for r in g]); pr = np.array([fnum(r["p_rev_median"]) for r in g])
            ec = np.array([fnum(r["eff_canon_mean"]) for r in g]); er = np.array([fnum(r["eff_rev_mean"]) for r in g])
            tot[kind][0] += int((pc < .05).sum()); tot[kind][1] += int((pr < .05).sum()); tot[kind][2] += len(g)
            gains[kind].append((ec.mean(), er.mean()))
            rows.append(f"{'topic-sorted' if kind == 'sorted' else 'uniform'} & {SHORT[m]} & {int((pc < .05).sum())}/{len(g)} & "
                        f"{texp(float(np.median(pc)))} & ${ec.mean():+.2f}\\pm{ec.std(ddof=1):.2f}$ & "
                        f"{int((pr < .05).sum())}/{len(g)} & {texp(float(np.median(pr)))} & ${er.mean():+.2f}\\pm{er.std(ddof=1):.2f}$ \\\\")
        if kind == "sorted":
            rows.append(r"\midrule")
    write("tab_ctrl.tex", "\n".join([
        r"\begin{tabular}{@{}llrllrll@{}}", r"\toprule",
        r" & & \multicolumn{3}{c}{canonical order} & \multicolumn{3}{c}{reversed order} \\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
        r"Fresh ordering & Model & $p<0.05$ & median $p$ & gain (mean$\pm$sd) & $p<0.05$ & median $p$ & gain (mean$\pm$sd) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))
    if gains["sorted"]:
        c = np.mean([g[0] for g in gains["sorted"]]); r_ = np.mean([g[1] for g in gains["sorted"]])
        macros.update({"sortedsig": f"{tot['sorted'][0]}", "sortedrevsig": f"{tot['sorted'][1]}", "ctrln": f"{tot['sorted'][2]}",
                       "sortedgain": f"{c:.2f}", "sortedrevgain": f"{r_:.2f}", "sortedkeep": f"{100 * r_ / c:.0f}",
                       "unifsig": f"{tot['shuffled'][0]}", "unifrevsig": f"{tot['shuffled'][1]}"})

    # ---------------------------------------------------------------- appendix: benchmark files, canonical order
    rows = []
    for stem, name in BENCH:
        cells = [texp(P(m, "paper50", stem)) for m in CLEAN]
        pp = PAPER_T2[name]
        rows.append(f"{name} & " + " & ".join(cells) + f" & {pp[0]:.3f} & {pp[1]:.3f} \\\\")
    write("tab_bench.tex", "\n".join([
        r"\begin{tabular}{@{}lccccc|cc@{}}", r"\toprule",
        r" & \multicolumn{5}{c|}{our $p$ (P50, canonical order)} & \multicolumn{2}{c}{paper, Table 2} \\",
        "Benchmark & " + " & ".join(SHORT[m] for m in CLEAN) + r" & Pythia-1.4B & GPT-2 XL \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]))

    # ---------------------------------------------------------------- appendix: all 57 MMLU files
    def cp(p):  # compact p-value for the wide appendix table
        if p != p:
            return "--"
        if p >= 0.0995:
            return f"{p:.2f}"
        if p >= 0.00095:
            return f"{p:.3f}"
        mant, exp = f"{p:.0e}".split("e")
        return f"{mant}e{int(exp)}"

    rows = []
    order = sorted(frag, key=lambda s: (-fnum(frag[s]["frac_fragment"]), s))
    have_mistral = any(r["model"] == MISTRAL for r in files)
    for s in order:
        fr_ = fnum(frag[s]["frac_fragment"])
        paper = t4.get(s)
        status = "kept" if s in sets["kept"] else ("dropped" if paper else "n/a")
        k10 = " & ".join(cp(P(m, "k10", s)) for m in CLEAN)
        fixed = " & ".join(cp(P(m, "k10_fixed", s)) if fr_ > 0 else "" for m in CLEAN)
        p50 = " & ".join(cp(P(m, "paper50", s)) for m in ("gpt2-xl", "EleutherAI/pythia-1.4b"))
        mist = ((" & " + cp(P(MISTRAL, "paper50", s)) + " & " + (cp(P(MISTRAL, "paper50_fixed", s)) if fr_ > 0 else ""))
                if have_mistral else "")
        pap = (" & ".join(("$<$1e-38" if paper[k] <= 1e-37 else f"{paper[k]:.3f}") for k in ("GPT-2 XL", "Pythia-1.4B", "LLaMA2-7B", "Mistral-7B"))
               if paper else "-- & -- & -- & --")
        rows.append(f"{s.replace('_', ' ')} & {frag[s]['n_elements']} & {100 * fr_:.1f} & {status} & {k10} & {fixed} & {p50}{mist} & {pap} \\\\")
    mcol = "cc|" if have_mistral else ""
    mhead1 = r" & \multicolumn{2}{c|}{P50, Mistral (ours)}" if have_mistral else ""
    mhead2 = " & auth. & rep." if have_mistral else ""
    write("tab_mmlu_all.tex", "\n".join([
        r"\begin{tabular}{@{}lrrl|ccccc|ccccc|cc|" + mcol + "cccc@{}}", r"\toprule",
        r" & & \% & paper's & \multicolumn{5}{c|}{K10, authors' file} & \multicolumn{5}{c|}{K10, boundaries repaired} & \multicolumn{2}{c|}{P50, ours}" + mhead1 + r" & \multicolumn{4}{c}{P50, paper's Table 4} \\",
        r"File & $n$ & split & filter & G & M & L & XL & Py & G & M & L & XL & Py & XL & Py" + mhead2 + r" & XL & Py & LLaMA2 & Mistral \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]))

    # ---------------------------------------------------------------- numbers for the text
    tot_el = sum(int(r["n_elements"]) for r in frag.values()); tot_fr = sum(int(r["n_fragment_elements"]) for r in frag.values())
    n_well = sum(int(r["n_records_wellformed"]) for r in frag.values())
    macros["wellformed"] = f"{n_well:,}".replace(",", "{,}")
    macros.update({"nelements": f"{tot_el:,}".replace(",", "{,}"), "nfragments": f"{tot_fr}",
                   "fragpct": f"{100 * tot_fr / tot_el:.1f}",
                   "nfragfiles": f"{sum(int(r['n_fragment_elements']) > 0 for r in frag.values())}"})
    for m, key in (("EleutherAI/pythia-1.4b", "Pythia-1.4B"), ("gpt2-xl", "GPT-2 XL")):
        ours = np.array([P(m, "paper50", s) for s in t4]); theirs = np.array([t4[s][key] for s in t4])
        ok = ~np.isnan(ours)
        macros["rho" + ("py" if "pythia" in m else "xl")] = f"{spearmanr(ours[ok], theirs[ok]).correlation:.2f}"

    # ---------------------------------------------------------------- main results table
    minor = [s for s in frag if 0 < fnum(frag[s]["frac_fragment"]) < 0.3]
    nofrag = [s for s in frag if fnum(frag[s]["frac_fragment"]) == 0]
    show = ["gpt2", "EleutherAI/pythia-1.4b"]
    rows = []
    for s, name in MAIN_FRAG:
        cells = []
        for m in show:
            # authors' file: median of the seeds (ranges are in the appendix); repaired: median [min, max]
            cells += [texp(P(m, "k10", s)), cell_range(m, "k10_fixed", s)]
        rows.append(f"{name} & {100 * fnum(frag[s]['frac_fragment']):.0f}\\% & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    # seed extremes over all five clean models, for the caption / text
    amax = max(P(m, "k10", s, "p_canon_max") for s, _ in MAIN_FRAG for m in CLEAN)
    fmin = min(P(m, "k10_fixed", s, "p_canon_min") for s, _ in MAIN_FRAG for m in CLEAN)
    hist = [s for s, _ in MAIN_FRAG if s != "professional_law"]
    macros["mainauthmax"] = f"{np.ceil(amax * 1000) / 1000:.3f}"   # "every seed below ..."
    macros["mainfixedmin"] = f"{np.floor(fmin * 100) / 100:.2f}"    # "every seed above ..."
    macros["histauthmax"] = f"{max(P(m, 'k10', s, 'p_canon_max') for s in hist for m in CLEAN):.3f}"
    g = [P(m, "k10_fixed", "professional_law", "eff_canon_mean") for m in CLEAN]
    macros["lawfixedgainmax"] = f"{max(g):.1f}"
    g0 = [P(m, "k10", "professional_law", "eff_canon_mean") for m in CLEAN]
    macros["lawgainmin"] = f"{min(g0):.0f}"; macros["lawgainmax"] = f"{max(g0):.0f}"
    rv = [P(CONTAM, "paper50", f, "p_rev_min") for f, *_ in CANARY]
    macros["canaryrevmin"] = f"{min(rv):.2f}"
    seedrows = []
    for s, name in MAIN_FRAG:
        for m in ("gpt2", "EleutherAI/pythia-1.4b"):
            seedrows.append(f"{name} & {SHORT[m]} & {cell_range(m, 'k10', s)} & {cell_range(m, 'k10_fixed', s)} & "
                            f"${P(m, 'k10', s, 'eff_canon_mean'):+.1f}\\pm{P(m, 'k10', s, 'eff_canon_se_shards'):.1f}$ & "
                            f"${P(m, 'k10_fixed', s, 'eff_canon_mean'):+.1f}\\pm{P(m, 'k10_fixed', s, 'eff_canon_se_shards'):.1f}$ \\\\")
    write("tab_seeds.tex", "\n".join([
        r"\begin{tabular}{@{}ll|ll|ll@{}}", r"\toprule",
        r" & & \multicolumn{2}{c|}{$p$: median [min, max] over 3 seeds} & \multicolumn{2}{c}{gain (nats/element) $\pm$ s.e.} \\",
        r"File & Model & authors' file & repaired & authors' file & repaired \\", r"\midrule",
        *seedrows, r"\bottomrule", r"\end{tabular}"]))

    def fisher_pair(S, m):
        o = [P(m, "k10", s) for s in S]; fx = [P(m, "k10_fixed", s) for s in S]
        n_fx = sum(x == x for x in fx)
        return [texp(fisher(o)), texp(fisher(fx)) if n_fx == len(S) else "--"]

    cells = sum((fisher_pair(minor, m) for m in show), [])
    rows.append(f"Fisher, {len(minor)} lightly split files & $<$6\\% & " + " & ".join(cells) + r" \\")
    cells = sum((fisher_pair(sets["kept_frag"], m) for m in show), [])
    rows.append(f"Fisher, the {len(sets['kept_frag'])} the paper kept & $<$4\\% & " + " & ".join(cells) + r" \\")
    cells = []
    for m in show:
        cells += [texp(fisher([P(m, 'k10', s) for s in nofrag])), "(same)"]
    rows.append(f"Fisher, {len(nofrag)} files not mis-split & 0\\% & " + " & ".join(cells) + r" \\")
    write("tab_main.tex", "\n".join([
        r"\begin{tabular}{@{}lr|ll|ll@{}}", r"\toprule",
        r" & mis- & \multicolumn{2}{c|}{GPT-2 (124M)} & \multicolumn{2}{c}{Pythia-1.4B} \\",
        r"MMLU file(s) & split & authors' file & repaired & authors' file & repaired \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]))

    # how often does repair raise the p-value in the lightly mis-split files?
    up = tot_cmp = 0
    for m in ("gpt2", "gpt2-xl", "EleutherAI/pythia-1.4b"):
        for s in minor:
            o, fx = P(m, "k10", s), P(m, "k10_fixed", s)
            if o == o and fx == fx:
                tot_cmp += 1; up += fx > o
    macros["repairup"] = f"{up}"; macros["repairn"] = f"{tot_cmp}"; macros["nminor"] = f"{len(minor)}"
    fl = [fisher([P(m, "k10", s) for s in minor]) for m in ("gpt2", "gpt2-xl", "EleutherAI/pythia-1.4b")]
    fr2 = [fisher([P(m, "k10_fixed", s) for s in minor]) for m in ("gpt2", "gpt2-xl", "EleutherAI/pythia-1.4b")]
    macros["minorfixedmin"] = f"{min(fr2):.2f}"; macros["minorfixedmax"] = f"{max(fr2):.2f}"
    print("lightly mis-split Fisher authors':", fl, "repaired:", fr2)
    ff = [fisher([P(m, "k10", s) for s in nofrag]) for m in CLEAN]
    print("fragment-free K10 Fisher per model:", dict(zip(CLEAN, ff)))

    # ---------------------------------------------------------------- appendix: P50 on the repaired main files
    rows = []
    for s, name in MAIN_FRAG:
        cells = []
        for m in ("gpt2-xl", "EleutherAI/pythia-1.4b"):
            cells += [texp(P(m, "paper50", s)), texp(P(m, "paper50_fixed", s))]
        xl5 = [texp(P(m, "k10", s)) + r" $\to$ " + texp(P(m, "k10_fixed", s)) for m in ("gpt2-medium", "gpt2-large", "gpt2-xl")]
        rows.append(f"{name} & " + " & ".join(cells) + " & " + " & ".join(xl5) + r" \\")
    write("tab_p50fixed.tex", "\n".join([
        r"\begin{tabular}{@{}l|cc|cc|ccc@{}}", r"\toprule",
        r" & \multicolumn{2}{c|}{P50, GPT-2 XL} & \multicolumn{2}{c|}{P50, Pythia-1.4B} & \multicolumn{3}{c}{K10, authors' $\to$ repaired (one seed)} \\",
        r"File & authors' & repaired & authors' & repaired & GPT-2 M & GPT-2 L & GPT-2 XL \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]))

    # ---------------------------------------------------------------- omnibus table (paper's numbers and ours)
    def fis_paper(model, subjects):
        return fisher([t4[s][model] for s in subjects])

    def fis_ours(model, tag, subjects):
        vals = [P(model, tag, s) for s in subjects]
        return fisher(vals) if all(v == v for v in vals) else float("nan")

    cols = [("LLaMA2", lambda S: fis_paper("LLaMA2-7B", S)), ("Mistral", lambda S: fis_paper("Mistral-7B", S)),
            ("Pythia", lambda S: fis_paper("Pythia-1.4B", S))]
    have_mistral = any(r["model"] == MISTRAL for r in files)
    p50_cols = [("Pythia", "EleutherAI/pythia-1.4b")] + ([("Mistral", MISTRAL)] if have_mistral else [])
    k10_cols = [("GPT-2", "gpt2"), ("GPT-2 XL", "gpt2-xl"), ("Pythia", "EleutherAI/pythia-1.4b")]
    body = []
    for label, S in ((f"all {len(sets['kept'])} retained files", sets["kept"]),
                     (f"the {len(sets['kept_frag'])} with mis-split elements", sets["kept_frag"]),
                     (f"the {len(sets['kept_clean'])} without", sets["kept_clean"])):
        cells = ([texp(fn(S)) for _, fn in cols] + [texp(fis_ours(m, "paper50", S)) for _, m in p50_cols]
                 + [texp(fis_ours(m, "k10", S)) for _, m in k10_cols])
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    cells = (["--"] * len(cols) + [texp(fis_ours(m, "paper50_fixed", sets["kept_frag"])) for _, m in p50_cols]
             + [texp(fis_ours(m, "k10_fixed", sets["kept_frag"])) for _, m in k10_cols])
    body.append(f"the {len(sets['kept_frag'])}, boundaries repaired & " + " & ".join(cells) + r" \\")
    write("tab_omnibus.tex", "\n".join([
        r"\begin{tabular}{@{}l|ccc|" + "c" * len(p50_cols) + "|ccc@{}}", r"\toprule",
        r" & \multicolumn{3}{c|}{paper's Table 4 (P50)} & \multicolumn{" + str(len(p50_cols)) + r"}{c|}{ours, P50} & \multicolumn{3}{c}{ours, K10} \\",
        "Fisher omnibus over & " + " & ".join(c for c, _ in cols) + " & " + " & ".join(c for c, _ in p50_cols)
        + " & " + " & ".join(c for c, _ in k10_cols) + r" \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]))

    with open(os.path.join(args.out, "numbers.tex"), "w", encoding="utf-8") as f:
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    print("macros:", macros)


if __name__ == "__main__":
    main()

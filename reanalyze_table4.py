"""Re-analysis of the paper's own Table 4 (per-subject MMLU p-values) in the light of record fragmentation.

The paper excludes the 14 MMLU files on which GPT-2 XL or BioMedLM reject at 0.05 and combines the remaining
43 p-values with Fisher's method (Table 2: LLaMA2 0.014, Mistral 0.011, Pythia 0.362).  Here we recompute that
omnibus from Table 4 and then additionally drop every file that contains at least one record fragment
(results/summary/mmlu_fragments.csv, from mmlu_fragments.py).

  python reanalyze_table4.py --fragments results/summary/mmlu_fragments.csv
"""
import argparse
import csv

import numpy as np
from scipy.stats import chi2

MODELS = ["LLaMA2-7B", "Mistral-7B", "Pythia-1.4B", "GPT-2 XL", "BioMedLM"]
# Table 4 of Oren et al. (2023); "1e-38" is the paper's marker for numerically zero.
TABLE4 = """
abstract_algebra 0.103 0.246 0.645 0.894 0.861
anatomy 0.586 0.605 0.605 0.439 0.649
astronomy 0.550 0.657 0.050 0.003 0.050
business_ethics 0.936 0.440 0.107 0.808 0.499
clinical_knowledge 0.199 0.108 0.081 0.004 0.268
college_biology 0.108 0.051 0.152 0.260 0.779
college_chemistry 0.526 0.614 0.595 0.355 0.265
college_computer_science 0.060 0.498 0.383 0.532 0.267
college_mathematics 0.059 0.151 0.162 0.321 0.727
college_medicine 0.397 0.106 0.340 0.440 0.067
college_physics 0.694 0.757 0.972 0.719 0.262
computer_security 0.214 0.007 0.314 0.180 0.928
conceptual_physics 0.554 0.333 0.811 0.710 0.924
econometrics 0.616 0.761 0.540 0.508 0.035
electrical_engineering 0.266 0.364 0.595 0.490 0.277
elementary_mathematics 0.059 0.416 0.260 0.355 0.528
formal_logic 0.666 0.750 0.990 0.930 0.398
global_facts 0.779 0.957 0.448 0.339 0.667
high_school_biology 0.645 0.279 0.476 0.499 0.416
high_school_chemistry 0.229 0.426 0.813 0.539 0.055
high_school_computer_science 0.023 0.085 0.106 0.045 0.484
high_school_european_history 0.009 1e-38 1e-38 1e-38 1e-38
high_school_geography 0.194 0.339 0.341 0.721 0.277
high_school_government_and_politics 0.294 0.066 0.025 0.372 0.003
high_school_macroeconomics 0.543 0.222 0.276 0.236 0.511
high_school_mathematics 0.473 0.182 0.001 0.272 0.032
high_school_microeconomics 0.862 0.797 0.712 0.122 0.181
high_school_physics 0.339 0.757 0.171 0.114 0.354
high_school_psychology 0.033 0.037 0.004 0.039 0.007
high_school_statistics 0.077 0.119 0.228 0.233 0.211
high_school_us_history 1e-38 1e-38 1e-38 1e-38 0.001
high_school_world_history 1e-38 1e-38 1e-38 1e-38 1e-38
human_aging 0.882 0.830 0.617 0.677 0.803
human_sexuality 0.807 0.959 0.502 0.789 0.471
international_law 0.061 0.545 0.532 0.582 0.817
jurisprudence 0.009 0.075 0.852 0.879 0.616
logical_fallacies 0.103 0.188 0.082 0.397 0.193
machine_learning 0.266 0.297 0.373 0.061 0.868
management 0.516 0.209 0.454 0.907 0.281
marketing 0.874 0.684 0.977 0.848 0.451
medical_genetics 0.501 0.037 0.523 0.425 0.729
miscellaneous 0.099 0.122 0.086 0.266 0.081
moral_disputes 0.017 0.011 0.097 0.190 0.157
moral_scenarios 0.652 0.022 0.121 0.487 0.413
nutrition 0.163 0.601 0.413 0.865 0.609
philosophy 0.011 0.094 0.003 0.049 0.044
prehistory 0.203 0.412 0.055 0.268 0.262
professional_accounting 0.708 0.242 0.197 0.910 0.798
professional_medicine 0.001 0.017 0.001 0.002 1e-38
professional_psychology 0.001 0.001 0.058 0.009 0.201
public_relations 0.402 0.237 0.162 0.512 0.622
security_studies 0.070 0.043 0.687 0.061 0.073
sociology 0.350 0.258 0.260 0.660 0.496
us_foreign_policy 0.934 0.252 0.778 0.646 0.173
virology 0.203 0.854 0.219 0.796 0.212
world_religions 0.311 0.882 0.933 0.222 0.851
"""


def table4():
    out = {}
    for line in TABLE4.strip().splitlines():
        parts = line.split()
        out[parts[0]] = dict(zip(MODELS, map(float, parts[1:])))
    return out


def fisher(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0)
    return float(chi2.sf(-2 * np.log(p).sum(), df=2 * len(p)))


def subject_sets(fragments_csv):
    """kept = the paper's 43 retained files; kept_frag / kept_clean split them by fragment content."""
    t4 = table4()
    frag = {r["subject"]: float(r["frac_fragment"]) for r in csv.DictReader(open(fragments_csv, encoding="utf-8"))}
    # the paper's filter: drop a file if GPT-2 XL or BioMedLM rejects at 0.05 (0.050 counts, to reach their 14 / 43)
    kept = [s for s, v in t4.items() if not (v["GPT-2 XL"] <= 0.05 or v["BioMedLM"] <= 0.05)]
    return {
        "kept": kept,
        "dropped": [s for s in t4 if s not in kept],
        "kept_frag": sorted((s for s in kept if frag[s] > 0), key=lambda s: -frag[s]),
        "kept_clean": [s for s in kept if frag[s] == 0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", default="results/summary/mmlu_fragments.csv")
    ap.add_argument("--out", default="results/summary/table4_reanalysis.csv")
    ap.add_argument("--print", dest="print_set", choices=["kept", "dropped", "kept_frag", "kept_clean"],
                    help="only print one subject list (space separated) and exit; used by the run scripts")
    args = ap.parse_args()
    if args.print_set:
        print(" ".join(subject_sets(args.fragments)[args.print_set]))
        return
    t4 = table4()
    frag = {r["subject"]: float(r["frac_fragment"]) for r in csv.DictReader(open(args.fragments, encoding="utf-8"))}
    kept = subject_sets(args.fragments)["kept"]
    dropped = [s for s in t4 if s not in kept]
    print(f"Table 4 has {len(t4)} subjects (professional_law is not listed); filter drops {len(dropped)}, keeps {len(kept)}")
    print(f"  dropped files containing fragments: {sum(frag[s] > 0 for s in dropped)}/{len(dropped)}")
    kept_frag = sorted((s for s in kept if frag[s] > 0), key=lambda s: -frag[s])
    kept_clean = [s for s in kept if frag[s] == 0]
    print(f"  kept files that still contain fragments: {len(kept_frag)}: "
          + ", ".join(f"{s} ({frag[s]:.1%})" for s in kept_frag))
    rows = []
    rng = np.random.default_rng(0)
    n_draws = 100_000
    for m in ["LLaMA2-7B", "Mistral-7B", "Pythia-1.4B"]:
        logp = {s: np.log(max(t4[s][m], 1e-300)) for s in kept}
        # Is the fragment subset special?  Compare its share of the evidence (sum of -2 log p) with
        # random subsets of the same size drawn from the 43 kept files.
        obs = -2 * sum(logp[s] for s in kept_frag)
        allv = np.array([-2 * logp[s] for s in kept])
        draws = np.array([allv[rng.choice(len(kept), size=len(kept_frag), replace=False)].sum()
                          for _ in range(n_draws)])
        perm_p = float((1 + (draws >= obs).sum()) / (1 + n_draws))
        row = {
            "model": m,
            "fisher_paper_filter": fisher([t4[s][m] for s in kept]),
            "n_paper_filter": len(kept),
            "fisher_fragment_free": fisher([t4[s][m] for s in kept_clean]),
            "n_fragment_free": len(kept_clean),
            "fisher_kept_with_fragments": fisher([t4[s][m] for s in kept_frag]),
            "n_kept_with_fragments": len(kept_frag),
            "perm_p_fragment_subset": perm_p,
        }
        rows.append(row)
        print(f"{m:12s} Fisher over paper's {len(kept)} files: {row['fisher_paper_filter']:.4f} | "
              f"fragment-free {len(kept_clean)} files: {row['fisher_fragment_free']:.4f} | "
              f"the {len(kept_frag)} kept files with fragments: {row['fisher_kept_with_fragments']:.4g} | "
              f"P(random {len(kept_frag)}-file subset carries as much evidence) = {perm_p:.4f}")
    # A selection rule that uses no model output at all: every Table-4 file without a mis-split element.
    nofrag = [s for s in t4 if frag[s] == 0]
    print(f"all {len(nofrag)} Table-4 files without mis-split elements (no negative-control filter):")
    for m in MODELS:
        v = fisher([t4[s][m] for s in nofrag])
        print(f"  {m:12s} Fisher = {v:.3f}")
        for row in rows:
            if row["model"] == m:
                row["fisher_all_fragment_free"] = v
                row["n_all_fragment_free"] = len(nofrag)
    # Was the paper's filter detecting mis-split files?  (professional_law: excluded by the paper, 32% mis-split)
    from scipy.stats import fisher_exact
    a = sum(frag[s] > 0 for s in dropped) + 1
    b = len(dropped) + 1 - a
    c, d = len(kept_frag), len(kept_clean)
    print(f"paper's filter: excluded files {a} mis-split / {b} not; retained files {c} / {d}; "
          f"Fisher exact p = {fisher_exact([[a, b], [c, d]])[1]:.4f}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

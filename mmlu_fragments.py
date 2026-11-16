"""Diagnose and repair record fragmentation in the authors' MMLU JSON files.

The files in benchmarks/mmlu_json/*.json are JSON lists of strings that the test
treats as exchangeable examples.  Many list elements are not whole CSV records
but pieces of one record (they end in a comma, with no answer letter and no
newline).  In canonical order consecutive pieces form continuous text; any
shuffle breaks it.  This script

  * measures, per subject, the share of elements that are fragments, and
  * writes a repaired file in which the same text is re-split at true CSV
    record boundaries (nothing else changes), for a one-factor rerun.

Usage:
  python mmlu_fragments.py --src <mmlu_json dir> --out <dir for repaired json> --csv results/summary/mmlu_fragments.csv
"""
import argparse
import csv
import glob
import io
import json
import os
import re

COMPLETE = re.compile(r",[A-D]\s*$")


def as_str(e):
    return e if isinstance(e, str) else json.dumps(e)


def repair(elements):
    """Re-split the concatenated text at CSV record boundaries. Returns list[str] of raw record text."""
    text = "".join(elements)
    rows = list(csv.reader(io.StringIO(text)))
    out = []
    for row in rows:
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow(row)
        out.append(buf.getvalue())
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    rows_out = []
    for path in sorted(glob.glob(os.path.join(args.src, "*.json"))):
        subj = os.path.splitext(os.path.basename(path))[0]
        elements = [as_str(e) for e in json.load(open(path, encoding="utf-8"))]
        n = len(elements)
        complete = [bool(COMPLETE.search(e)) for e in elements]
        n_frag = n - sum(complete)
        # a record "spans" a boundary when element i is not complete: i and i+1 belong together
        fixed, parsed = repair(elements)
        n_six = sum(len(r) == 6 and r[5].strip() in "ABCD" and r[5].strip() != "" for r in parsed)
        with open(os.path.join(args.out, subj + ".json"), "w", encoding="utf-8") as f:
            json.dump(fixed, f)
        rows_out.append({
            "subject": subj,
            "n_elements": n,
            "n_fragment_elements": n_frag,
            "frac_fragment": round(n_frag / n, 4),
            "n_records_repaired": len(fixed),
            "n_records_wellformed": n_six,
            "chars_original": sum(map(len, elements)),
            "chars_repaired": sum(map(len, fixed)),
        })
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    tot = sum(r["n_elements"] for r in rows_out)
    frag = sum(r["n_fragment_elements"] for r in rows_out)
    print(f"{len(rows_out)} subjects, {tot} elements, {frag} fragments ({frag / tot:.1%}); "
          f"{sum(r['n_fragment_elements'] > 0 for r in rows_out)} subjects affected")
    for r in sorted(rows_out, key=lambda r: -r["frac_fragment"]):
        print(f"{r['subject']:38s} n={r['n_elements']:5d} frag={r['frac_fragment']:.3f} "
              f"repaired_records={r['n_records_repaired']:5d} wellformed={r['n_records_wellformed']:5d} "
              f"chars {r['chars_original']}->{r['chars_repaired']}")


if __name__ == "__main__":
    main()

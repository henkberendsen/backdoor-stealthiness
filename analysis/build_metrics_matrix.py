#!/usr/bin/env python3
"""
Build the canonical per-configuration stealthiness-metric matrix for the
S&P resubmission analyses (Analysis #1 metric-disagreement, #4 ASR-sensitivity,
#6 cross-arch, #7 minimal-set, #8 space-imbalance all consume this one table).

SOURCE OF TRUTH: residual_correlations/metric_tables.tex (verified to match the submitted
paper for all 16 metrics). We PARSE it rather than re-transcribe, so the matrix
has provenance back to the paper tables (addresses the "hand-curated CSV" weakness).

Output: analysis/metrics_matrix.csv  -- one row per (Dataset, Attack, PR_level),
ResNet18, with the 16 stealthiness metrics + ASR + BA.

  Spaces:  input  = l1 l2 l_inf MSE PSNR SSIM LPIPS IS pHash SAM   (Table tab:input-stealthiness-all)
           feature= SS DSWD CDBI                                    (SS/DSWD from feature table, CDBI from new-metric table)
           param  = UCLC TAC TUP                                    (UCLC/TAC from param table, TUP from new-metric table)
  BMS is intentionally EXCLUDED (not one of the paper's 16; unimplemented in eval code).

Notes / decisions (documented):
 * Input metrics are poisoning-rate-independent (same trigger) -> repeated for both PR levels.
 * For asymmetric-trigger attacks (Adap-Patch, Adap-Blend, Narcissus) the input table
   reports (train) and (test) rows; we use the TEST values (deployment-time footprint),
   consistent with dataframing_results_df.csv.
 * DFBA has a single config (no PR) and no SS/CDBI -> PR_level='none', SS/CDBI = NaN.
 * KNOWN PAPER TYPO corrected here: Imagenette/Blend l1 is written 1.62e-3 in the table
   (inconsistent with l2=13.8); the correct magnitude is 1.62e3. We correct it and log it.
"""
import re
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEX = REPO / "residual_correlations" / "metric_tables.tex"
OUT = REPO / "analysis" / "metrics_matrix.csv"

DATASETS = ["CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "Imagenette"]
ATTACKS = ["BadNets", "Blend", "WaNet", "BppAttack", "Adap-Patch",
           "Adap-Blend", "DFST", "Narcissus", "Grond", "DFBA"]
# pretty -> canonical key (matches dataframing_results_df.csv)
KEY = {"BadNets": "badnet", "Blend": "blended", "WaNet": "wanet", "BppAttack": "bpp",
       "Adap-Patch": "adaptive_patch", "Adap-Blend": "adaptive_blend", "DFST": "dfst",
       "Narcissus": "narcissus", "Grond": "grond", "DFBA": "dfba"}
INPUT_METRICS = ["l1", "l2", "l_inf", "MSE", "PSNR", "SSIM", "LPIPS", "IS", "pHash", "SAM"]

text = TEX.read_text()


def clean(line: str) -> str:
    """Strip LaTeX formatting macros, leaving plain &-separated cells."""
    line = line.rstrip()
    line = line.replace(r"\\", "")
    # \multirow{2}{*}{NAME} -> NAME ; \textcolor{red}{X} -> X
    line = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", line)
    line = re.sub(r"\\textcolor\{[^}]*\}\{([^}]*)\}", r"\1", line)
    line = re.sub(r"\\scriptsize\{[^}]*\}", "", line)        # PR exception labels (unused)
    # strip remaining single-arg macros innermost-first (handles \best{\underline{x}})
    prev = None
    while prev != line:
        prev = line
        line = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", line)
    line = re.sub(r"\$[^$]*\$", "", line)                    # $^*$, $\downarrow$
    line = line.replace("^*", "").replace("~", "")
    return line


def fnum(tok: str):
    tok = tok.strip()
    if tok in ("", "-"):
        return ""  # NaN sentinel for CSV
    return float(tok)


def table_body(label: str) -> list:
    """Return cleaned data-row token-lists for the table with the given \\label.

    In the feature/parameter tables the \\multirow{attack} sits on its own line,
    separate from the data cells, so we carry such fragment lines (no '&') onto
    the following data row before tokenising.
    """
    i = text.index("\\label{%s}" % label)
    seg = text[i:]
    seg = seg[:seg.index("\\end{tabular}")]
    mid = seg.index("\\midrule")
    seg = seg[mid + len("\\midrule"):]
    rows = []
    carry = ""
    for raw in seg.splitlines():
        s = raw.strip()
        if not s or s.startswith("\\cmidrule") or s.startswith("\\bottomrule") \
           or s.startswith("\\midrule"):
            continue
        if "&" not in s:                 # fragment (e.g. standalone \multirow{..}{Name})
            carry += " " + s
            continue
        full = (carry + " " + s).strip() if carry else s
        carry = ""
        toks = [t.strip() for t in clean(full).split("&")]
        rows.append(toks)
    return rows


# ---- Table 1: performance (BA, ASR) ; 2 cols per dataset (BA, ASR) ----
perf = {}        # (dataset, attack, level) -> (BA, ASR)
benign_ba = {}
for toks in table_body("tab:attack-performance-4datasets"):
    name = toks[0]
    if name == "Benign":
        for k, ds in enumerate(DATASETS):
            benign_ba[ds] = fnum(toks[2 + 2 * k])
        continue
    if name == "DFBA":
        cur, level = "DFBA", "none"
    elif name in ATTACKS:
        cur, level = name, "high"
    else:
        level = "low"   # continuation row of current attack
    for k, ds in enumerate(DATASETS):
        perf[(ds, cur, level)] = (fnum(toks[2 + 2 * k]), fnum(toks[3 + 2 * k]))


# ---- Table 2: input space ; per dataset block of 13 attack-variant rows, 10 metrics ----
inp = {}         # (dataset, attack_variant_string) -> {metric: val}
cur_ds = None
for toks in table_body("tab:input-stealthiness-all"):
    if toks[0] in DATASETS:
        cur_ds = toks[0]
        variant = toks[1]
        vals = toks[2:12]
    else:
        variant = toks[1] if len(toks) > 1 else toks[0]
        vals = toks[2:12]
    inp[(cur_ds, variant)] = {m: fnum(v) for m, v in zip(INPUT_METRICS, vals)}


def input_for(ds, attack):
    """Pick the right input row; TEST values for asymmetric attacks."""
    asym = {"Adap-Patch": "Adap-Patch (test)", "Adap-Blend": "Adap-Blend (test)",
            "Narcissus": "Narcissus (test)"}
    variant = asym.get(attack, attack)
    return inp[(ds, variant)]


# ---- generic parser for the 2-or-3-metric-per-dataset tables ----
def parse_metric_table(label, per_ds_cols, take):
    """take: list of (out_name, col_index_within_dataset_block)."""
    out = {}
    cur, level = None, None
    for toks in table_body(label):
        name = toks[0]
        if name == "Benign":
            continue
        if name in ATTACKS and name != "DFBA":
            cur, level = name, "high"
        elif name == "DFBA":
            cur, level = "DFBA", "none"
        else:
            level = "low"
        for k, ds in enumerate(DATASETS):
            base = 2 + per_ds_cols * k
            for out_name, off in take:
                out[(ds, cur, level, out_name)] = fnum(toks[base + off])
    return out


feat = parse_metric_table("tab:results_feature_resnet18", 3, [("SS", 0), ("DSWD", 1)])  # skip BMS (off 2)
para = parse_metric_table("tab:uclc-tac", 2, [("UCLC", 0), ("TAC", 1)])
new = parse_metric_table("tab:results_newmetric_resnet18", 2, [("CDBI", 0), ("TUP", 1)])


# ---- assemble matrix ----
PR_NOMINAL = {"high": 0.05, "low": 0.003, "none": ""}
rows = []
for attack in ATTACKS:
    levels = ["none"] if attack == "DFBA" else ["high", "low"]
    for ds in DATASETS:
        for level in levels:
            ba, asr = perf.get((ds, attack, level), ("", ""))
            row = {"Dataset": ds, "Attack": KEY[attack], "PR_level": level,
                   "PR": PR_NOMINAL[level], "BA": ba, "ASR": asr}
            row.update(input_for(ds, attack))
            row["SS"] = feat.get((ds, attack, level, "SS"), "")
            row["DSWD"] = feat.get((ds, attack, level, "DSWD"), "")
            row["CDBI"] = new.get((ds, attack, level, "CDBI"), "")
            row["UCLC"] = para.get((ds, attack, level, "UCLC"), "")
            row["TAC"] = para.get((ds, attack, level, "TAC"), "")
            row["TUP"] = new.get((ds, attack, level, "TUP"), "")
            rows.append(row)

# ---- correct the known paper typo: Imagenette/Blend l1 = 1.62e-3 -> 1.62e3 ----
for row in rows:
    if row["Dataset"] == "Imagenette" and row["Attack"] == "blended" and row["l1"] == 1.62e-3:
        row["l1"] = 1.62e3
        print("CORRECTED paper typo: Imagenette/blended l1 1.62e-3 -> 1.62e3")

cols = (["Dataset", "Attack", "PR_level", "PR", "BA", "ASR"] + INPUT_METRICS
        + ["SS", "DSWD", "CDBI", "UCLC", "TAC", "TUP"])
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Wrote {len(rows)} rows x {len(cols)} cols -> {OUT}")

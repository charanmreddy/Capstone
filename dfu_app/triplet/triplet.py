"""Builds test_triplets/case_XXX_<truth>/ folders (rgb + thermal + plantar) from the pseudo-pairing CSV.
Usage:  python make_test_triplets.py "<path to Binary_Classification>" [cases_per_class=10]
Only HOLDOUT pairs are used (images the models were not trained on)."""
import os, re, shutil, sys
import pandas as pd

ROOT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = os.path.join(os.getcwd(), "test_triplets")
J = lambda *p: os.path.join(ROOT, *p)

paired = pd.read_csv(J("CrossModelPairing", "paired_holdout.csv"))
th = pd.read_csv(J("Thermal", "thermal_features.csv"))
pl = pd.read_csv(J("Plantar", "plantar_features.csv"))
# same indexing the pairing/fusion notebooks used: split-filtered + reset_index
th = th[th["split"] == "val"].reset_index(drop=True)
pl = pl[pl["split"] == "test"].reset_index(drop=True)

# RGB: cleaned (RGB_C) copies, because the app applies the preprocessing itself
rgb_c = J("RGB", "RGB_C")
by_name = {}
for r, _, fs in os.walk(rgb_c):
    for f in fs:
        by_name.setdefault(f, os.path.join(r, f))


def find_rgb(p):
    rel = re.split(r"RGB_P[\\/]", p.replace("\\", "/"), maxsplit=1)[-1]
    cand = os.path.join(rgb_c, *rel.split("/"))
    return cand if os.path.exists(cand) else by_name.get(os.path.basename(p))


def pick(df):
    return df.sample(min(N, len(df)), random_state=42)


sel = pd.concat([pick(paired[paired.fused_label == 1]), pick(paired[paired.fused_label == 0])])
shutil.rmtree(OUT, ignore_errors=True)
rows, missing = [], 0
for i, (_, r) in enumerate(sel.iterrows(), 1):
    t, p = th.loc[r.thermal_row_idx], pl.loc[r.plantar_row_idx]
    src = {"rgb": find_rgb(r.rgb_path),
           "thermal": J("Thermal", "Thermal_C", "val", "DM Group" if t["label"] == "DM" else "Control Group", t["filename"]),
           "plantar": J("Plantar", "Plantar_C", "test", p["label"], p["filename"])}
    if not all(s and os.path.exists(s) for s in src.values()):
        missing += 1
        print("skip (file not found):", {k: v for k, v in src.items() if not (v and os.path.exists(v))})
        continue
    truth = "DFU" if r.fused_label == 1 else "NoDFU"
    d = os.path.join(OUT, f"case_{i:02d}_{truth}")
    os.makedirs(d)
    for k, s in src.items():
        shutil.copy(s, os.path.join(d, k + os.path.splitext(s)[1]))
    rows.append({"case": os.path.basename(d), "truth_(RGB_label)": truth, "thermal_group": t["label"], "plantar_type": p["label"]})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
print(f"Done: {len(rows)} cases in {OUT} ({missing} skipped)")
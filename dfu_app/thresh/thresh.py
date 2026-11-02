import pandas as pd, numpy as np
from sklearn.metrics import roc_curve, accuracy_score, recall_score, f1_score

d = pd.read_csv("fusion_holdout_probs.csv")
w = np.array([0.898, 0.055, 0.046]); w /= w.sum()
p = d[["p_rgb", "p_thermal_risk", "p_plantar_risk"]].values @ w
y = d["fused_label"].values

fpr, tpr, thr = roc_curve(y, p)
t = thr[np.argmax(tpr - fpr)]
for name, th in [("default 0.5", 0.5), ("Youden", t)]:
    pr = (p >= th).astype(int)
    print(f"{name}: thr={th:.4f} acc={accuracy_score(y, pr):.4f} "
          f"sens={recall_score(y, pr):.4f} spec={recall_score(y, pr, pos_label=0):.4f} F1={f1_score(y, pr):.4f}")
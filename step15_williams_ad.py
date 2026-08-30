"""
IMPPAT QSPR study — Step 15: Applicability Domain (Williams Plot)
====================================================================
For each of the 8 properties, compute leverage (hat values) and standardized
residuals for every compound (working-set CV + blind-set predictions), and
classify compounds against the Williams-plot AD boundaries.

Leverage:
    Hat matrix is built from the STANDARDIZED final 9-descriptor panel, fit
    (mean/SD, and (X'X)^-1) on the WORKING SET ONLY -- consistent with the
    leakage-free principle (Step 2-4): AD boundaries are defined from training
    data, then every compound (working + blind) is projected onto that space.
        h_i = x_i' (X_train' X_train)^-1 x_i
    Warning leverage threshold:
        h* = 3(p+1)/n,   p = 9 final descriptors,  n = 1202 (working-set size
        used to define the AD reference space)

Standardized residuals:
    SR_i = residual_i / SD(residual_working-set,CV)     [residual = Observed - Predicted]
    (per property; the working-set CV residual SD is used as the reference scale
    for both working-set and blind-set compounds, since the working-set CV
    residuals are the closest available "training" residual distribution)

Compound classification:
    - Normal:            h_i <= h*  and |SR_i| <= 3
    - High leverage:      h_i > h*  and |SR_i| <= 3   (structurally novel but well predicted)
    - Response outlier:   h_i <= h*  and |SR_i| > 3   (within AD but poorly predicted)
    - Outside AD:         h_i > h*  and |SR_i| > 3    (structurally novel AND poorly predicted)

Outputs:
    figures/Figure7_Williams_Plots.png                  (8-panel Williams plot)
    IMPPAT_Step15_Applicability_Domain.xlsx              (Table + compound-level leverage/SR + flagged list)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORKING_FILE = "/mnt/user-data/uploads/IMPPAT_Working_Set_1202.xlsx"
BLIND_FILE = "/mnt/user-data/uploads/IMPPAT_Blind_Set_133.xlsx"
STEP5_FILE = "/mnt/user-data/uploads/IMPPAT_Step5_CV_Ensemble_Results.xlsx"
STEP11_FILE = "/mnt/user-data/uploads/IMPPAT_Step11_Blind_Validation.xlsx"

FIG_DIR = "/home/claude/work/figures"
OUT_DIR = "/home/claude/work/outputs"

FINAL_DESCRIPTORS = [
    "Narumi_Katayama_index", "Multiplicative_Zagreb1", "Multiplicative_Zagreb2",
    "Mostar_index", "Szeged_index", "Balaban_J_index",
    "Average_eccentricity", "Sigma_index", "Spectral_radius",
]
PROPERTIES = [
    "Molecular_Weight", "Polar_Area_TPSA", "Complexity_BertzCT", "XLogP_Crippen",
    "Heavy_Atom_Count", "H_Bond_Donor_Count", "H_Bond_Acceptor_Count", "Rotatable_Bond_Count",
]
PROPERTY_LABELS = {
    "Molecular_Weight": "Molecular Weight", "Polar_Area_TPSA": "Polar Surface Area (TPSA)",
    "Complexity_BertzCT": "Complexity (BertzCT)", "XLogP_Crippen": "XLogP (Crippen)",
    "Heavy_Atom_Count": "Heavy Atom Count", "H_Bond_Donor_Count": "H-Bond Donor Count",
    "H_Bond_Acceptor_Count": "H-Bond Acceptor Count", "Rotatable_Bond_Count": "Rotatable Bond Count",
}
PANEL_LETTERS = list("ABCDEFGH")

p_final = len(FINAL_DESCRIPTORS)
n_working = 1202
h_star = 3 * (p_final + 1) / n_working

# ------------------------------------------------------------------
# Leverage: fit standardization + hat-matrix core on working set, apply to all
# ------------------------------------------------------------------
work_df = pd.read_excel(WORKING_FILE)
blind_df = pd.read_excel(BLIND_FILE)

X_train_raw = work_df[FINAL_DESCRIPTORS].values
mu = X_train_raw.mean(axis=0)
sigma = X_train_raw.std(axis=0, ddof=1)

X_train_std = (X_train_raw - mu) / sigma
XtX_inv = np.linalg.inv(X_train_std.T @ X_train_std)


def leverage(X_raw):
    X_std = (X_raw - mu) / sigma
    return np.einsum("ij,jk,ik->i", X_std, XtX_inv, X_std)


h_working = leverage(work_df[FINAL_DESCRIPTORS].values)
h_blind = leverage(blind_df[FINAL_DESCRIPTORS].values)

work_ids = work_df["IMPPAT Phytochemical identifier"].values
blind_ids = blind_df["IMPPAT Phytochemical identifier"].values

# ------------------------------------------------------------------
# Load CV (OOF) and blind residuals for each property
# ------------------------------------------------------------------
def load_pred(filepath, prefix, prop):
    df = pd.read_excel(filepath, sheet_name=f"{prefix}_{prop}")
    df = df.rename(columns={"Pred_Ensemble": "Predicted"})
    df["Residual"] = df["Observed"] - df["Predicted"]
    return df.set_index("IMPPAT Phytochemical identifier")


all_compound_rows = []
summary_rows = []

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, prop in enumerate(PROPERTIES):
    cv_pred = load_pred(STEP5_FILE, "OOF", prop)
    bl_pred = load_pred(STEP11_FILE, "Blind", prop)

    sd_cv_resid = cv_pred["Residual"].std(ddof=1)

    sr_work = (cv_pred.loc[work_ids, "Residual"].values) / sd_cv_resid
    sr_blind = (bl_pred.loc[blind_ids, "Residual"].values) / sd_cv_resid

    def classify(h, sr):
        if h > h_star and abs(sr) > 3:
            return "Outside AD"
        elif h > h_star:
            return "High leverage"
        elif abs(sr) > 3:
            return "Response outlier"
        else:
            return "Normal"

    for cid, h, sr in zip(work_ids, h_working, sr_work):
        all_compound_rows.append({
            "Property": prop, "Set": "Working (CV)", "Compound_ID": cid,
            "Leverage_h": h, "Standardized_Residual": sr, "Classification": classify(h, sr),
        })
    for cid, h, sr in zip(blind_ids, h_blind, sr_blind):
        all_compound_rows.append({
            "Property": prop, "Set": "Blind", "Compound_ID": cid,
            "Leverage_h": h, "Standardized_Residual": sr, "Classification": classify(h, sr),
        })

    n_high_lev = np.sum(h_working > h_star) + np.sum(h_blind > h_star)
    n_resp_out = np.sum(np.abs(sr_work) > 3) + np.sum(np.abs(sr_blind) > 3)
    n_outside_ad = sum(1 for r in all_compound_rows[-(len(work_ids)+len(blind_ids)):]
                        if r["Property"] == prop and r["Classification"] == "Outside AD")
    n_total = len(work_ids) + len(blind_ids)
    summary_rows.append({
        "Property": prop, "h_star": h_star, "n_total": n_total,
        "N_high_leverage(h>h*)": int(n_high_lev),
        "N_response_outliers(|SR|>3)": int(n_resp_out),
        "N_outside_AD(both)": int(n_outside_ad),
        "N_normal": int(n_total - int(n_high_lev) - int(n_resp_out) + int(n_outside_ad)),
    })

    # Williams plot panel
    ax = axes[i]
    ax.scatter(h_working, sr_work, s=12, alpha=0.4, color="#2C6E9E", label="Working (CV)")
    ax.scatter(h_blind, sr_blind, s=16, alpha=0.6, color="#C1502E", label="Blind", marker="^")
    ax.axvline(h_star, color="grey", ls="--", lw=1)
    ax.axhline(3, color="red", ls=":", lw=1)
    ax.axhline(-3, color="red", ls=":", lw=1)
    ax.set_title(f"({PANEL_LETTERS[i]}) {PROPERTY_LABELS[prop]}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Leverage (h)", fontsize=9)
    ax.set_ylabel("Standardized residual", fontsize=9)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

fig.suptitle(f"Figure 7. Williams Plots — Applicability Domain (h* = {h_star:.4f})",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIG_DIR}/Figure7_Williams_Plots.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR}/Figure7_Williams_Plots.png")

compound_df = pd.DataFrame(all_compound_rows)
summary_df = pd.DataFrame(summary_rows)

flagged_df = compound_df[compound_df["Classification"] != "Normal"].copy()

with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_Step15_Applicability_Domain.xlsx", engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="AD_Summary_by_Property", index=False)
    flagged_df.to_excel(writer, sheet_name="Flagged_Compounds", index=False)
    # full per-compound leverage/SR data, one sheet per property (kept compact)
    for prop in PROPERTIES:
        sub = compound_df[compound_df["Property"] == prop].drop(columns=["Property"])
        sub.to_excel(writer, sheet_name=f"AD_{prop}"[:31], index=False)
    pd.DataFrame({
        "Parameter": ["p (final descriptors)", "n (working-set, AD reference)", "h* = 3(p+1)/n",
                      "Leverage basis", "Standardized residual basis"],
        "Value": [p_final, n_working, round(h_star, 4),
                  "Hat matrix (X'X)^-1 fit on standardized working-set descriptors only; "
                  "applied to working + blind compounds",
                  "Residual / SD(working-set CV residual), per property; applied to both sets"],
    }).to_excel(writer, sheet_name="Settings", index=False)

print("Saved IMPPAT_Step15_Applicability_Domain.xlsx")
print(f"\nh* (leverage warning threshold) = {h_star:.4f}\n")
print(summary_df.to_string(index=False))

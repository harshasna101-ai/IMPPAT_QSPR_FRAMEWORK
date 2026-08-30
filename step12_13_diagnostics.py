"""
IMPPAT QSPR study — Step 12 & Step 13
======================================
Step 12: Observed vs Predicted plots (Working/CV -> Figure 3A-H; Blind -> Figure 4A-H)
Step 13: Residual analysis (Figure 5) for the eight ensemble QSPR models

Inputs (outputs already produced from Steps 1-11):
    IMPPAT_Step5_CV_Ensemble_Results.xlsx   -> sheets 'OOF_<Property>' (working-set, out-of-fold CV predictions)
    IMPPAT_Step11_Blind_Validation.xlsx     -> sheets 'Blind_<Property>' (blind-set predictions)

All predictions used here are the ENSEMBLE predictions ('Pred_Ensemble'), since the
ensemble is the reported model in Table4/Table5. Plots are generated directly from
compound-level predictions (never fabricated from summary statistics), per the
study protocol.

Outputs:
    figures/Figure3_Observed_vs_Predicted_CV.png        (8 panels, working/CV)
    figures/Figure4_Observed_vs_Predicted_Blind.png      (8 panels, blind external)
    figures/Figure5A_Residuals_vs_Predicted.png          (8 panels, CV)
    figures/Figure5B_Residual_Distribution.png           (8 panels, CV)
    IMPPAT_Step12_ObsVsPred_Data.xlsx                    (compound-level obs/pred/residual, CV + blind)
    IMPPAT_Step13_Residual_Diagnostics.xlsx              (Table: residual summary stats, CV + blind)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CV_FILE = "/mnt/user-data/uploads/IMPPAT_Step5_CV_Ensemble_Results.xlsx"
BLIND_FILE = "/mnt/user-data/uploads/IMPPAT_Step11_Blind_Validation.xlsx"

FIG_DIR = "/home/claude/work/figures"
OUT_DIR = "/home/claude/work/outputs"

PROPERTIES = [
    "Molecular_Weight",
    "Polar_Area_TPSA",
    "Complexity_BertzCT",
    "XLogP_Crippen",
    "Heavy_Atom_Count",
    "H_Bond_Donor_Count",
    "H_Bond_Acceptor_Count",
    "Rotatable_Bond_Count",
]

PROPERTY_LABELS = {
    "Molecular_Weight": "Molecular Weight",
    "Polar_Area_TPSA": "Polar Surface Area (TPSA)",
    "Complexity_BertzCT": "Complexity (BertzCT)",
    "XLogP_Crippen": "XLogP (Crippen)",
    "Heavy_Atom_Count": "Heavy Atom Count",
    "H_Bond_Donor_Count": "H-Bond Donor Count",
    "H_Bond_Acceptor_Count": "H-Bond Acceptor Count",
    "Rotatable_Bond_Count": "Rotatable Bond Count",
}

PANEL_LETTERS = list("ABCDEFGH")


# ------------------------------------------------------------------
# Helper: load compound-level Observed / Pred_Ensemble for each property
# ------------------------------------------------------------------
def load_predictions(filepath, sheet_prefix):
    """Returns dict: property -> DataFrame[id, Observed, Predicted]"""
    data = {}
    for prop in PROPERTIES:
        sheet = f"{sheet_prefix}_{prop}"
        df = pd.read_excel(filepath, sheet_name=sheet)
        df = df.rename(columns={"Pred_Ensemble": "Predicted"})
        df = df[["IMPPAT Phytochemical identifier", "Observed", "Predicted"]].copy()
        df["Residual"] = df["Observed"] - df["Predicted"]
        data[prop] = df
    return data


def r2_rmse_mae(obs, pred):
    r2 = stats.pearsonr(obs, pred)[0] ** 2
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    return r2, rmse, mae


cv_data = load_predictions(CV_FILE, "OOF")
blind_data = load_predictions(BLIND_FILE, "Blind")

# ------------------------------------------------------------------
# STEP 12 — Observed vs Predicted plots
# ------------------------------------------------------------------
def plot_obs_vs_pred(data_dict, title, outfile, color, letters=PANEL_LETTERS):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, prop in enumerate(PROPERTIES):
        ax = axes[i]
        df = data_dict[prop]
        obs, pred = df["Observed"].values, df["Predicted"].values
        r2, rmse, mae = r2_rmse_mae(obs, pred)

        ax.scatter(obs, pred, s=14, alpha=0.45, color=color, edgecolors="none")

        lo = min(obs.min(), pred.min())
        hi = max(obs.max(), pred.max())
        pad = 0.03 * (hi - lo if hi > lo else 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(f"({letters[i]}) {PROPERTY_LABELS[prop]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Observed", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.text(
            0.05, 0.95,
            f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(obs)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="grey"),
        )
        ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


plot_obs_vs_pred(
    cv_data,
    "Figure 3. Observed vs Predicted — Working Set (Out-of-Fold 5-Fold CV, Ensemble Model)",
    f"{FIG_DIR}/Figure3_Observed_vs_Predicted_CV.png",
    color="#2C6E9E",
)

plot_obs_vs_pred(
    blind_data,
    "Figure 4. Observed vs Predicted — Blind External Validation Set (Ensemble Model)",
    f"{FIG_DIR}/Figure4_Observed_vs_Predicted_Blind.png",
    color="#C1502E",
)

# Save compound-level obs/pred/residual data (both CV and blind) for the manuscript SI
with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_Step12_ObsVsPred_Data.xlsx", engine="openpyxl") as writer:
    summary_rows = []
    for prop in PROPERTIES:
        cv_data[prop].to_excel(writer, sheet_name=f"CV_{prop}"[:31], index=False)
        blind_data[prop].to_excel(writer, sheet_name=f"Blind_{prop}"[:31], index=False)

        r2_cv, rmse_cv, mae_cv = r2_rmse_mae(cv_data[prop]["Observed"], cv_data[prop]["Predicted"])
        r2_bl, rmse_bl, mae_bl = r2_rmse_mae(blind_data[prop]["Observed"], blind_data[prop]["Predicted"])
        summary_rows.append({
            "Property": prop, "n_CV": len(cv_data[prop]), "R2_CV": r2_cv, "RMSE_CV": rmse_cv, "MAE_CV": mae_cv,
            "n_Blind": len(blind_data[prop]), "R2_Blind": r2_bl, "RMSE_Blind": rmse_bl, "MAE_Blind": mae_bl,
        })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

print("Saved IMPPAT_Step12_ObsVsPred_Data.xlsx")

# ------------------------------------------------------------------
# STEP 13 — Residual analysis
# ------------------------------------------------------------------
# Residual diagnostics are computed on the working-set (CV) predictions, which is
# the standard basis for diagnosing model fit/heteroscedasticity; blind-set residual
# summary stats are also reported in the diagnostics table for completeness.

def residual_diagnostics(df):
    """df has Observed, Predicted, Residual columns."""
    obs, pred, resid = df["Observed"].values, df["Predicted"].values, df["Residual"].values
    mean_r = np.mean(resid)
    sd_r = np.std(resid, ddof=1)
    skew_r = stats.skew(resid)
    kurt_r = stats.kurtosis(resid)

    # heteroscedasticity proxy: correlation between |residual| and predicted value
    hetero_r, hetero_p = stats.pearsonr(np.abs(resid), pred)

    # extreme residuals: |residual| > 3 SD from mean
    extreme_mask = np.abs(resid - mean_r) > 3 * sd_r
    n_extreme = int(extreme_mask.sum())

    return {
        "n": len(resid),
        "Mean_residual": mean_r,
        "SD_residual": sd_r,
        "Skewness": skew_r,
        "Kurtosis": kurt_r,
        "Hetero_corr(|resid|,pred)": hetero_r,
        "Hetero_p_value": hetero_p,
        "Heteroscedastic_flag(p<0.05)": "Yes" if hetero_p < 0.05 else "No",
        "N_extreme_residuals(>3SD)": n_extreme,
    }, extreme_mask


def plot_residual_vs_predicted(data_dict, outfile, color):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, prop in enumerate(PROPERTIES):
        ax = axes[i]
        df = data_dict[prop]
        pred, resid = df["Predicted"].values, df["Residual"].values
        ax.scatter(pred, resid, s=14, alpha=0.45, color=color, edgecolors="none")
        ax.axhline(0, color="k", ls="--", lw=1)
        sd = np.std(resid, ddof=1)
        ax.axhline(3 * sd, color="red", ls=":", lw=1)
        ax.axhline(-3 * sd, color="red", ls=":", lw=1)
        ax.set_title(f"({PANEL_LETTERS[i]}) {PROPERTY_LABELS[prop]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Residual (Obs - Pred)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Figure 5A. Residual vs Predicted — Working Set (CV, Ensemble Model)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_residual_distribution(data_dict, outfile, color):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, prop in enumerate(PROPERTIES):
        ax = axes[i]
        resid = data_dict[prop]["Residual"].values
        ax.hist(resid, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(f"({PANEL_LETTERS[i]}) {PROPERTY_LABELS[prop]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Residual (Obs - Pred)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Figure 5B. Residual Distribution — Working Set (CV, Ensemble Model)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


plot_residual_vs_predicted(cv_data, f"{FIG_DIR}/Figure5A_Residuals_vs_Predicted.png", color="#2C6E9E")
plot_residual_distribution(cv_data, f"{FIG_DIR}/Figure5B_Residual_Distribution.png", color="#2C6E9E")

# Build Table: residual diagnostics for CV and Blind sets
cv_rows, blind_rows = [], []
extreme_compounds = []

for prop in PROPERTIES:
    diag_cv, mask_cv = residual_diagnostics(cv_data[prop])
    diag_cv["Property"] = prop
    diag_cv["Set"] = "Working (CV)"
    cv_rows.append(diag_cv)

    diag_bl, mask_bl = residual_diagnostics(blind_data[prop])
    diag_bl["Property"] = prop
    diag_bl["Set"] = "Blind"
    blind_rows.append(diag_bl)

    flagged = cv_data[prop].loc[mask_cv, ["IMPPAT Phytochemical identifier", "Observed", "Predicted", "Residual"]].copy()
    flagged.insert(0, "Property", prop)
    extreme_compounds.append(flagged)

diag_table = pd.DataFrame(cv_rows + blind_rows)
col_order = ["Property", "Set", "n", "Mean_residual", "SD_residual", "Skewness", "Kurtosis",
             "Hetero_corr(|resid|,pred)", "Hetero_p_value", "Heteroscedastic_flag(p<0.05)",
             "N_extreme_residuals(>3SD)"]
diag_table = diag_table[col_order]

extreme_df = pd.concat(extreme_compounds, ignore_index=True) if extreme_compounds else pd.DataFrame()

with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_Step13_Residual_Diagnostics.xlsx", engine="openpyxl") as writer:
    diag_table.to_excel(writer, sheet_name="Table_Residual_Diagnostics", index=False)
    if not extreme_df.empty:
        extreme_df.to_excel(writer, sheet_name="Extreme_Residuals_CV", index=False)
    else:
        pd.DataFrame({"Note": ["No extreme residuals (>3 SD) found in the working/CV set."]}).to_excel(
            writer, sheet_name="Extreme_Residuals_CV", index=False
        )

print("Saved IMPPAT_Step13_Residual_Diagnostics.xlsx")
print("\nResidual diagnostics (CV set):")
print(diag_table[diag_table["Set"] == "Working (CV)"].to_string(index=False))

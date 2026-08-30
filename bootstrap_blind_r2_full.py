"""
Bootstrap 95% CI for blind-set R^2 -- all 9 modeled properties
(8 physicochemical properties + QEDw)

INPUTS (edit paths if needed):
  1) IMPPAT_Step12_ObsVsPred_Data.xlsx
     -> sheets "Blind_<Property>" with columns:
        IMPPAT Phytochemical identifier | Observed | Predicted | Residual
     -> sheets "CV_<Property>" with the same columns (out-of-fold CV preds)

  2) IMPPAT_Step18_QEDw_Modelling.xlsx
     -> sheet "Blind_QEDw" with columns:
        IMPPAT Phytochemical identifier | Observed | Pred_RF | Pred_GB | Pred_XGB | Pred_Ensemble
     -> sheet "OOF_QEDw" with the same columns (CV out-of-fold)

OUTPUT:
  blind_r2_bootstrap_ci_full.csv -- one row per property with:
    CV_R2, Blind_R2, 95% CI (lower/upper), bootstrap SE,
    and whether the CV point estimate falls inside the blind CI.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ============================================================
# FILE PATHS -- edit these if your files are located elsewhere
# ============================================================
PHYSCHEM_FILE = "/mnt/user-data/uploads/1788099883186_IMPPAT_Step12_ObsVsPred_Data.xlsx"
QEDW_FILE     = "/mnt/user-data/uploads/IMPPAT_Step18_QEDw_Modelling.xlsx"

N_BOOT = 10000
CI_LEVEL = 95
SEED = 42   # matches the manuscript's master seed (Section 2.15)

# Map: display name (as used in the manuscript tables) -> sheet suffix
PHYSCHEM_PROPERTIES = {
    "Molecular Weight":       "Molecular_Weight",
    "Polar Area (TPSA)":      "Polar_Area_TPSA",
    "Complexity (BertzCT)":   "Complexity_BertzCT",
    "XLogP (Crippen)":        "XLogP_Crippen",
    "Heavy-Atom Count":       "Heavy_Atom_Count",
    "H-Bond Donor Count":     "H_Bond_Donor_Count",
    "H-Bond Acceptor Count":  "H_Bond_Acceptor_Count",
    "Rotatable-Bond Count":   "Rotatable_Bond_Count",
}


# ============================================================
# BOOTSTRAP CI FUNCTION
# ============================================================
def bootstrap_r2_ci(y_true, y_pred, n_boot=N_BOOT, ci=CI_LEVEL, seed=SEED):
    """
    Nonparametric bootstrap CI for R^2 on a held-out (blind) set.
    Resamples compounds (rows) with replacement -- appropriate given
    the documented heteroscedasticity (manuscript Section 3.6), rather
    than resampling residuals, which would assume constant variance.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    if len(y_pred) != n:
        raise ValueError(f"Length mismatch: y_true has {n}, y_pred has {len(y_pred)}")

    boot_r2 = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.var(yt) < 1e-12:      # guard against a degenerate resample
            boot_r2[b] = np.nan
            continue
        boot_r2[b] = r2_score(yt, yp)

    boot_r2 = boot_r2[~np.isnan(boot_r2)]
    lo = np.percentile(boot_r2, (100 - ci) / 2)
    hi = np.percentile(boot_r2, 100 - (100 - ci) / 2)
    point = r2_score(y_true, y_pred)

    return {
        "point_R2": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "boot_se": boot_r2.std(ddof=1),
        "n_valid_boot": len(boot_r2),
    }


# ============================================================
# STAGE 1: LOAD THE 8 PHYSICOCHEMICAL PROPERTIES
# ============================================================
xls_phys = pd.ExcelFile(PHYSCHEM_FILE)

blind_true, blind_pred, cv_true, cv_pred = {}, {}, {}, {}

for display_name, suffix in PHYSCHEM_PROPERTIES.items():
    bdf = pd.read_excel(xls_phys, sheet_name=f"Blind_{suffix}")
    cdf = pd.read_excel(xls_phys, sheet_name=f"CV_{suffix}")

    blind_true[display_name] = bdf["Observed"].values
    blind_pred[display_name] = bdf["Predicted"].values
    cv_true[display_name] = cdf["Observed"].values
    cv_pred[display_name] = cdf["Predicted"].values

# ============================================================
# STAGE 2: LOAD QEDw (separate workbook, "Pred_Ensemble" column)
# ============================================================
xls_qedw = pd.ExcelFile(QEDW_FILE)

blind_qedw_df = pd.read_excel(xls_qedw, sheet_name="Blind_QEDw")
oof_qedw_df = pd.read_excel(xls_qedw, sheet_name="OOF_QEDw")

blind_true["QEDw"] = blind_qedw_df["Observed"].values
blind_pred["QEDw"] = blind_qedw_df["Pred_Ensemble"].values
cv_true["QEDw"] = oof_qedw_df["Observed"].values
cv_pred["QEDw"] = oof_qedw_df["Pred_Ensemble"].values

# ============================================================
# STAGE 3: RUN BOOTSTRAP FOR ALL 9 PROPERTIES
# ============================================================
all_properties = list(PHYSCHEM_PROPERTIES.keys()) + ["QEDw"]

results = {}
cv_point = {}
for prop in all_properties:
    results[prop] = bootstrap_r2_ci(blind_true[prop], blind_pred[prop])
    cv_point[prop] = r2_score(cv_true[prop], cv_pred[prop])

# ============================================================
# STAGE 4: PRINT + EXPORT SUMMARY TABLE
# ============================================================
print(f"{'Property':25s} {'CV R2':>8s} {'Blind R2':>9s} {'95% CI':>18s} {'SE':>7s}  {'CV in CI?':>10s}")

rows = []
for prop in all_properties:
    r = results[prop]
    ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
    cv_in_ci = r["ci_lower"] <= cv_point[prop] <= r["ci_upper"]
    print(f"{prop:25s} {cv_point[prop]:8.3f} {r['point_R2']:9.3f} {ci_str:>18s} {r['boot_se']:7.3f}  {str(cv_in_ci):>10s}")
    rows.append({
        "Property": prop,
        "CV_R2": round(cv_point[prop], 3),
        "Blind_R2": round(r["point_R2"], 3),
        "CI_lower_95": round(r["ci_lower"], 3),
        "CI_upper_95": round(r["ci_upper"], 3),
        "Bootstrap_SE": round(r["boot_se"], 3),
        "CV_R2_within_blind_CI": cv_in_ci,
    })

summary_df = pd.DataFrame(rows)
out_path = "/home/claude/blind_r2_bootstrap_ci_full.csv"
summary_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

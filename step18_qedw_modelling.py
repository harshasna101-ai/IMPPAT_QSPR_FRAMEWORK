"""
Step 18 - QEDw modelling
=========================
Treats QEDw (weighted-mean QED, RDKit default) as a composite drug-likeness
endpoint (NOT an independent bioactivity endpoint) and models it using the
same final 9-descriptor topological panel and the same
Working(1202)/Blind(133) split used in Steps 5-15.

Repeats:
  18a. 5-fold CV  (RF+GB+XGB ensemble, n_estimators=300, seed=42)   -> mirrors Step 5
  18b. Blind (external) validation                                  -> mirrors Step 11
  18c. Y-randomization (200 permutations, n_estimators=30, seed=42) -> mirrors Step 14
  18d. Applicability Domain (Williams-plot leverage/residual)        -> mirrors Step 15

Input files required (all already in hand - nothing new needed from user):
  - 8_imppat_cleaned_1335.csv          (SMILES source, to compute QEDw)
  - IMPPAT_Working_Set_1202.xlsx       (defines the 1202 training/CV IDs)
  - IMPPAT_Blind_Set_133.xlsx          (defines the 133 external-test IDs)
  - Final_Panel_Data.xlsx              (9-descriptor panel values, working set)
  - IMPPAT_Final_Descriptor_Panel.xlsx (names of the 9 final descriptors)

Output:
  - IMPPAT_Step18_QEDw_Modelling.xlsx  (all sub-tables, mirrors Step 5/11/14/15 formats)
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED   = 42
N_EST_MAIN    = 300      # CV / blind validation (matches Step 5 / Step 11)
N_EST_YRAND   = 30       # Y-randomization (matches Step 14)
CV_FOLDS      = 5
N_PERMS       = 200      # Y-randomization permutations (matches Step 14)

UPLOAD_DIR = "/mnt/user-data/uploads/"
OUT_FILE   = "IMPPAT_Step18_QEDw_Modelling.xlsx"

ID_COL = "IMPPAT Phytochemical identifier"

# ---------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------
full_df   = pd.read_csv(UPLOAD_DIR + "8_imppat_cleaned_1335.csv")
work_ids  = pd.read_excel(UPLOAD_DIR + "IMPPAT_Working_Set_1202.xlsx")[ID_COL].tolist()
blind_ids = pd.read_excel(UPLOAD_DIR + "IMPPAT_Blind_Set_133.xlsx")[ID_COL].tolist()
panel_names = pd.read_excel(UPLOAD_DIR + "IMPPAT_Final_Descriptor_Panel.xlsx",
                             sheet_name="Final_Descriptor_Panel")["Descriptor"].tolist()

print(f"Total compounds: {len(full_df)}")
print(f"Working set: {len(work_ids)} | Blind set: {len(blind_ids)}")
print(f"Final descriptor panel ({len(panel_names)}): {panel_names}")

# ---------------------------------------------------------------
# 1. Compute QEDw for all 1335 compounds from SMILES
# ---------------------------------------------------------------
def compute_qedw(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return QED.qed(mol)  # RDKit default = weighted-mean desirability (QEDw)
    except Exception:
        return np.nan

full_df["QEDw"] = full_df["SMILES"].apply(compute_qedw)

n_missing = full_df["QEDw"].isna().sum()
print(f"QEDw computed for {len(full_df) - n_missing}/{len(full_df)} compounds "
      f"({n_missing} failed SMILES parsing).")

if n_missing > 0:
    failed = full_df.loc[full_df["QEDw"].isna(), [ID_COL, "Chemical name", "SMILES"]]
else:
    failed = pd.DataFrame(columns=[ID_COL, "Chemical name", "SMILES"])

# ---------------------------------------------------------------
# 2. Build working/blind design matrices (panel descriptors + QEDw)
#    using the SAME split as Steps 5-15
# ---------------------------------------------------------------
qedw_lookup = full_df.set_index(ID_COL)["QEDw"]

# descriptor values: pull straight from the full cleaned file so we have
# every panel descriptor for both working and blind IDs in one place
desc_df = full_df.set_index(ID_COL)[panel_names]

work_df = desc_df.loc[work_ids].copy()
work_df["QEDw"] = qedw_lookup.loc[work_ids]
work_df = work_df.dropna(subset=["QEDw"] + panel_names)

blind_df = desc_df.loc[blind_ids].copy()
blind_df["QEDw"] = qedw_lookup.loc[blind_ids]
blind_df = blind_df.dropna(subset=["QEDw"] + panel_names)

print(f"Usable working set (complete QEDw+descriptors): {len(work_df)}")
print(f"Usable blind set  (complete QEDw+descriptors): {len(blind_df)}")

X_work = work_df[panel_names].values
y_work = work_df["QEDw"].values
work_index = work_df.index.values

X_blind = blind_df[panel_names].values
y_blind = blind_df["QEDw"].values
blind_index = blind_df.index.values


def make_models(n_estimators, seed):
    rf  = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    gb  = GradientBoostingRegressor(n_estimators=n_estimators, random_state=seed)
    xgb = XGBRegressor(n_estimators=n_estimators, random_state=seed,
                        verbosity=0, n_jobs=-1)
    return {"RF": rf, "GB": gb, "XGB": xgb}


# =================================================================
# 18a. 5-FOLD CROSS-VALIDATION (mirrors Step 5)
# =================================================================
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

oof_pred = {m: np.full(len(y_work), np.nan) for m in ["RF", "GB", "XGB"]}
per_fold_rows = []

for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X_work), start=1):
    Xtr, Xte = X_work[tr_idx], X_work[te_idx]
    ytr, yte = y_work[tr_idx], y_work[te_idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    models = make_models(N_EST_MAIN, RANDOM_SEED)
    fold_preds = {}
    for name, mdl in models.items():
        mdl.fit(Xtr_s, ytr)
        pred = mdl.predict(Xte_s)
        fold_preds[name] = pred
        oof_pred[name][te_idx] = pred

    ens_pred = np.mean([fold_preds[m] for m in ["RF", "GB", "XGB"]], axis=0)
    per_fold_rows.append({
        "Property": "QEDw", "Fold": fold_i,
        "RF R2": r2_score(yte, fold_preds["RF"]),
        "GB R2": r2_score(yte, fold_preds["GB"]),
        "XGB R2": r2_score(yte, fold_preds["XGB"]),
        "Ensemble R2": r2_score(yte, ens_pred),
        "Ensemble RMSE": np.sqrt(mean_squared_error(yte, ens_pred)),
        "Ensemble MAE": mean_absolute_error(yte, ens_pred),
    })

oof_ensemble = np.mean([oof_pred[m] for m in ["RF", "GB", "XGB"]], axis=0)

cv_summary = pd.DataFrame([{
    "Property": "QEDw",
    "RF R2": r2_score(y_work, oof_pred["RF"]),
    "GB R2": r2_score(y_work, oof_pred["GB"]),
    "XGB R2": r2_score(y_work, oof_pred["XGB"]),
    "Ensemble R2": r2_score(y_work, oof_ensemble),
    "Q2_CV": r2_score(y_work, oof_ensemble),   # Q2 = OOF R2, consistent with Step 5 definition
    "RMSE": np.sqrt(mean_squared_error(y_work, oof_ensemble)),
    "MAE": mean_absolute_error(y_work, oof_ensemble),
}])

oof_table = pd.DataFrame({
    ID_COL: work_index,
    "Observed": y_work,
    "Pred_RF": oof_pred["RF"],
    "Pred_GB": oof_pred["GB"],
    "Pred_XGB": oof_pred["XGB"],
    "Pred_Ensemble": oof_ensemble,
})
per_fold_df = pd.DataFrame(per_fold_rows)

print("\n--- 18a. CV performance (QEDw) ---")
print(cv_summary.to_string(index=False))

# =================================================================
# 18b. BLIND (EXTERNAL) VALIDATION (mirrors Step 11)
# =================================================================
scaler_full = StandardScaler().fit(X_work)
X_work_s  = scaler_full.transform(X_work)
X_blind_s = scaler_full.transform(X_blind)

final_models = make_models(N_EST_MAIN, RANDOM_SEED)
blind_preds = {}
for name, mdl in final_models.items():
    mdl.fit(X_work_s, y_work)
    blind_preds[name] = mdl.predict(X_blind_s)

blind_ensemble = np.mean([blind_preds[m] for m in ["RF", "GB", "XGB"]], axis=0)

blind_table = pd.DataFrame({
    ID_COL: blind_index,
    "Observed": y_blind,
    "Pred_RF": blind_preds["RF"],
    "Pred_GB": blind_preds["GB"],
    "Pred_XGB": blind_preds["XGB"],
    "Pred_Ensemble": blind_ensemble,
})

blind_summary = pd.DataFrame([{
    "Property": "QEDw",
    "CV R2": cv_summary["Ensemble R2"].iloc[0],
    "CV RMSE": cv_summary["RMSE"].iloc[0],
    "Blind R2": r2_score(y_blind, blind_ensemble),
    "Blind RMSE": np.sqrt(mean_squared_error(y_blind, blind_ensemble)),
    "Blind MAE": mean_absolute_error(y_blind, blind_ensemble),
}])

print("\n--- 18b. Blind validation (QEDw) ---")
print(blind_summary.to_string(index=False))

# =================================================================
# 18c. Y-RANDOMIZATION (mirrors Step 14)
# =================================================================
Xtr_y, Xte_y, ytr_y, yte_y = train_test_split(
    X_work, y_work, test_size=0.20, random_state=RANDOM_SEED
)
scaler_y = StandardScaler().fit(Xtr_y)
Xtr_y_s, Xte_y_s = scaler_y.transform(Xtr_y), scaler_y.transform(Xte_y)

# actual (non-permuted) performance at the reduced n_estimators, fixed split
models_actual = make_models(N_EST_YRAND, RANDOM_SEED)
actual_preds = {}
for name, mdl in models_actual.items():
    mdl.fit(Xtr_y_s, ytr_y)
    actual_preds[name] = mdl.predict(Xte_y_s)
actual_ens = np.mean([actual_preds[m] for m in ["RF", "GB", "XGB"]], axis=0)
actual_r2_fixedsplit = r2_score(yte_y, actual_ens)

rng = np.random.RandomState(RANDOM_SEED)
perm_r2 = []
for p in range(N_PERMS):
    y_perm_tr = rng.permutation(ytr_y)
    models_p = make_models(N_EST_YRAND, RANDOM_SEED)
    preds_p = {}
    for name, mdl in models_p.items():
        mdl.fit(Xtr_y_s, y_perm_tr)
        preds_p[name] = mdl.predict(Xte_y_s)
    ens_p = np.mean([preds_p[m] for m in ["RF", "GB", "XGB"]], axis=0)
    perm_r2.append(r2_score(yte_y, ens_p))
perm_r2 = np.array(perm_r2)

yrand_table = pd.DataFrame([{
    "Property": "QEDw",
    "Actual_Ensemble_R2_CV": cv_summary["Ensemble R2"].iloc[0],   # true benchmark (300 est, 5-fold)
    "Actual_Q2_CV": cv_summary["Q2_CV"].iloc[0],
    "Mean_Randomized_R2": perm_r2.mean(),
    "SD_Randomized_R2": perm_r2.std(),
    "Max_Randomized_R2": perm_r2.max(),
    "Min_Randomized_R2": perm_r2.min(),
    "N_Permutations": N_PERMS,
    "Permutation_p_value(R2>=actual)": (np.sum(perm_r2 >= actual_r2_fixedsplit) + 1) / (N_PERMS + 1),
}])
raw_perm_df = pd.DataFrame({"Permutation": np.arange(1, N_PERMS + 1), "QEDw": perm_r2})

print("\n--- 18c. Y-randomization (QEDw) ---")
print(yrand_table.to_string(index=False))

# =================================================================
# 18d. APPLICABILITY DOMAIN / Williams plot (mirrors Step 15)
# =================================================================
p = len(panel_names)
n_ref = len(work_df)
h_star = 3 * (p + 1) / n_ref

# Hat matrix on standardized working-set descriptors
X_ref_s = StandardScaler().fit(X_work).transform(X_work)
hat_core = np.linalg.pinv(X_ref_s.T @ X_ref_s)

def leverage(x_row_std):
    return float(x_row_std @ hat_core @ x_row_std.T)

lev_work = np.array([leverage(x) for x in X_ref_s])
X_blind_s_ad = StandardScaler().fit(X_work).transform(X_blind)
lev_blind = np.array([leverage(x) for x in X_blind_s_ad])

# standardized residuals: residual / SD(working-set CV residual)
resid_work = y_work - oof_ensemble
sd_resid = resid_work.std()
sr_work = resid_work / sd_resid

resid_blind = y_blind - blind_ensemble
sr_blind = resid_blind / sd_resid

def classify(h, sr, h_star):
    high_lev = h > h_star
    outlier  = abs(sr) > 3
    if high_lev and outlier:
        return "Outside AD (both)"
    elif high_lev:
        return "High leverage"
    elif outlier:
        return "Response outlier"
    else:
        return "Normal"

ad_work = pd.DataFrame({
    "Set": "Working", "Compound_ID": work_index,
    "Leverage_h": lev_work, "Standardized_Residual": sr_work,
})
ad_work["Classification"] = ad_work.apply(
    lambda r: classify(r["Leverage_h"], r["Standardized_Residual"], h_star), axis=1)

ad_blind = pd.DataFrame({
    "Set": "Blind", "Compound_ID": blind_index,
    "Leverage_h": lev_blind, "Standardized_Residual": sr_blind,
})
ad_blind["Classification"] = ad_blind.apply(
    lambda r: classify(r["Leverage_h"], r["Standardized_Residual"], h_star), axis=1)

ad_full = pd.concat([ad_work, ad_blind], ignore_index=True)

ad_summary = pd.DataFrame([{
    "Property": "QEDw",
    "h_star": h_star,
    "n_total": len(ad_full),
    "N_high_leverage(h>h*)": (ad_full["Leverage_h"] > h_star).sum(),
    "N_response_outliers(|SR|>3)": (ad_full["Standardized_Residual"].abs() > 3).sum(),
    "N_outside_AD(both)": (ad_full["Classification"] == "Outside AD (both)").sum(),
    "N_normal": (ad_full["Classification"] == "Normal").sum(),
}])

flagged = ad_full[ad_full["Classification"] != "Normal"].copy()
flagged.insert(0, "Property", "QEDw")

print("\n--- 18d. Applicability Domain (QEDw) ---")
print(ad_summary.to_string(index=False))

# =================================================================
# Settings / provenance sheets
# =================================================================
settings_main = pd.DataFrame({
    "Parameter": ["n_estimators (CV/Blind)", "Random seed", "CV folds",
                  "Descriptor panel size", "Working set size", "Blind set size",
                  "QEDw source", "QEDw weighting"],
    "Value": [N_EST_MAIN, RANDOM_SEED, CV_FOLDS, p, len(work_df), len(blind_df),
              "RDKit Chem.QED.qed()", "Weighted-mean desirability (default RDKit QEDw)"]
})
settings_yrand = pd.DataFrame({
    "Parameter": ["N_permutations", "N_estimators (RF/GB/XGB)", "Train/holdout split",
                  "Random state", "Note"],
    "Value": [N_PERMS, N_EST_YRAND, "80/20 fixed split (reused across permutations, only target shuffled)",
              RANDOM_SEED,
              "Same reduced-cost protocol as Step 14; benchmark R2/Q2 taken from full 300-estimator 5-fold CV (18a)."]
})
settings_ad = pd.DataFrame({
    "Parameter": ["p (final descriptors)", "n (working-set, AD reference)", "h* = 3(p+1)/n",
                  "Leverage basis", "Standardized residual basis"],
    "Value": [p, n_ref, h_star,
              "Hat matrix (X'X)^-1 fit on standardized working-set descriptors only; applied to working + blind compounds",
              "Residual / SD(working-set CV residual), applied to both sets"]
})

caveat_note = pd.DataFrame({
    "Caveat": [
        "QEDw is a composite drug-likeness index",
        "Not independent biological validation",
        "Interpretation"
    ],
    "Note": [
        "QEDw is calculated from molecular weight, ALogP, TPSA, HBD, HBA, rotatable bonds, "
        "aromatic ring count and structural alerts (Bickerton et al. 2012), several of which "
        "overlap conceptually with the properties already modelled in Steps 5-15.",
        "Because QEDw is built from physicochemical descriptors rather than an assayed "
        "biological endpoint, strong performance here demonstrates that topological "
        "descriptors can model a composite drug-likeness score - it does NOT constitute "
        "independent evidence of bioactivity prediction.",
        "Results should be reported as: 'the topological descriptor panel can also model "
        "QEDw (composite drug-likeness), reinforcing but not independently validating the "
        "physicochemical predictions in Steps 5-15.'"
    ]
})

qedw_full_table = full_df[[ID_COL, "Chemical name", "SMILES", "QEDw"]].copy()

# =================================================================
# Write output workbook
# =================================================================
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    qedw_full_table.to_excel(writer, sheet_name="QEDw_All_1335", index=False)
    failed.to_excel(writer, sheet_name="QEDw_Failed_SMILES", index=False)

    cv_summary.to_excel(writer, sheet_name="Table_CV_Performance", index=False)
    per_fold_df.to_excel(writer, sheet_name="Per_Fold_Detail", index=False)
    oof_table.to_excel(writer, sheet_name="OOF_QEDw", index=False)

    blind_summary.to_excel(writer, sheet_name="Table_Blind_Validation", index=False)
    blind_table.to_excel(writer, sheet_name="Blind_QEDw", index=False)

    yrand_table.to_excel(writer, sheet_name="Table_Y_Randomization", index=False)
    raw_perm_df.to_excel(writer, sheet_name="Raw_Permutation_R2", index=False)

    ad_summary.to_excel(writer, sheet_name="AD_Summary", index=False)
    ad_full.to_excel(writer, sheet_name="AD_QEDw_Full", index=False)
    flagged.to_excel(writer, sheet_name="Flagged_Compounds", index=False)

    settings_main.to_excel(writer, sheet_name="Settings_Main", index=False)
    settings_yrand.to_excel(writer, sheet_name="Settings_YRand", index=False)
    settings_ad.to_excel(writer, sheet_name="Settings_AD", index=False)
    caveat_note.to_excel(writer, sheet_name="Caveats", index=False)

print(f"\nSaved: {OUT_FILE}")

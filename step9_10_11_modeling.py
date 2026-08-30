"""
IMPPAT Descriptor QC Pipeline
=============================
Step 9  — Ensemble QSPR modeling (RF + GB + XGB)
Step 10 — Leakage-free five-fold cross-validation
Step 11 — Blind external validation

Descriptor panel: the 9 VIF-selected, non-redundant topological indices
fixed in Step 8 (Narumi_Katayama_index, Multiplicative_Zagreb1,
Mostar_index, Szeged_index, Spectral_radius, Average_eccentricity,
Sigma_index, Balaban_J_index, Multiplicative_Zagreb2).

IMPORTANT leakage note (per Step 10 instructions):
  This pipeline does NOT perform any target-dependent (supervised)
  descriptor selection -- the 9-descriptor panel was fixed entirely by
  the VIF<10 criterion in Step 4/8, which only looks at the descriptor
  matrix and never touches any of the 8 target properties (y). Because
  there is no target-dependent selection step, there is nothing that
  needs to be "re-done inside every fold" for descriptor selection --
  the panel is legitimately fixed before CV starts, exactly as Step 8
  specifies ("determined by the predefined VIF criterion").
  What DOES have to happen inside every fold (and does, below) is:
    - StandardScaler is fit on the internal training fold ONLY, then
      applied to the held-out validation fold (no scaler fit on data
      the model will later be scored on).
    - RF / GB / XGB are fit on the internal training fold ONLY.
  This is the leakage-free protocol Step 10 asks for.

Targets (exact column names in the working set / blind set):
  Molecular_Weight, Polar_Area_TPSA, Complexity_BertzCT, XLogP_Crippen,
  Heavy_Atom_Count, H_Bond_Donor_Count, H_Bond_Acceptor_Count,
  Rotatable_Bond_Count

Output:
  IMPPAT_Step9_10_CV_Results.xlsx   (Table 4 + per-fold detail + raw OOF predictions)
  IMPPAT_Step11_Blind_Validation.xlsx (Table 5 + raw blind predictions)
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

OUT_DIR = "/mnt/user-data/outputs"
WORKING_SET_FILE = "/mnt/user-data/uploads/IMPPAT_Working_Set_1202.xlsx"
BLIND_SET_FILE = "/mnt/user-data/uploads/IMPPAT_Blind_Set_133.xlsx"

RANDOM_SEED = 42
N_ESTIMATORS = 300
N_FOLDS = 5

# ------------------------------------------------------------------
# 0. Load data and fix descriptor panel / target list
# ------------------------------------------------------------------
work_df = pd.read_excel(WORKING_SET_FILE)
blind_df = pd.read_excel(BLIND_SET_FILE)

vif_sheet = pd.read_excel(f"{OUT_DIR}/IMPPAT_VIF_Selected_Descriptors.xlsx",
                           sheet_name="VIF_Selected_Descriptors")
FINAL_PANEL = vif_sheet.loc[vif_sheet["Retained/Removed"] == "Retained", "Descriptor"].tolist()
print(f"Final descriptor panel ({len(FINAL_PANEL)}): {FINAL_PANEL}")

TARGET_PROPERTIES = [
    "Molecular_Weight", "Polar_Area_TPSA", "Complexity_BertzCT", "XLogP_Crippen",
    "Heavy_Atom_Count", "H_Bond_Donor_Count", "H_Bond_Acceptor_Count", "Rotatable_Bond_Count"
]
missing = [t for t in TARGET_PROPERTIES if t not in work_df.columns or t not in blind_df.columns]
assert not missing, f"Missing target columns: {missing}"

ID_COL = "IMPPAT Phytochemical identifier"

X_work_full = work_df[FINAL_PANEL].values
X_blind_full = blind_df[FINAL_PANEL].values
n_work = X_work_full.shape[0]
n_blind = X_blind_full.shape[0]
print(f"Working set: {n_work} compounds | Blind set: {n_blind} compounds")


def make_models(seed=RANDOM_SEED, n_estimators=N_ESTIMATORS):
    """Fresh, identically-configured RF/GB/XGB models (fixed seed)."""
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=n_estimators, random_state=seed)
    xgb = XGBRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1,
                        objective="reg:squarederror", verbosity=0)
    return {"RF": rf, "GB": gb, "XGB": xgb}


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


# ==================================================================
# STEP 9 + STEP 10 — Ensemble models, leakage-free 5-fold CV
# ==================================================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

table4_rows = []
cv_fold_detail_rows = []
oof_predictions_all = {}   # property -> DataFrame of compound-level OOF predictions

for prop in TARGET_PROPERTIES:
    print(f"\n=== CV: {prop} ===")
    y = work_df[prop].values.astype(float)

    oof_pred = {"RF": np.full(n_work, np.nan), "GB": np.full(n_work, np.nan),
                "XGB": np.full(n_work, np.nan)}

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_work_full), start=1):
        X_train, X_val = X_work_full[train_idx], X_work_full[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- preprocessing fit on internal TRAINING fold only ---
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)

        # --- no target-dependent descriptor selection step is used
        #     (panel fixed by unsupervised VIF criterion, Step 4/8) ---

        models = make_models()
        fold_metrics = {}
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            pred_val = model.predict(X_val_s)
            oof_pred[name][val_idx] = pred_val
            r2, rmse, mae = compute_metrics(y_val, pred_val)
            fold_metrics[name] = (r2, rmse, mae)

        ens_val = np.mean([oof_pred[n][val_idx] for n in ["RF", "GB", "XGB"]], axis=0)
        r2_e, rmse_e, mae_e = compute_metrics(y_val, ens_val)

        cv_fold_detail_rows.append({
            "Property": prop, "Fold": fold_i,
            "RF R2": fold_metrics["RF"][0], "GB R2": fold_metrics["GB"][0],
            "XGB R2": fold_metrics["XGB"][0], "Ensemble R2": r2_e,
            "Ensemble RMSE": rmse_e, "Ensemble MAE": mae_e,
        })
        print(f"  Fold {fold_i}: RF R2={fold_metrics['RF'][0]:.3f}  "
              f"GB R2={fold_metrics['GB'][0]:.3f}  XGB R2={fold_metrics['XGB'][0]:.3f}  "
              f"Ensemble R2={r2_e:.3f}")

    # --- aggregate out-of-fold predictions -> overall CV metrics ---
    ens_oof = np.mean([oof_pred["RF"], oof_pred["GB"], oof_pred["XGB"]], axis=0)

    r2_rf, rmse_rf, mae_rf = compute_metrics(y, oof_pred["RF"])
    r2_gb, rmse_gb, mae_gb = compute_metrics(y, oof_pred["GB"])
    r2_xgb, rmse_xgb, mae_xgb = compute_metrics(y, oof_pred["XGB"])
    r2_ens, rmse_ens, mae_ens = compute_metrics(y, ens_oof)

    # Q2_CV: cross-validated R^2 computed from OOF PRESS (identical formula
    # to R2 here since predictions are already strictly out-of-fold)
    q2_cv = r2_ens

    table4_rows.append({
        "Property": prop,
        "RF R2": r2_rf, "GB R2": r2_gb, "XGB R2": r2_xgb,
        "Ensemble R2": r2_ens, "Q2_CV": q2_cv,
        "RMSE": rmse_ens, "MAE": mae_ens,
        "RF RMSE": rmse_rf, "GB RMSE": rmse_gb, "XGB RMSE": rmse_xgb,
        "RF MAE": mae_rf, "GB MAE": mae_gb, "XGB MAE": mae_xgb,
    })

    oof_df = pd.DataFrame({
        ID_COL: work_df[ID_COL].values,
        "Observed": y,
        "Pred_RF": oof_pred["RF"], "Pred_GB": oof_pred["GB"], "Pred_XGB": oof_pred["XGB"],
        "Pred_Ensemble": ens_oof,
    })
    oof_predictions_all[prop] = oof_df

    print(f"  >> Overall CV (OOF): Ensemble R2={r2_ens:.4f}  RMSE={rmse_ens:.4f}  MAE={mae_ens:.4f}")

table4 = pd.DataFrame(table4_rows)
print("\n=== Table 4. Five-fold cross-validation performance ===")
print(table4[["Property", "RF R2", "GB R2", "XGB R2", "Ensemble R2", "RMSE", "MAE"]]
      .round(4).to_string(index=False))

cv_fold_detail = pd.DataFrame(cv_fold_detail_rows)

# ------------------------------------------------------------------
# Save Step 9/10 outputs
# ------------------------------------------------------------------
cv_results_path = f"{OUT_DIR}/IMPPAT_Step9_10_CV_Results.xlsx"
with pd.ExcelWriter(cv_results_path, engine="openpyxl") as writer:
    table4_export = table4[["Property", "RF R2", "GB R2", "XGB R2", "Ensemble R2",
                             "Q2_CV", "RMSE", "MAE"]].round(4)
    table4_export.to_excel(writer, sheet_name="Table4_CV_Performance", index=False)

    table4_full = table4.round(4)
    table4_full.to_excel(writer, sheet_name="Table4_Full_AllModels", index=False)

    cv_fold_detail.round(4).to_excel(writer, sheet_name="Per_Fold_Detail", index=False)

    for prop, oof_df in oof_predictions_all.items():
        sheet_name = f"OOF_{prop}"[:31]   # Excel sheet name limit
        oof_df.round(4).to_excel(writer, sheet_name=sheet_name, index=False)

    settings = pd.DataFrame({
        "Parameter": ["n_estimators", "Random seed", "CV folds", "Descriptor panel size",
                      "Working set size"],
        "Value": [N_ESTIMATORS, RANDOM_SEED, N_FOLDS, len(FINAL_PANEL), n_work]
    })
    settings.to_excel(writer, sheet_name="Settings", index=False)

print(f"\nSaved: {cv_results_path}")

# ==================================================================
# STEP 11 — Blind external validation
# ==================================================================
print("\n\n=== STEP 11: Blind external validation ===")

table5_rows = []
blind_predictions_all = {}

for prop in TARGET_PROPERTIES:
    y_work = work_df[prop].values.astype(float)
    y_blind = blind_df[prop].values.astype(float)

    # --- fit final models on the COMPLETE working set only ---
    scaler_final = StandardScaler().fit(X_work_full)
    X_work_s = scaler_final.transform(X_work_full)
    X_blind_s = scaler_final.transform(X_blind_full)   # blind set transformed, never fit on

    models = make_models()
    blind_preds = {}
    for name, model in models.items():
        model.fit(X_work_s, y_work)
        blind_preds[name] = model.predict(X_blind_s)

    ens_blind_pred = np.mean([blind_preds["RF"], blind_preds["GB"], blind_preds["XGB"]], axis=0)

    r2_ext, rmse_ext, mae_ext = compute_metrics(y_blind, ens_blind_pred)

    cv_row = table4[table4["Property"] == prop].iloc[0]
    table5_rows.append({
        "Property": prop,
        "CV R2": cv_row["Ensemble R2"],
        "CV RMSE": cv_row["RMSE"],
        "Blind R2": r2_ext,
        "Blind RMSE": rmse_ext,
        "Blind MAE": mae_ext,
    })

    blind_df_out = pd.DataFrame({
        ID_COL: blind_df[ID_COL].values,
        "Observed": y_blind,
        "Pred_RF": blind_preds["RF"], "Pred_GB": blind_preds["GB"], "Pred_XGB": blind_preds["XGB"],
        "Pred_Ensemble": ens_blind_pred,
    })
    blind_predictions_all[prop] = blind_df_out

    print(f"{prop:25s}  CV R2={cv_row['Ensemble R2']:.4f}  Blind R2={r2_ext:.4f}  "
          f"Blind RMSE={rmse_ext:.4f}  Blind MAE={mae_ext:.4f}")

table5 = pd.DataFrame(table5_rows)
print("\n=== Table 5. External blind-set validation of the ensemble QSPR models ===")
print(table5.round(4).to_string(index=False))

# ------------------------------------------------------------------
# Save Step 11 outputs
# ------------------------------------------------------------------
blind_results_path = f"{OUT_DIR}/IMPPAT_Step11_Blind_Validation.xlsx"
with pd.ExcelWriter(blind_results_path, engine="openpyxl") as writer:
    table5.round(4).to_excel(writer, sheet_name="Table5_Blind_Validation", index=False)
    for prop, bdf in blind_predictions_all.items():
        sheet_name = f"Blind_{prop}"[:31]
        bdf.round(4).to_excel(writer, sheet_name=sheet_name, index=False)

    settings = pd.DataFrame({
        "Parameter": ["n_estimators", "Random seed", "Working set size (train)",
                      "Blind set size (external test)", "Descriptor panel size"],
        "Value": [N_ESTIMATORS, RANDOM_SEED, n_work, n_blind, len(FINAL_PANEL)]
    })
    settings.to_excel(writer, sheet_name="Settings", index=False)

print(f"\nSaved: {blind_results_path}")
print("\n=== DONE (Steps 9-11) ===")

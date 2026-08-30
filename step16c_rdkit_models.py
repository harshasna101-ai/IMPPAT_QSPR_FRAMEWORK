"""
IMPPAT QSPR study — Step 16 (Part C): RDKit-panel ensemble modeling + benchmark
===================================================================================
Same leakage-free workflow as topological model (Steps 9-11):
    RF/GB/XGB (300 estimators) -> 5-fold CV on working set -> blind validation
using the final 5-descriptor VIF-reduced RDKit panel.

Then build Table 7: Topological model vs RDKit model, for all 8 properties.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb

OUT_DIR = "/home/claude/work/outputs"
STEP5_FILE = "/mnt/user-data/uploads/IMPPAT_Step5_CV_Ensemble_Results.xlsx"
STEP11_FILE = "/mnt/user-data/uploads/IMPPAT_Step11_Blind_Validation.xlsx"

with open(f"{OUT_DIR}/rdkit_final_panel.txt") as f:
    FINAL_RDKIT_DESCRIPTORS = f.read().strip().split(",")
print("Final RDKit panel:", FINAL_RDKIT_DESCRIPTORS)

work_desc = pd.read_excel(f"{OUT_DIR}/IMPPAT_RDKit_Descriptors_Working.xlsx")
blind_desc = pd.read_excel(f"{OUT_DIR}/IMPPAT_RDKit_Descriptors_Blind.xlsx")

work_targets = pd.read_excel("/mnt/user-data/uploads/IMPPAT_Working_Set_1202.xlsx")
blind_targets = pd.read_excel("/mnt/user-data/uploads/IMPPAT_Blind_Set_133.xlsx")

PROPERTIES = [
    "Molecular_Weight", "Polar_Area_TPSA", "Complexity_BertzCT", "XLogP_Crippen",
    "Heavy_Atom_Count", "H_Bond_Donor_Count", "H_Bond_Acceptor_Count", "Rotatable_Bond_Count",
]

N_ESTIMATORS = 300
RANDOM_STATE = 42
N_FOLDS = 5

X_work = work_desc[FINAL_RDKIT_DESCRIPTORS].values
X_blind = blind_desc[FINAL_RDKIT_DESCRIPTORS].values
ids_work = work_desc["IMPPAT Phytochemical identifier"].values
ids_blind = blind_desc["IMPPAT Phytochemical identifier"].values

work_targets_idx = work_targets.set_index("IMPPAT Phytochemical identifier")
blind_targets_idx = blind_targets.set_index("IMPPAT Phytochemical identifier")


def fit_predict_ensemble(X_tr, y_tr, X_te, seed):
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1)
    gb = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=seed)
    xg = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1, verbosity=0)
    rf.fit(X_tr, y_tr); gb.fit(X_tr, y_tr); xg.fit(X_tr, y_tr)
    return (rf.predict(X_te) + gb.predict(X_te) + xg.predict(X_te)) / 3.0


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_rows = []
blind_rows = []
oof_store = {}
blind_pred_store = {}

for prop in PROPERTIES:
    y_work = work_targets_idx.loc[ids_work, prop].values
    y_blind = blind_targets_idx.loc[ids_blind, prop].values

    oof_pred = np.zeros(len(y_work))
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_work)):
        pred = fit_predict_ensemble(X_work[tr_idx], y_work[tr_idx], X_work[te_idx], seed=RANDOM_STATE + fold)
        oof_pred[te_idx] = pred

    r2_cv = r2_score(y_work, oof_pred)
    rmse_cv = np.sqrt(np.mean((y_work - oof_pred) ** 2))
    mae_cv = np.mean(np.abs(y_work - oof_pred))
    cv_rows.append({"Property": prop, "RDKit_Ensemble_R2_CV": r2_cv, "RDKit_RMSE_CV": rmse_cv, "RDKit_MAE_CV": mae_cv})
    oof_store[prop] = pd.DataFrame({"IMPPAT Phytochemical identifier": ids_work, "Observed": y_work, "Pred_Ensemble": oof_pred})

    # final model on full working set -> predict blind
    blind_pred = fit_predict_ensemble(X_work, y_work, X_blind, seed=RANDOM_STATE)
    r2_bl = r2_score(y_blind, blind_pred)
    rmse_bl = np.sqrt(np.mean((y_blind - blind_pred) ** 2))
    mae_bl = np.mean(np.abs(y_blind - blind_pred))
    blind_rows.append({"Property": prop, "RDKit_Ensemble_R2_Blind": r2_bl, "RDKit_RMSE_Blind": rmse_bl, "RDKit_MAE_Blind": mae_bl})
    blind_pred_store[prop] = pd.DataFrame({"IMPPAT Phytochemical identifier": ids_blind, "Observed": y_blind, "Pred_Ensemble": blind_pred})

    print(f"{prop}: CV R2={r2_cv:.3f} RMSE={rmse_cv:.3f} | Blind R2={r2_bl:.3f} RMSE={rmse_bl:.3f}")

rdkit_cv = pd.DataFrame(cv_rows)
rdkit_blind = pd.DataFrame(blind_rows)
rdkit_perf = rdkit_cv.merge(rdkit_blind, on="Property")

# ------------------------------------------------------------------
# Load topological performance for comparison
# ------------------------------------------------------------------
topo_cv = pd.read_excel(STEP5_FILE, sheet_name="Table4_CV_Performance")[["Property", "Ensemble R2", "RMSE"]]
topo_cv = topo_cv.rename(columns={"Ensemble R2": "Topological_Ensemble_R2_CV", "RMSE": "Topological_RMSE_CV"})
topo_blind = pd.read_excel(STEP11_FILE, sheet_name="Table5_Blind_Validation")[["Property", "Blind R2", "Blind RMSE"]]
topo_blind = topo_blind.rename(columns={"Blind R2": "Topological_Ensemble_R2_Blind", "Blind RMSE": "Topological_RMSE_Blind"})

table7 = topo_cv.merge(topo_blind, on="Property").merge(rdkit_perf, on="Property")
table7["DeltaR2_CV(Topo-RDKit)"] = table7["Topological_Ensemble_R2_CV"] - table7["RDKit_Ensemble_R2_CV"]
table7["DeltaR2_Blind(Topo-RDKit)"] = table7["Topological_Ensemble_R2_Blind"] - table7["RDKit_Ensemble_R2_Blind"]

col_order = [
    "Property",
    "Topological_Ensemble_R2_CV", "RDKit_Ensemble_R2_CV", "DeltaR2_CV(Topo-RDKit)",
    "Topological_RMSE_CV", "RDKit_RMSE_CV",
    "Topological_Ensemble_R2_Blind", "RDKit_Ensemble_R2_Blind", "DeltaR2_Blind(Topo-RDKit)",
    "Topological_RMSE_Blind", "RDKit_RMSE_Blind",
]
table7 = table7[col_order]

with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_Step16_RDKit_Benchmark.xlsx", engine="openpyxl") as writer:
    table7.to_excel(writer, sheet_name="Table7_Topo_vs_RDKit", index=False)
    for prop in PROPERTIES:
        oof_store[prop].to_excel(writer, sheet_name=f"RDKit_OOF_{prop}"[:31], index=False)
        blind_pred_store[prop].to_excel(writer, sheet_name=f"RDKit_Blind_{prop}"[:31], index=False)
    pd.DataFrame({
        "Parameter": ["Final RDKit panel", "N estimators", "N CV folds", "Random state"],
        "Value": [", ".join(FINAL_RDKIT_DESCRIPTORS), N_ESTIMATORS, N_FOLDS, RANDOM_STATE],
    }).to_excel(writer, sheet_name="Settings", index=False)

print("\nSaved IMPPAT_Step16_RDKit_Benchmark.xlsx")
print("\nTable 7:")
print(table7.round(3).to_string(index=False))

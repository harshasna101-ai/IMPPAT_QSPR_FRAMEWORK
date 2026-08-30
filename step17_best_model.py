"""
IMPPAT QSPR study — Step 17: Best-performing model per property (Table 8)
=============================================================================
For each of the 8 properties, compare RF / GB / XGB / Ensemble on the BLIND
external validation set (not CV alone, per protocol: "Do not select a model
based only on CV R2"). Selection criteria (in priority order):
    1. Highest Blind R2
    2. Lowest Blind RMSE
    3. Lowest Blind MAE
If the three criteria disagree on the top model, this is flagged explicitly
rather than silently resolved.

Inputs:
    IMPPAT_Step5_CV_Ensemble_Results.xlsx   -> Table4_Full_AllModels (CV context)
    IMPPAT_Step11_Blind_Validation.xlsx     -> Blind_<Property> sheets (compound-level
                                                 Pred_RF, Pred_GB, Pred_XGB, Pred_Ensemble)

Output:
    IMPPAT_Step17_Best_Model_Selection.xlsx  (Table 8 + full per-model blind metrics)
"""
import numpy as np
import pandas as pd

STEP5_FILE = "/mnt/user-data/uploads/IMPPAT_Step5_CV_Ensemble_Results.xlsx"
STEP11_FILE = "/mnt/user-data/uploads/IMPPAT_Step11_Blind_Validation.xlsx"
OUT_DIR = "/home/claude/work/outputs"

PROPERTIES = [
    "Molecular_Weight", "Polar_Area_TPSA", "Complexity_BertzCT", "XLogP_Crippen",
    "Heavy_Atom_Count", "H_Bond_Donor_Count", "H_Bond_Acceptor_Count", "Rotatable_Bond_Count",
]
MODELS = ["RF", "GB", "XGB", "Ensemble"]

cv_table = pd.read_excel(STEP5_FILE, sheet_name="Table4_Full_AllModels").set_index("Property")


def r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return 1 - ss_res / ss_tot


def rmse(obs, pred):
    return np.sqrt(np.mean((obs - pred) ** 2))


def mae(obs, pred):
    return np.mean(np.abs(obs - pred))


all_model_rows = []
table8_rows = []

for prop in PROPERTIES:
    df = pd.read_excel(STEP11_FILE, sheet_name=f"Blind_{prop}")
    obs = df["Observed"].values

    metrics = {}
    for m in MODELS:
        pred = df[f"Pred_{m}"].values
        metrics[m] = {"Blind_R2": r2(obs, pred), "Blind_RMSE": rmse(obs, pred), "Blind_MAE": mae(obs, pred)}

    for m in MODELS:
        row = {"Property": prop, "Model": m}
        row.update(metrics[m])
        row["CV_R2"] = cv_table.loc[prop, f"{m} R2"] if m != "Ensemble" else cv_table.loc[prop, "Ensemble R2"]
        all_model_rows.append(row)

    # determine best by each criterion
    best_by_r2 = max(MODELS, key=lambda m: metrics[m]["Blind_R2"])
    best_by_rmse = min(MODELS, key=lambda m: metrics[m]["Blind_RMSE"])
    best_by_mae = min(MODELS, key=lambda m: metrics[m]["Blind_MAE"])

    agreement = (best_by_r2 == best_by_rmse == best_by_mae)
    # priority: R2 first, but flag disagreement
    best_model = best_by_r2

    table8_rows.append({
        "Property": prop,
        "Best_Model(Blind_R2_priority)": best_model,
        "Best_Model_Blind_R2": metrics[best_model]["Blind_R2"],
        "Best_Model_Blind_RMSE": metrics[best_model]["Blind_RMSE"],
        "Best_Model_Blind_MAE": metrics[best_model]["Blind_MAE"],
        "Best_by_R2": best_by_r2,
        "Best_by_RMSE": best_by_rmse,
        "Best_by_MAE": best_by_mae,
        "Criteria_Agree": "Yes" if agreement else "No",
        "RF_Blind_R2": metrics["RF"]["Blind_R2"],
        "GB_Blind_R2": metrics["GB"]["Blind_R2"],
        "XGB_Blind_R2": metrics["XGB"]["Blind_R2"],
        "Ensemble_Blind_R2": metrics["Ensemble"]["Blind_R2"],
    })

table8 = pd.DataFrame(table8_rows)
all_models_df = pd.DataFrame(all_model_rows)

with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_Step17_Best_Model_Selection.xlsx", engine="openpyxl") as writer:
    table8.to_excel(writer, sheet_name="Table8_Best_Model", index=False)
    all_models_df.to_excel(writer, sheet_name="All_Model_Metrics(CV+Blind)", index=False)
    pd.DataFrame({
        "Criterion_priority": ["1. Highest Blind R2", "2. Lowest Blind RMSE", "3. Lowest Blind MAE"],
        "Note": ["Primary selection criterion", "Tie-breaker / cross-check", "Tie-breaker / cross-check"],
    }).to_excel(writer, sheet_name="Selection_Criteria", index=False)

print("Saved IMPPAT_Step17_Best_Model_Selection.xlsx\n")
print("Table 8 — Best-performing model per property (blind validation basis):\n")
print(table8[["Property", "Best_Model(Blind_R2_priority)", "Best_Model_Blind_R2",
              "Best_Model_Blind_RMSE", "Best_Model_Blind_MAE", "Criteria_Agree"]].to_string(index=False))

print("\nFull candidate comparison (Blind R2 by model):\n")
print(table8[["Property", "RF_Blind_R2", "GB_Blind_R2", "XGB_Blind_R2", "Ensemble_Blind_R2"]].to_string(index=False))

disagreements = table8[table8["Criteria_Agree"] == "No"]
if len(disagreements):
    print("\n*** Properties where R2 / RMSE / MAE disagree on the best model: ***")
    print(disagreements[["Property", "Best_by_R2", "Best_by_RMSE", "Best_by_MAE"]].to_string(index=False))
else:
    print("\nAll three criteria (R2, RMSE, MAE) agree on the best model for every property.")

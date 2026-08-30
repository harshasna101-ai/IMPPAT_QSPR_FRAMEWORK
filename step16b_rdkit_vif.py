"""
IMPPAT QSPR study — Step 16 (Part B): VIF-based reduction of the RDKit panel
================================================================================
Same procedure as Step 3-4 (topological panel): zero-variance removal, then
iterative VIF pruning (VIF < 10) on the WORKING SET ONLY.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUT_DIR = "/home/claude/work/outputs"
work_desc = pd.read_excel(f"{OUT_DIR}/IMPPAT_RDKit_Descriptors_Working.xlsx")

candidate_cols = [c for c in work_desc.columns if c != "IMPPAT Phytochemical identifier"]
X = work_desc[candidate_cols].copy()

# zero-variance removal
zero_var = X.columns[X.var() == 0].tolist()
X = X.drop(columns=zero_var)
print(f"Zero-variance descriptors removed: {zero_var if zero_var else 'none'}")
print(f"Descriptors after zero-variance filter: {X.shape[1]}")

# iterative VIF pruning
removal_log = []
remaining = list(X.columns)
step = 0
while True:
    vifs = pd.Series(
        [variance_inflation_factor(X[remaining].values, i) for i in range(len(remaining))],
        index=remaining,
    )
    max_vif = vifs.max()
    if max_vif < 10 or len(remaining) <= 1:
        break
    worst = vifs.idxmax()
    removal_log.append({"Step": step, "Removed": worst, "VIF_at_removal": max_vif})
    remaining.remove(worst)
    step += 1

final_vifs = pd.Series(
    [variance_inflation_factor(X[remaining].values, i) for i in range(len(remaining))],
    index=remaining,
)

print(f"\nFinal RDKit panel ({len(remaining)} descriptors, all VIF < 10):")
print(final_vifs.round(3).to_string())

removal_df = pd.DataFrame(removal_log)
with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_RDKit_VIF_Reduction.xlsx", engine="openpyxl") as writer:
    pd.DataFrame({"Final_Descriptor": remaining, "Final_VIF": final_vifs.values}).to_excel(
        writer, sheet_name="Final_RDKit_Panel", index=False
    )
    removal_df.to_excel(writer, sheet_name="Removal_Log", index=False)
    pd.DataFrame({
        "Stage": ["Candidate pool (excl. all target-equivalent descriptors)",
                  "After zero-variance filter", "After VIF pruning (final panel)"],
        "Descriptors": [len(candidate_cols), X.shape[1], len(remaining)],
    }).to_excel(writer, sheet_name="Reduction_Summary", index=False)

print(f"\nSaved IMPPAT_RDKit_VIF_Reduction.xlsx")

with open(f"{OUT_DIR}/rdkit_final_panel.txt", "w") as f:
    f.write(",".join(remaining))
print("Saved final panel list to rdkit_final_panel.txt")

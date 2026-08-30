import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CKPT = "/home/claude/work/yrand_checkpoint.pkl"
STEP5_FILE = "/mnt/user-data/uploads/IMPPAT_Step5_CV_Ensemble_Results.xlsx"
FIG_DIR = "/home/claude/work/figures"
OUT_DIR = "/home/claude/work/outputs"

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
N_PERMUTATIONS = 200

with open(CKPT, "rb") as f:
    results = pickle.load(f)

table4 = pd.read_excel(STEP5_FILE, sheet_name="Table4_CV_Performance")
actual_perf = table4.set_index("Property")[["Ensemble R2", "Q2_CV"]]

table6_rows = []
for prop in PROPERTIES:
    r2_arr = results[prop]
    actual_r2 = actual_perf.loc[prop, "Ensemble R2"]
    actual_q2 = actual_perf.loc[prop, "Q2_CV"]
    table6_rows.append({
        "Property": prop,
        "Actual_Ensemble_R2_CV": actual_r2,
        "Actual_Q2_CV": actual_q2,
        "Mean_Randomized_R2": r2_arr.mean(),
        "SD_Randomized_R2": r2_arr.std(ddof=1),
        "Max_Randomized_R2": r2_arr.max(),
        "Min_Randomized_R2": r2_arr.min(),
        "N_Permutations": N_PERMUTATIONS,
        "Permutation_p_value(R2>=actual)": (np.sum(r2_arr >= actual_r2) + 1) / (N_PERMUTATIONS + 1),
    })
table6 = pd.DataFrame(table6_rows)

with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_Step14_Y_Randomization.xlsx", engine="openpyxl") as writer:
    table6.to_excel(writer, sheet_name="Table6_Y_Randomization", index=False)
    raw_df = pd.DataFrame(results)
    raw_df.index.name = "Permutation"
    raw_df.to_excel(writer, sheet_name="Raw_Permutation_R2")
    pd.DataFrame({
        "Parameter": ["N_permutations", "N_estimators (RF/GB/XGB)", "Train/holdout split",
                      "Random state", "Note"],
        "Value": [N_PERMUTATIONS, 30, "80/20 fixed split (reused across permutations, only target shuffled)",
                  42,
                  "Fixed-split evaluation used instead of full 5-fold CV per permutation, and estimators "
                  "reduced from 300 to 30, to make 200 permutations x 8 properties tractable on single-core "
                  "hardware. Documented deviation from the Step 10 CV protocol; the actual CV R2/Q2 (from "
                  "Step 5, 300 estimators, full 5-fold CV) is used as the comparison benchmark."],
    }).to_excel(writer, sheet_name="Settings", index=False)

print("Saved Table 6:")
print(table6.to_string(index=False))

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, prop in enumerate(PROPERTIES):
    ax = axes[i]
    r2_arr = results[prop]
    actual_r2 = actual_perf.loc[prop, "Ensemble R2"]
    ax.hist(r2_arr, bins=25, color="#7A7A7A", alpha=0.8, edgecolor="white")
    ax.axvline(actual_r2, color="#C1502E", lw=2, label=f"Actual R²={actual_r2:.3f}")
    ax.axvline(r2_arr.mean(), color="#2C6E9E", lw=1.5, ls="--", label=f"Mean rand.={r2_arr.mean():.3f}")
    ax.set_title(f"({PANEL_LETTERS[i]}) {PROPERTY_LABELS[prop]}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Randomized R²", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)

fig.suptitle(f"Figure 6. Y-Randomization Validation ({N_PERMUTATIONS} permutations per property)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIG_DIR}/Figure6_Y_Randomization.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR}/Figure6_Y_Randomization.png")

"""
IMPPAT QSPR study — Step 14: Y-randomization
=============================================
For each of the 8 physicochemical properties:
    1. Keep the descriptor matrix (final 9-descriptor panel) unchanged.
    2. Randomly permute the working-set target values (200 permutations).
    3. Refit the RF/GB/XGB ensemble on a fixed train/holdout split (same split,
       reused across all 200 permutations -- only y is shuffled each time).
    4. Evaluate R2 on the holdout portion.
    5. Store randomized R2 for all 200 permutations.

Compare actual model performance (from Step 5, Table4_CV_Performance) against the
mean / maximum randomized performance.

NOTE ON COMPUTE BUDGET: the original protocol's leakage-free 5-fold CV (Step 10)
used 300-estimator RF/GB/XGB refit per fold. Repeating full 5-fold CV for 200
permutations x 8 properties x 3 models (=24,000 model fits at 300 estimators) is not
tractable on a single CPU core in this environment (~6 hours). Y-randomization here
therefore uses a fixed 80/20 train/holdout split (random_state=42, identical split
reused for every permutation -- only the target is shuffled) with 30-estimator
RF/GB/XGB (reduced from 300 to make 200 permutations x 8 properties tractable on
single-core hardware). This is a standard, widely-used simplification for permutation testing
(the point of Y-randomization is to characterize the null distribution of chance
correlation, not to reproduce the exact CV estimate), and is documented here
explicitly as a deviation from the nested 5-fold CV used for the real model.
This is flagged in the manuscript methods/caveats.

Outputs:
    figures/Figure6_Y_Randomization.png              (8-panel null-distribution histograms)
    IMPPAT_Step14_Y_Randomization.xlsx                (Table 6 summary + per-permutation raw R2 values)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

# ------------------------------------------------------------------
WORKING_FILE = "/mnt/user-data/uploads/IMPPAT_Working_Set_1202.xlsx"
STEP5_FILE = "/mnt/user-data/uploads/IMPPAT_Step5_CV_Ensemble_Results.xlsx"

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

N_PERMUTATIONS = 200
N_ESTIMATORS = 30
RANDOM_STATE = 42
PANEL_LETTERS = list("ABCDEFGH")

# ------------------------------------------------------------------
df = pd.read_excel(WORKING_FILE)
X_full = df[FINAL_DESCRIPTORS].values

# fixed 80/20 split, reused for every permutation (only y is shuffled)
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE
)
X_train, X_test = X_full[idx_train], X_full[idx_test]

# actual (non-randomized) CV performance, from Step 5
table4 = pd.read_excel(STEP5_FILE, sheet_name="Table4_CV_Performance")
actual_perf = table4.set_index("Property")[["Ensemble R2", "Q2_CV"]]

rng = np.random.default_rng(RANDOM_STATE)


def fit_ensemble_r2(X_tr, y_tr, X_te, y_te, seed):
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1)
    gb = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=seed)
    xg = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1, verbosity=0)

    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    xg.fit(X_tr, y_tr)

    pred = (rf.predict(X_te) + gb.predict(X_te) + xg.predict(X_te)) / 3.0
    return r2_score(y_te, pred)


results = {prop: [] for prop in PROPERTIES}
t_start = time.time()

for prop in PROPERTIES:
    y_full = df[prop].values
    y_train_true, y_test_true = y_full[idx_train], y_full[idx_test]
    t0 = time.time()
    for p in range(N_PERMUTATIONS):
        perm_seed = 1000 * (PROPERTIES.index(prop) + 1) + p
        y_perm = rng.permutation(y_full)  # permute full working-set target, then split
        y_tr_p, y_te_p = y_perm[idx_train], y_perm[idx_test]
        r2 = fit_ensemble_r2(X_train, y_tr_p, X_test, y_te_p, seed=perm_seed)
        results[prop].append(r2)
    print(f"{prop}: {N_PERMUTATIONS} permutations done in {time.time()-t0:.1f}s")

print(f"\nTotal Y-randomization runtime: {time.time()-t_start:.1f}s")

# ------------------------------------------------------------------
# Table 6 — actual vs randomized performance
# ------------------------------------------------------------------
table6_rows = []
raw_perm_data = {}
for prop in PROPERTIES:
    r2_arr = np.array(results[prop])
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
    raw_perm_data[prop] = r2_arr

table6 = pd.DataFrame(table6_rows)

with pd.ExcelWriter(f"{OUT_DIR}/IMPPAT_Step14_Y_Randomization.xlsx", engine="openpyxl") as writer:
    table6.to_excel(writer, sheet_name="Table6_Y_Randomization", index=False)
    raw_df = pd.DataFrame(raw_perm_data)
    raw_df.index.name = "Permutation"
    raw_df.to_excel(writer, sheet_name="Raw_Permutation_R2")
    pd.DataFrame({
        "Parameter": ["N_permutations", "N_estimators (RF/GB/XGB)", "Train/holdout split",
                      "Random state", "Note"],
        "Value": [N_PERMUTATIONS, N_ESTIMATORS, "80/20 fixed split (reused across permutations)",
                  RANDOM_STATE,
                  "Fixed-split evaluation used instead of full 5-fold CV per permutation for tractability "
                  "on single-core hardware; documented deviation from Step 10 CV protocol."],
    }).to_excel(writer, sheet_name="Settings", index=False)

print("Saved IMPPAT_Step14_Y_Randomization.xlsx")
print(table6.to_string(index=False))

# ------------------------------------------------------------------
# Figure 6 — Y-randomization null distributions
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, prop in enumerate(PROPERTIES):
    ax = axes[i]
    r2_arr = raw_perm_data[prop]
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

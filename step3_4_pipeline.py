"""
IMPPAT Descriptor QC Pipeline
=============================
Step 3 — Zero-variance descriptor removal
Step 4 — Iterative VIF-based descriptor reduction

Input : IMPPAT_Working_Set_1202.xlsx   (the working set produced by the
        earlier train/blind split step — Step 3/4 must run on this set
        ONLY, never on the blind set or the full cleaned dataset).
Output: IMPPAT_VIF_Selected_Descriptors.xlsx
        Table2_QC_Summary.xlsx
        Figure1_VIF_reduction_profile.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ------------------------------------------------------------------
# 0. Load the working set
# ------------------------------------------------------------------
INPUT_FILE = "/mnt/user-data/uploads/IMPPAT_Working_Set_1202.xlsx"
OUT_DIR    = "/mnt/user-data/outputs"

df = pd.read_excel(INPUT_FILE)

cols = df.columns.tolist()
start = cols.index("First_Zagreb_M1")
end   = cols.index("Bonchev_Trinajstic_info_index")
topo_cols = cols[start:end + 1]          # the 44 topological indices

assert len(topo_cols) == 44, f"Expected 44 topological indices, found {len(topo_cols)}"

X_full = df[topo_cols].copy()
n_initial = X_full.shape[1]
print(f"Initial descriptor count: {n_initial}")

# ------------------------------------------------------------------
# STEP 3 — Zero-variance descriptor removal
# ------------------------------------------------------------------
variances = X_full.var(axis=0, ddof=1)
zero_var_descriptors = variances[variances == 0].index.tolist()
nonzero_var_descriptors = variances[variances != 0].index.tolist()

n_removed_zerovar = len(zero_var_descriptors)
n_after_zerovar = len(nonzero_var_descriptors)

print(f"Zero-variance descriptors removed: {n_removed_zerovar} -> {zero_var_descriptors}")
print(f"Descriptors remaining after Step 3: {n_after_zerovar}")

X_step3 = X_full[nonzero_var_descriptors].copy()

# ------------------------------------------------------------------
# STEP 4 — Iterative VIF-based descriptor reduction
# ------------------------------------------------------------------
def compute_vif(frame):
    """
    Return a pandas Series of VIF values for every column in `frame`.

    NOTE: the raw topological indices span many orders of magnitude
    (e.g. Wiener_index ~ 1e5-1e6 vs Randic_index ~ 1-10). Feeding the
    raw values into OLS for VIF produces a badly conditioned design
    matrix and numerically wrong (even sub-1, impossible) VIF values.
    VIF is scale-invariant in theory, so we z-score standardize each
    descriptor first purely to stabilise the matrix inversion; the
    resulting VIF values are the mathematically correct ones.
    """
    Xz = (frame - frame.mean()) / frame.std(ddof=1)
    Xc = add_constant(Xz, has_constant="add")
    with np.errstate(divide="ignore", invalid="ignore"):
        vifs = pd.Series(
            [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
            index=frame.columns,
        )
    # perfect collinearity -> R^2 = 1 -> VIF = inf; treat as extreme, finite-ranked value
    return vifs

X_vif = X_step3.copy()

# Record the INITIAL VIF (computed once, before any pruning) for every
# descriptor that survived Step 3.
initial_vif = compute_vif(X_vif)

removal_log = []          # (descriptor, vif_at_removal, order)
order_counter = 0
vif_history = [initial_vif.copy()]   # keep every iteration for plotting/audit

while True:
    current_vif = compute_vif(X_vif)
    max_vif = current_vif.max()
    if max_vif <= 10 or X_vif.shape[1] <= 1:
        final_vif_remaining = current_vif
        break
    worst_descriptor = current_vif.idxmax()
    order_counter += 1
    removal_log.append((worst_descriptor, max_vif, order_counter))
    X_vif = X_vif.drop(columns=[worst_descriptor])
    vif_history.append(compute_vif(X_vif))

retained_descriptors = X_vif.columns.tolist()
n_after_vif = len(retained_descriptors)

print(f"Descriptors removed by VIF pruning: {order_counter}")
print(f"Descriptors remaining after Step 4 (IMPPAT Reduced Descriptor Panel): {n_after_vif}")
print("Retained descriptors:", retained_descriptors)

# ------------------------------------------------------------------
# Build the descriptor-level summary table
# ------------------------------------------------------------------
removed_vif_map  = {d: v for d, v, o in removal_log}
removed_order_map = {d: o for d, v, o in removal_log}

rows = []

# a) zero-variance removed descriptors (removed at Step 3, before VIF calc)
for d in zero_var_descriptors:
    rows.append({
        "Descriptor": d,
        "Initial VIF": np.nan,
        "Final VIF": np.nan,
        "Removal order": 0,                     # 0 = removed at zero-variance stage
        "Retained/Removed": "Removed (zero-variance)"
    })

# b) descriptors removed during iterative VIF pruning
for d in [d for d, v, o in removal_log]:
    rows.append({
        "Descriptor": d,
        "Initial VIF": initial_vif[d],
        "Final VIF": removed_vif_map[d],         # VIF value at the moment it was removed
        "Removal order": removed_order_map[d],
        "Retained/Removed": "Removed (VIF > 10)"
    })

# c) descriptors retained in the final IMPPAT Reduced Descriptor Panel
for d in retained_descriptors:
    rows.append({
        "Descriptor": d,
        "Initial VIF": initial_vif[d],
        "Final VIF": final_vif_remaining[d],
        "Removal order": np.nan,                 # never removed
        "Retained/Removed": "Retained"
    })

summary_df = pd.DataFrame(rows)

# order: retained (by final VIF ascending) then removed (by removal order), zero-var last
order_map = {"Retained": 0, "Removed (VIF > 10)": 1, "Removed (zero-variance)": 2}
summary_df["_sort"] = summary_df["Retained/Removed"].map(order_map)
summary_df = summary_df.sort_values(
    by=["_sort", "Removal order", "Final VIF"],
    ascending=[True, True, False],
    na_position="first"
).drop(columns="_sort").reset_index(drop=True)

# ------------------------------------------------------------------
# Table 2 — QC / reduction summary
# ------------------------------------------------------------------
table2 = pd.DataFrame({
    "Stage": ["Initial descriptors", "Zero-variance removal", "VIF pruning"],
    "Descriptors retained": [n_initial, n_after_zerovar, n_after_vif]
})
print("\nTable 2 — Descriptor QC and reduction summary")
print(table2.to_string(index=False))

# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------
import os
os.makedirs(OUT_DIR, exist_ok=True)

summary_export = summary_df.copy()
# Excel/openpyxl cannot store IEEE inf as a numeric cell value; represent
# perfect collinearity (R^2 = 1) as text "Inf" so the file opens cleanly.
summary_export["Initial VIF"] = summary_export["Initial VIF"].apply(
    lambda v: "Inf" if np.isinf(v) else v)
summary_export["Final VIF"] = summary_export["Final VIF"].apply(
    lambda v: "Inf" if np.isinf(v) else v)

vif_selected_path = f"{OUT_DIR}/IMPPAT_VIF_Selected_Descriptors.xlsx"
with pd.ExcelWriter(vif_selected_path, engine="openpyxl") as writer:
    summary_export.to_excel(writer, sheet_name="VIF_Selected_Descriptors", index=False)
    table2.to_excel(writer, sheet_name="Table2_QC_Summary", index=False)

# also keep the reduced working-set data matrix (IDs + retained descriptors)
id_cols = [c for c in ["IMPPAT Phytochemical identifier", "Chemical name", "SMILES"] if c in df.columns]
reduced_panel_df = pd.concat([df[id_cols], df[retained_descriptors]], axis=1)
reduced_panel_path = f"{OUT_DIR}/IMPPAT_Reduced_Descriptor_Panel_Data.xlsx"
reduced_panel_df.to_excel(reduced_panel_path, index=False)

print(f"\nSaved: {vif_selected_path}")
print(f"Saved: {reduced_panel_path}")

# ------------------------------------------------------------------
# Figure 1 — VIF reduction profile (before vs after pruning)
# ------------------------------------------------------------------
plot_df = summary_df[summary_df["Retained/Removed"] != "Removed (zero-variance)"].copy()

# Cap infinite VIFs (perfect collinearity, R^2 = 1) to a fixed sentinel so
# they remain visible on the log-scale bar plot instead of breaking it.
SENTINEL = 1e7
initial_capped = plot_df["Initial VIF"].replace([np.inf, -np.inf], SENTINEL)
final_capped = plot_df["Final VIF"].fillna(plot_df["Initial VIF"]).replace([np.inf, -np.inf], SENTINEL)
plot_df["_Initial_capped"] = initial_capped
plot_df["_Final_capped"] = final_capped
plot_df = plot_df.sort_values("_Initial_capped", ascending=False)

fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(plot_df))
width = 0.4

bars1 = ax.bar(x - width/2, plot_df["_Initial_capped"], width, label="Initial VIF", color="#4C72B0")
bars2 = ax.bar(x + width/2, plot_df["_Final_capped"], width, label="Final VIF", color="#55A868")

colors = ["#DD8452" if s.startswith("Removed") else "#55A868" for s in plot_df["Retained/Removed"]]
for bar, c in zip(bars2, colors):
    bar.set_color(c)

# mark bars that were capped from infinity
for i, (iv, fv) in enumerate(zip(plot_df["Initial VIF"], plot_df["Final VIF"].fillna(plot_df["Initial VIF"]))):
    if np.isinf(iv):
        ax.text(i - width/2, SENTINEL * 1.15, "∞", ha="center", fontsize=9, color="#4C72B0", fontweight="bold")
    if np.isinf(fv):
        ax.text(i + width/2, SENTINEL * 1.15, "∞", ha="center", fontsize=9, color="#DD8452", fontweight="bold")

ax.axhline(10, color="red", linestyle="--", linewidth=1.5, label="VIF = 10 threshold")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["Descriptor"], rotation=90, fontsize=8)
ax.set_ylabel("Variance Inflation Factor (VIF, log scale)")
ax.set_title("Figure 1. VIF Reduction Profile — Before vs After Iterative Pruning\n"
             "(green = retained in IMPPAT Reduced Descriptor Panel, orange = removed; "
             "∞ = perfect collinearity, capped at 1e7 for display)")
ax.legend()
ax.set_yscale("log")
ax.set_ylim(0.5, SENTINEL * 3)
plt.tight_layout()

fig1_path = f"{OUT_DIR}/Figure1_VIF_reduction_profile.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}")

# ------------------------------------------------------------------
# Bonus figure — Table 2 stage-wise funnel bar chart
# ------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.bar(table2["Stage"], table2["Descriptors retained"], color=["#4C72B0", "#55A868", "#C44E52"])
for i, v in enumerate(table2["Descriptors retained"]):
    ax2.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
ax2.set_ylabel("Number of descriptors retained")
ax2.set_title("Descriptor Quality-Control and Reduction Funnel (Table 2)")
plt.tight_layout()
fig2_path = f"{OUT_DIR}/Figure_TableS2_QC_funnel.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved: {fig2_path}")

print("\n=== DONE ===")
print(f"Final IMPPAT Reduced Descriptor Panel ({n_after_vif} descriptors):")
for d in retained_descriptors:
    print(" -", d)

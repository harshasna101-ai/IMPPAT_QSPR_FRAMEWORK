"""
Step 19 (REVISED) - Structural-complexity / phytochemical prioritization
==========================================================================
Change from v1: "structural distinctiveness" is now measured by
Tanimoto/ECFP4 chemical-space novelty (independent of the topological
QSPR panel), instead of leverage on the topological descriptor panel -
to avoid conflating the prioritization axis with the applicability-domain
(AD) axis, which also uses leverage. Leverage is retained as a reported
(non-scoring) column for context.

Composite Structural-Drug-likeness Priority Score, per AD-reliable compound:
    Priority_Score = 0.5 x (percentile rank of QEDw)
                    + 0.5 x (percentile rank of Tanimoto novelty)

Novelty definition:
    ECFP4 / Morgan fingerprints (radius=2, 2048 bits) generated in RDKit.
    Structural novelty(i) = 1 - mean Tanimoto similarity of compound i to
    its k=5 nearest neighbors (by Tanimoto similarity) within the full
    1,335-compound IMPPAT set (self excluded).

AD-reliability filter is UNCHANGED from v1 (based on Step 15 / Step 18
response-outlier flags - a genuinely separate, target-based reliability
axis, not a novelty axis).

Inputs (all already produced/uploaded):
  - 8_imppat_cleaned_1335.csv               (SMILES, chemical names, panel descriptors)
  - IMPPAT_Step15_Applicability_Domain.xlsx (leverage h, per-property AD classification)
  - IMPPAT_Step18_QEDw_Modelling.xlsx       (QEDw values, QEDw AD classification)  [local]

Output:
  - IMPPAT_Step19_Prioritization.xlsx  (Table 9 + supporting sheets; overwrites v1)
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
warnings.filterwarnings("ignore")

UPLOAD_DIR = "/mnt/user-data/uploads/"
STEP18_DIR = "/home/claude/step18/"
OUT_FILE   = "IMPPAT_Step19_Prioritization.xlsx"
ID_COL = "IMPPAT Phytochemical identifier"

TOP_N = 25
K_NEIGHBORS = 5
FP_RADIUS = 2      # ECFP4
FP_BITS   = 2048

# ---------------------------------------------------------------
# 1. Load base compound info + final topological panel (for reference only)
# ---------------------------------------------------------------
full_df = pd.read_csv(UPLOAD_DIR + "8_imppat_cleaned_1335.csv")
panel_names = pd.read_excel(UPLOAD_DIR + "IMPPAT_Final_Descriptor_Panel.xlsx",
                             sheet_name="Final_Descriptor_Panel")["Descriptor"].tolist()

base = full_df[[ID_COL, "Chemical name", "SMILES"] + panel_names].copy()

# ---------------------------------------------------------------
# 2. ECFP4 / Morgan fingerprints + Tanimoto k-NN novelty
# ---------------------------------------------------------------
def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)

fps = base["SMILES"].apply(get_fp)
valid_mask = fps.notna()
print(f"Fingerprints generated: {valid_mask.sum()}/{len(base)} "
      f"({(~valid_mask).sum()} failed SMILES parsing)")

fp_list = fps[valid_mask].tolist()
valid_ids = base.loc[valid_mask, ID_COL].tolist()
n = len(fp_list)

novelty = np.zeros(n)
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list)
    sims[i] = -1  # exclude self
    top_k = np.sort(sims)[-K_NEIGHBORS:]   # k highest similarities (nearest neighbors)
    novelty[i] = 1.0 - top_k.mean()

novelty_df = pd.DataFrame({
    ID_COL: valid_ids,
    "Mean_Tanimoto_to_5NN": 1.0 - novelty,
    "Structural_Novelty_Tanimoto": novelty,
})

print(f"Novelty score range: {novelty.min():.3f} - {novelty.max():.3f}, "
      f"mean = {novelty.mean():.3f}")

# ---------------------------------------------------------------
# 3. AD reliability (UNCHANGED from v1) + leverage (reported, not scored)
# ---------------------------------------------------------------
AD8_PROPS = ["Molecular_Weight", "Polar_Area_TPSA", "Complexity_BertzCT",
             "XLogP_Crippen", "Heavy_Atom_Count", "H_Bond_Donor_Count",
             "H_Bond_Acceptor_Count", "Rotatable_Bond_Count"]

ad15_sheets = {p: pd.read_excel(UPLOAD_DIR + "IMPPAT_Step15_Applicability_Domain.xlsx",
                                 sheet_name=f"AD_{p}") for p in AD8_PROPS}

lev_df = ad15_sheets[AD8_PROPS[0]][["Compound_ID", "Leverage_h"]].rename(
    columns={"Compound_ID": ID_COL})

outlier_flags = pd.DataFrame({ID_COL: lev_df[ID_COL]})
for p in AD8_PROPS:
    s = ad15_sheets[p].set_index("Compound_ID")["Classification"]
    outlier_flags[f"outlier_{p}"] = outlier_flags[ID_COL].map(
        lambda cid: s.get(cid, "Unknown") in ("Response outlier", "Outside AD (both)")
    )
outlier_cols = [c for c in outlier_flags.columns if c.startswith("outlier_")]
outlier_flags["N_physchem_outlier_flags"] = outlier_flags[outlier_cols].sum(axis=1)

h_star = 3 * (len(panel_names) + 1) / 1202

qedw_all = pd.read_excel(STEP18_DIR + "IMPPAT_Step18_QEDw_Modelling.xlsx",
                          sheet_name="QEDw_All_1335")[[ID_COL, "QEDw"]]
ad_qedw = pd.read_excel(STEP18_DIR + "IMPPAT_Step18_QEDw_Modelling.xlsx",
                         sheet_name="AD_QEDw_Full").rename(columns={"Compound_ID": ID_COL})
ad_qedw_flag = ad_qedw[[ID_COL, "Classification"]].rename(
    columns={"Classification": "QEDw_AD_Classification"})
ad_qedw_flag["QEDw_outlier_flag"] = ad_qedw_flag["QEDw_AD_Classification"].isin(
    ["Response outlier", "Outside AD (both)"])

# ---------------------------------------------------------------
# 4. Merge everything
# ---------------------------------------------------------------
df = (base.merge(novelty_df, on=ID_COL, how="left")
           .merge(lev_df, on=ID_COL, how="left")
           .merge(outlier_flags[[ID_COL, "N_physchem_outlier_flags"]], on=ID_COL, how="left")
           .merge(qedw_all, on=ID_COL, how="left")
           .merge(ad_qedw_flag[[ID_COL, "QEDw_AD_Classification", "QEDw_outlier_flag"]],
                  on=ID_COL, how="left"))

df = df.dropna(subset=["QEDw", "Structural_Novelty_Tanimoto"]).reset_index(drop=True)

df["AD_Reliable"] = (df["N_physchem_outlier_flags"] == 0) & (~df["QEDw_outlier_flag"])

# ---------------------------------------------------------------
# 5. Composite score (Tanimoto novelty replaces leverage)
# ---------------------------------------------------------------
reliable = df[df["AD_Reliable"]].copy()
reliable["QEDw_percentile"] = reliable["QEDw"].rank(pct=True)
reliable["Novelty_percentile"] = reliable["Structural_Novelty_Tanimoto"].rank(pct=True)
reliable["Priority_Score"] = 0.5 * reliable["QEDw_percentile"] + 0.5 * reliable["Novelty_percentile"]
reliable["High_Leverage(h>h*)"] = reliable["Leverage_h"] > h_star

df = df.merge(
    reliable[[ID_COL, "QEDw_percentile", "Novelty_percentile",
              "Priority_Score", "High_Leverage(h>h*)"]],
    on=ID_COL, how="left"
)

ranked = reliable.sort_values("Priority_Score", ascending=False).reset_index(drop=True)
ranked.insert(0, "Priority_Rank", np.arange(1, len(ranked) + 1))

# ---------------------------------------------------------------
# 6. Table 9 - top N prioritized candidates
# ---------------------------------------------------------------
table9_cols = [
    "Priority_Rank", ID_COL, "Chemical name", "SMILES",
    "QEDw", "QEDw_percentile",
    "Structural_Novelty_Tanimoto", "Novelty_percentile",
    "Leverage_h", "High_Leverage(h>h*)",
    "Priority_Score"
]
table9 = ranked[table9_cols].head(TOP_N).copy()
table9 = table9.rename(columns={
    "QEDw_percentile": "QEDw_Percentile_within_Reliable_Set",
    "Structural_Novelty_Tanimoto": "Structural_Novelty(1-Tanimoto_5NN)",
    "Novelty_percentile": "Novelty_Percentile_within_Reliable_Set",
    "Leverage_h": "Leverage_h(AD_context_only,not_scored)",
    "High_Leverage(h>h*)": "High_Leverage_flag(context_only)",
})

print(f"\nTotal compounds with complete data: {len(df)}")
print(f"AD-reliable compounds (eligible for prioritization): {len(reliable)} "
      f"({100*len(reliable)/len(df):.1f}%)")
print(f"\nTop 10 of Table 9 (Tanimoto-novelty version):")
print(table9.head(10).to_string(index=False))

# ---------------------------------------------------------------
# 7. Supporting sheets
# ---------------------------------------------------------------
reliability_summary = pd.DataFrame([{
    "n_total_compounds": len(df),
    "n_AD_reliable": len(reliable),
    "pct_AD_reliable": round(100 * len(reliable) / len(df), 2),
    "n_excluded_physchem_outlier": int((df["N_physchem_outlier_flags"] > 0).sum()),
    "n_excluded_QEDw_outlier": int(df["QEDw_outlier_flag"].sum()),
    "h_star": h_star,
    "fingerprint": f"ECFP4 (Morgan, radius={FP_RADIUS}, {FP_BITS} bits)",
    "k_nearest_neighbors": K_NEIGHBORS,
    "mean_novelty_score": round(df["Structural_Novelty_Tanimoto"].mean(), 3),
}])

methodology_notes = pd.DataFrame({
    "Note": [
        "Structural distinctiveness = Tanimoto novelty, defined as 1 minus the mean Tanimoto "
        "similarity of a compound to its 5 nearest neighbors (by ECFP4/Morgan fingerprint, "
        "radius=2, 2048 bits) within the full 1,335-compound IMPPAT set. This is computed "
        "entirely independently of the 9-descriptor topological QSPR panel used for property "
        "prediction, avoiding conflation with the leverage-based applicability-domain (AD) "
        "diagnostic used in Steps 15/18.",
        "Leverage (h) on the topological panel is retained here as a REPORTED, NON-SCORING "
        "column for context only - e.g. to flag when a Tanimoto-novel compound also sits in a "
        "sparsely populated region of descriptor space - but it does not contribute to the "
        "Priority Score.",
        "AD-reliable = not a response outlier (|standardized residual| > 3) for ANY of the "
        "8 physicochemical property models (Step 15) AND not a response/AD outlier for the "
        "QEDw model (Step 18). This filter is unchanged from the leverage-based v1 of Step 19.",
        "Priority_Score = 0.5 x (percentile rank of QEDw) + 0.5 x (percentile rank of Tanimoto "
        "novelty), computed only within the AD-reliable subset. Equal weighting is a simple, "
        "transparent default; weights can be adjusted depending on whether drug-likeness or "
        "structural novelty should dominate prioritization.",
        "IMPORTANT: these are COMPUTATIONALLY PRIORITIZED CANDIDATES based on predicted "
        "drug-likeness (QEDw) and chemical-space novelty (Tanimoto/ECFP4) - NOT confirmed drug "
        "leads. No bioactivity, target engagement, ADMET, toxicity, or synthetic accessibility "
        "data were used. Experimental validation is required before any candidate is considered "
        "further.",
        "Change log: v1 used leverage on the topological panel as the structural-distinctiveness "
        "axis; v2 (this version) replaces it with Tanimoto/ECFP4 novelty to keep the "
        "prioritization axis independent of the AD diagnostic. Steps 1-18 (VIF, PCA/bootstrap, "
        "CV, blind validation, Y-randomization, AD, QEDw modelling) are unaffected by this change.",
    ]
})

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    table9.to_excel(writer, sheet_name="Table9_Prioritized_Candidates", index=False)
    ranked.to_excel(writer, sheet_name="Full_Ranked_Reliable_Set", index=False)
    df.to_excel(writer, sheet_name="All_1335_Scored", index=False)
    reliability_summary.to_excel(writer, sheet_name="Reliability_Summary", index=False)
    methodology_notes.to_excel(writer, sheet_name="Methodology_and_Caveats", index=False)

print(f"\nSaved: {OUT_FILE}")

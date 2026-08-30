"""
IMPPAT QSPR study — Step 16 (Part A): RDKit descriptor computation
=====================================================================
Compute a predefined, GENERIC RDKit 2D-descriptor panel for all 1,335 IMPPAT
compounds (working + blind), for benchmarking against the compact 9-descriptor
topological-index model.

IMPORTANT: the panel below deliberately contains NO descriptor that is
definitionally identical or directly equivalent to any of the 8 target
properties (MolWt/ExactMolWt, TPSA, BertzCT, MolLogP/Crippen-LogP,
HeavyAtomCount, NumHDonors, NumHAcceptors, NumRotatableBonds are all excluded
from the candidate pool entirely) -- so the SAME final RDKit panel can be used
to predict all 8 properties fairly, exactly mirroring how the topological
9-descriptor panel is used for all 8 properties.

Candidate generic RDKit descriptors (18):
    MolMR, LabuteASA, BalabanJ, Kappa1, Kappa2, Kappa3, Chi0, Chi1,
    HallKierAlpha, RingCount, NumAromaticRings, FractionCSP3,
    NumSaturatedRings, NumAliphaticRings, NHOHCount, NOCount,
    NumHeteroatoms, NumValenceElectrons
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

WORKING_FILE = "/mnt/user-data/uploads/IMPPAT_Working_Set_1202.xlsx"
BLIND_FILE = "/mnt/user-data/uploads/IMPPAT_Blind_Set_133.xlsx"
OUT_DIR = "/home/claude/work/outputs"

CANDIDATE_DESCRIPTORS = {
    "MolMR": Crippen.MolMR,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "Chi0": Descriptors.Chi0,
    "Chi1": Descriptors.Chi1,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles,
    "Ipc": Descriptors.Ipc,
}


def compute_descriptors(df, label):
    rows = []
    failed = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            failed.append(row["IMPPAT Phytochemical identifier"])
            rows.append({name: np.nan for name in CANDIDATE_DESCRIPTORS})
            continue
        vals = {}
        for name, func in CANDIDATE_DESCRIPTORS.items():
            try:
                vals[name] = func(mol)
            except Exception:
                vals[name] = np.nan
        rows.append(vals)
    desc_df = pd.DataFrame(rows)
    desc_df.insert(0, "IMPPAT Phytochemical identifier", df["IMPPAT Phytochemical identifier"].values)
    print(f"{label}: {len(df)} compounds, {len(failed)} SMILES parse failures")
    if failed:
        print("  Failed IDs:", failed)
    return desc_df


work_df = pd.read_excel(WORKING_FILE)
blind_df = pd.read_excel(BLIND_FILE)

work_desc = compute_descriptors(work_df, "Working set")
blind_desc = compute_descriptors(blind_df, "Blind set")

# QC: missing / infinite / constant check on working set
X = work_desc[list(CANDIDATE_DESCRIPTORS.keys())]
n_missing = X.isna().sum().sum()
n_inf = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
const_cols = X.columns[X.nunique() <= 1].tolist()
print(f"\nQC (working set): missing={n_missing}, infinite={n_inf}, constant columns={const_cols}")

work_desc.to_excel(f"{OUT_DIR}/IMPPAT_RDKit_Descriptors_Working.xlsx", index=False)
blind_desc.to_excel(f"{OUT_DIR}/IMPPAT_RDKit_Descriptors_Blind.xlsx", index=False)
print("\nSaved RDKit descriptor tables (working + blind).")
print(work_desc.describe().T[["mean", "std", "min", "max"]])

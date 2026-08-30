"""
Step 1 — Data assembly (IMPPAT cross-library QSPR pipeline)
=============================================================
Merges the two raw IMPPAT input files into a single, validated,
leakage-controlled modeling table and freezes a manifest so every
downstream step (VIF pruning, PCA, ensemble QSPR, Y-randomization,
Williams plot, etc.) starts from an identical, reproducible dataset.

Inputs
------
IMPPAT_1335_Physicochemical_Properties.csv
IMPPAT_1335_Topological_Indices.csv

Outputs (written to OUT_DIR)
-----------------------------
imppat_master.csv       merged table: id, SMILES, 8 targets, 44 descriptors
imppat_manifest.json    seed, column roles, file hashes, shape, checks
imppat_master.parquet   same table, faster to reload downstream

Design choices that matter for later leak-free modeling
---------------------------------------------------------
1. RANDOM_SEED is fixed and written into the manifest so every later
   split (train/blind, CV folds, bootstrap, Y-randomization permutation)
   can be reproduced exactly from this one value.
2. 'Monoisotopic_Mass' is deliberately EXCLUDED from the modeling table.
   It is an almost-deterministic function of Molecular_Weight for a
   fixed elemental composition -> including it as a feature while
   predicting Molecular_Weight (or vice versa) would be a direct
   target leak. It is kept in a side file only, never merged into
   the feature/target matrix.
3. 'n_atoms' and 'n_bonds' are excluded from the 44-descriptor
   topological panel (per the study's definition of the panel) but are
   retained separately since several downstream steps (e.g. AD leverage
   h* = 3(p+1)/n) need compound count, not per-compound size.
4. SMILES columns from both source files are compared row-by-row after
   the join as an integrity check: a mismatch would mean the two files
   are not describing the same compound on that row, which would
   silently corrupt every downstream model.
5. No scaling, imputation, or feature selection happens here. Anything
   that estimates parameters from the data (VIF, PCA loadings, scalers)
   belongs to later steps and must only ever be fit on a training
   partition -- doing it here, before the train/blind split exists,
   would itself be a leakage source.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
IN_DIR = Path("/mnt/user-data/uploads")
OUT_DIR = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHYS_PATH = IN_DIR / "IMPPAT_1335_Physicochemical_Properties.csv"
TOPO_PATH = IN_DIR / "IMPPAT_1335_Topological_Indices.csv"

ID_COL = "IMPPAT Phytochemical identifier"
SMILES_COL = "SMILES"

# The 8 physicochemical properties that serve as QSPR targets.
# Monoisotopic_Mass is intentionally NOT in this list (see module docstring, point 2).
TARGET_COLS = [
    "Molecular_Weight",
    "Polar_Area_TPSA",
    "Complexity_BertzCT",
    "XLogP_Crippen",
    "Heavy_Atom_Count",
    "H_Bond_Donor_Count",
    "H_Bond_Acceptor_Count",
    "Rotatable_Bond_Count",
]
LEAK_RISK_COL = "Monoisotopic_Mass"  # kept aside, never merged into modeling table

# Columns present in the topology file that are NOT part of the 44-index panel.
TOPO_NON_PANEL_COLS = ["n_atoms", "n_bonds"]


def sha256_of(path: Path) -> str:
    """File hash, recorded in the manifest so the exact input version is traceable."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PHYS_PATH.exists():
        raise FileNotFoundError(f"Missing input: {PHYS_PATH}")
    if not TOPO_PATH.exists():
        raise FileNotFoundError(f"Missing input: {TOPO_PATH}")

    phys = pd.read_csv(PHYS_PATH)
    topo = pd.read_csv(TOPO_PATH)
    return phys, topo


def validate_source(df: pd.DataFrame, name: str) -> None:
    if ID_COL not in df.columns:
        raise ValueError(f"{name}: missing identifier column '{ID_COL}'")
    dup = df[ID_COL].duplicated().sum()
    if dup:
        raise ValueError(f"{name}: {dup} duplicated identifiers -- refusing to proceed")
    n_null = df.isnull().sum().sum()
    if n_null:
        raise ValueError(f"{name}: {n_null} null values found -- resolve before merging")


def merge_tables(phys: pd.DataFrame, topo: pd.DataFrame) -> pd.DataFrame:
    validate_source(phys, "Physicochemical file")
    validate_source(topo, "Topological file")

    if set(phys[ID_COL]) != set(topo[ID_COL]):
        only_phys = set(phys[ID_COL]) - set(topo[ID_COL])
        only_topo = set(topo[ID_COL]) - set(phys[ID_COL])
        raise ValueError(
            f"Identifier sets differ. Only in phys: {len(only_phys)}, "
            f"only in topo: {len(only_topo)}"
        )

    merged = phys.merge(
        topo,
        on=ID_COL,
        how="inner",
        suffixes=("_phys", "_topo"),
        validate="one_to_one",  # hard-fails on any accidental fan-out
    )

    # Integrity check: SMILES must agree between the two source files per row.
    mismatches = merged["SMILES_phys"] != merged["SMILES_topo"]
    if mismatches.any():
        bad_ids = merged.loc[mismatches, ID_COL].tolist()
        raise ValueError(
            f"SMILES mismatch between source files for {len(bad_ids)} compounds "
            f"(e.g. {bad_ids[:5]}) -- files are not row-aligned by identifier, "
            f"do not proceed with modeling on this merge."
        )

    merged = merged.rename(columns={"SMILES_phys": SMILES_COL}).drop(columns=["SMILES_topo"])
    return merged


def assemble() -> tuple[pd.DataFrame, dict]:
    phys, topo = load_inputs()
    merged = merge_tables(phys, topo)

    descriptor_cols = [
        c for c in topo.columns if c not in (ID_COL, SMILES_COL, *TOPO_NON_PANEL_COLS)
    ]

    if len(descriptor_cols) != 44:
        raise ValueError(
            f"Expected 44 topological descriptors, found {len(descriptor_cols)}: "
            f"{descriptor_cols}"
        )
    if len(TARGET_COLS) != 8:
        raise ValueError(f"Expected 8 physicochemical targets, found {len(TARGET_COLS)}")

    keep_cols = [ID_COL, "Chemical name", SMILES_COL, *TARGET_COLS, *descriptor_cols]
    master = merged[keep_cols].copy()

    # Extra defensive checks before freezing the dataset
    numeric_cols = TARGET_COLS + descriptor_cols
    non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(master[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found among descriptors/targets: {non_numeric}")

    n_inf = np.isinf(master[numeric_cols].to_numpy(dtype=float)).sum()
    if n_inf:
        raise ValueError(f"{n_inf} infinite values found in descriptors/targets")

    constant_cols = [c for c in numeric_cols if master[c].nunique() <= 1]
    if constant_cols:
        raise ValueError(f"Constant (zero-variance) columns found: {constant_cols}")

    if master[ID_COL].duplicated().sum():
        raise ValueError("Duplicate identifiers survived the merge -- aborting")

    manifest = {
        "random_seed": RANDOM_SEED,
        "n_compounds": int(master.shape[0]),
        "id_col": ID_COL,
        "smiles_col": SMILES_COL,
        "target_cols": TARGET_COLS,
        "n_targets": len(TARGET_COLS),
        "descriptor_cols": descriptor_cols,
        "n_descriptors": len(descriptor_cols),
        "excluded_leak_risk_col": LEAK_RISK_COL,
        "excluded_non_panel_topo_cols": TOPO_NON_PANEL_COLS,
        "source_files": {
            "physicochemical": {
                "path": str(PHYS_PATH),
                "sha256": sha256_of(PHYS_PATH),
                "n_rows": int(phys.shape[0]),
            },
            "topological": {
                "path": str(TOPO_PATH),
                "sha256": sha256_of(TOPO_PATH),
                "n_rows": int(topo.shape[0]),
            },
        },
    }
    return master, manifest


def main() -> None:
    master, manifest = assemble()

    csv_path = OUT_DIR / "imppat_master.csv"
    parquet_path = OUT_DIR / "imppat_master.parquet"
    manifest_path = OUT_DIR / "imppat_manifest.json"

    master.to_csv(csv_path, index=False)
    master.to_parquet(parquet_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Assembled master table: {master.shape[0]} compounds x "
          f"{manifest['n_targets']} targets + {manifest['n_descriptors']} descriptors")
    print(f"Random seed frozen at: {RANDOM_SEED}")
    print(f"Excluded (leak risk): {LEAK_RISK_COL}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()

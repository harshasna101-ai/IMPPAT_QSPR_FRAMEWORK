"""Run Y-randomization for ONE property at a time (called repeatedly to stay under
the per-call execution time limit). Saves results incrementally to a pickle checkpoint."""
import sys, time, pickle, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

WORKING_FILE = "/mnt/user-data/uploads/IMPPAT_Working_Set_1202.xlsx"
CKPT = "/home/claude/work/yrand_checkpoint.pkl"

FINAL_DESCRIPTORS = [
    "Narumi_Katayama_index", "Multiplicative_Zagreb1", "Multiplicative_Zagreb2",
    "Mostar_index", "Szeged_index", "Balaban_J_index",
    "Average_eccentricity", "Sigma_index", "Spectral_radius",
]
N_PERMUTATIONS = 200
N_ESTIMATORS = 30
RANDOM_STATE = 42

prop = sys.argv[1]

df = pd.read_excel(WORKING_FILE)
X_full = df[FINAL_DESCRIPTORS].values
idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE)
X_train, X_test = X_full[idx_train], X_full[idx_test]

y_full = df[prop].values
rng = np.random.default_rng(RANDOM_STATE + hash(prop) % 10000)


def fit_ensemble_r2(X_tr, y_tr, X_te, y_te, seed):
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1)
    gb = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=seed)
    xg = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1, verbosity=0)
    rf.fit(X_tr, y_tr); gb.fit(X_tr, y_tr); xg.fit(X_tr, y_tr)
    pred = (rf.predict(X_te) + gb.predict(X_te) + xg.predict(X_te)) / 3.0
    return r2_score(y_te, pred)


t0 = time.time()
r2_list = []
for p in range(N_PERMUTATIONS):
    y_perm = rng.permutation(y_full)
    y_tr_p, y_te_p = y_perm[idx_train], y_perm[idx_test]
    r2_list.append(fit_ensemble_r2(X_train, y_tr_p, X_test, y_te_p, seed=p))

elapsed = time.time() - t0
print(f"{prop}: {N_PERMUTATIONS} permutations in {elapsed:.1f}s")

# load/update checkpoint
if os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        results = pickle.load(f)
else:
    results = {}
results[prop] = np.array(r2_list)
with open(CKPT, "wb") as f:
    pickle.dump(results, f)

print(f"Checkpoint now has: {list(results.keys())}")

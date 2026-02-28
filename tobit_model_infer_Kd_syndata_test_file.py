#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate genotype–phenotype data with epistatic effects, run Tobit model fitting,
and compare fitted coefficients/predictions to ground truth.

Created on Thu Feb 26 15:17:21 2026
@author: Alexandra
"""

from pathlib import Path
import itertools
import string
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Settings
# ----------------------------
RNG_SEED = 1111
MODEL_ORDER = 4
NUM_GENO_SITES = 10

ANTIGEN = "mock_antigen"
KD_COL = "nlog10_Kd"

CENSOR_L = 8
CENSOR_R = 13
LEFT_CENSOR_VALUE = 1
RIGHT_CENSOR_VALUE = 100

OUTDIR = Path("results")
OUTPREFIX = "mock_file_results"

FIT_SEED = 123
NUM_FOLDS = 3
FOLD_IDX = 0
PENALIZATION = 0.1
REG_TYPE = "l1"


# ----------------------------
# Helpers
# ----------------------------
def term_key(names):
    """Sort a collection of site names and join with underscores."""
    return "_".join(sorted(names))


# ----------------------------
# Simulate genotypes
# ----------------------------
rng = np.random.default_rng(RNG_SEED)

site_names = list(string.ascii_lowercase[:NUM_GENO_SITES])
site_name_idx = dict(enumerate(site_names))

# All binary genotypes (shape: 2**NUM_GENO_SITES x NUM_GENO_SITES)
geno_combinations = np.array(
    list(itertools.product([0, 1], repeat=NUM_GENO_SITES)),
    dtype=bool,
)

# Variant IDs like var_a_b_c (and var_wt for wildtype)
geno_var_id = []
for row in geno_combinations:
    mut_idx = np.flatnonzero(row)
    mut_names = [site_name_idx[i] for i in mut_idx]
    geno_var_id.append("var_" + term_key(mut_names) if mut_names else "var_wt")


# ----------------------------
# Simulate beta values + Kd values
# ----------------------------
intercept = rng.uniform(4, 10)

beta_vals = {}
for k in range(1, MODEL_ORDER + 1):
    # shrink higher-order effects
    scale = 1.0 / (4 ** (k - 1))
    for comb in itertools.combinations(site_names, k):
        beta_vals[term_key(comb)] = rng.normal(loc=0.0, scale=1.0) * scale

kds = []
for row in geno_combinations:
    mut_idx = np.flatnonzero(row)
    mut_names = [site_name_idx[i] for i in mut_idx]
    num_muts = len(mut_names)

    kd = intercept
    for k in range(1, min(num_muts, MODEL_ORDER) + 1):
        for comb in itertools.combinations(mut_names, k):
            kd += beta_vals[term_key(comb)]

    kd += rng.normal(loc=0.0, scale=0.3)
    kds.append(kd)


# ----------------------------
# Quick QC plot of simulated Kd distribution
# ----------------------------
fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(kds, ax=ax)
ax.set_xlabel(r"$K_D$ true")
ax.set_ylabel("Count")
ax.set_title("Simulated Kd values")


# ----------------------------
# Build mock dataset and censor
# ----------------------------
site_cols = [f"site_{s}" for s in site_names]
char_array = np.where(geno_combinations, "M", "G")

mock_data_df = pd.DataFrame(char_array, columns=site_cols)
mock_data_df["variant_id"] = geno_var_id
mock_data_df["antigen"] = ANTIGEN
mock_data_df[KD_COL] = kds

mock_data_df.loc[mock_data_df[KD_COL] < CENSOR_L, KD_COL] = LEFT_CENSOR_VALUE
mock_data_df.loc[mock_data_df[KD_COL] > CENSOR_R, KD_COL] = RIGHT_CENSOR_VALUE

mock_csv = Path("mock_data.csv")
mock_data_df.to_csv(mock_csv, index=False)


# ----------------------------
# Run Tobit model
# ----------------------------
OUTDIR.mkdir(exist_ok=True)

cmd = [
    "python3", "tobit_model_infer_Kd.py",
    "--csv_loc", str(mock_csv),
    "--antigen", ANTIGEN,
    "--antigen_column", "antigen",
    "--order", str(MODEL_ORDER),
    "--num_folds", str(NUM_FOLDS),
    "--fold_idx", str(FOLD_IDX),
    "--penalization", str(PENALIZATION),
    "--reg_type", REG_TYPE,
    "--output_save_loc", str(OUTDIR),
    "--output_file_prefix", OUTPREFIX,
    "--seed", str(FIT_SEED),
    "--left_censor", str(CENSOR_L),
    "--right_censor", str(CENSOR_R),
    "--kd_column_name", KD_COL,
]

# Use subprocess over os.system for safer execution + better error reporting
subprocess.run(cmd, check=True)


# ----------------------------
# Compare fitted betas to true betas
# ----------------------------
beta_path = OUTDIR / f"{OUTPREFIX}_model_results.csv"
model_fit_beta_res = pd.read_csv(beta_path)

# Turn "site_a,site_b" -> "a_b" (and keep intercept as intercept)
model_fit_beta_res["renamed_beta"] = (
    model_fit_beta_res["beta"]
    .apply(lambda x: "intercept" if x == "intercept"
           else term_key(y.replace("site_", "") for y in x.split(",")))
)

model_fit_beta_res["beta_true"] = model_fit_beta_res["renamed_beta"].map(beta_vals)

fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=model_fit_beta_res, x="beta_val", y="beta_true", ax=ax)
ax.axline((0, 0), slope=1, color='grey', linestyle='-')
ax.set_xlabel("Fitted beta")
ax.set_ylabel("True beta")
ax.set_title("Recovered coefficients")


# ----------------------------
# Compare predicted vs true Kd on train/test splits
# ----------------------------
pred_path = OUTDIR / f"{OUTPREFIX}_train_test_results.csv"
data_fit = pd.read_csv(pred_path)

fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(
    data=data_fit,
    x="true_kd",
    y="predicted_kd",
    hue="train_type",
    s=10,
    ax=ax,
)
ax.axline((0, 0), slope=1, color='grey', linestyle='-')
ax.set_xlabel(r"$K_D$ true")
ax.set_ylabel(r"$K_D$ predicted")
ax.set_title("Predicted vs true Kd")

plt.show()
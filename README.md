## Overview and Mathematical background

This package contains Python scripts for fitting a Tobit regression model to a linear model of mutational effects on antibody binding affinity derived from Tite-Seq fluorescence measurements. For background, see *“Binding affinity landscapes constrain the evolution of broadly neutralizing anti-influenza antibodies”* by Phillips et al. ([https://doi.org/10.7554/eLife.71393](https://doi.org/10.7554/eLife.71393)).

In brief, the goal is to infer the effects of individual mutations from a panel of genetically distinct antibody variants and their associated $K_D$ values, obtained by fitting Tite-Seq titration data to a Hill curve. We then model biochemical epistasis using a linear expansion of the form:

$$ -\log_{10} K_D =  \beta_0 + \sum_{i=1}^{16} x_i \beta_i + \sum_{i=1}^{16} \sum_{j>i}^{16} x_i x_j \beta_{ij} * \cdots  $$

Here, $( x_i \in {0,1} )$ indicates the presence or absence of mutation $i$ in a given antibody variant. The coefficients $\beta_i$ quantify the additive effects of individual mutations, while $\beta_{ij}$ (and higher-order terms) capture epistatic interactions affecting binding affinity.

Because Tite-Seq measurements are censored—i.e., affinities below a detection threshold cannot be directly quantified, though variants in this regime are known—we use a Tobit regression framework to properly account for censoring. For left-censored data with threshold $y_L$, define

$$
I(y_j) =
\begin{cases}
0 & \text{if } y_j \le y_L, \
1 & \text{if } y_j > y_L
\end{cases}
$$

Assuming Gaussian residuals with standard deviation $\sigma$, the likelihood for $ N $ observations is

$$
\mathcal{L}(\beta, \sigma)=
\prod_{j=1}^{N}
\left[
\frac{1}{\sigma}
\varphi!\left(\frac{y_j - X_j \beta}{\sigma}\right)
\right]^{I(y_j)}
\left[
1 - \Phi!\left(\frac{X_j \beta - y_L}{\sigma}\right)
\right]^{1 - I(y_j)}
$$

where $\Phi$ and $\varphi$ denote the standard normal cumulative distribution function and probability density function, respectively. This notation is taken from the Wikipedia description of a Tobit model (https://en.wikipedia.org/w/index.php?title=Tobit_model). 

An analogous formulation applies for right-censored data — this package handles both right and left censoring. To fit this maximum likelihood–based model, I use the bounded Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimization algorithm as implemented in `scipy`.


## Running Package Quickstart

To run this package, you'll need an input file with columns of the form:

* `variant_id` (string identifier per variant)
* one or more genotype columns containing `"G"` or `"M"`; the script detects these as any column name containing `"site"` (e.g., `site1`, `site2`, …). Here G indicate the 'germline' variant present and 'M' the 'mutant' variant.
* an antigen label column (specified by `--antigen_column`). Many of the scripts used in the Desai lab contain measurements for multiple antigens concatenated in the same sheet.
* a phenotype column containing (-\log_{10}(K_D)) (specified by `--kd_column_name`)
* optionally, a censoring column (specified by `--censor_column_name`) with values in `{-1,0,1}` 

Here are the command line arguments needed to run this file:

## Command-line options

### Required arguments

* `--csv_loc` *(str)*: Path to input `.csv` or `.tsv`.
* `--antigen` *(str)*: Antigen name to analyze (used to filter rows).
* `--antigen_column` *(str)*: Column containing antigen labels.
* `--kd_column_name` *(str)*: Column containing (-\log_{10}(K_D)).
* `--order` *(int)*: Maximum interaction order (e.g., 1 = additive, 2 = pairwise, 3 = third-order…).
* `--num_folds` *(int)*: Number of folds for CV splitting.
* `--fold_idx` *(int)*: Which fold to train/test on. Use `-1` to train on all data (disables CV split).
* `--penalization` *(float)*: Regularization strength (\lambda) (internally scaled by training set size).
* `--reg_type` *(str)*: Regularization type: `l1` or `l2`.
* `--seed` *(int)*: Random seed (controls shuffling and reproducibility).
* `--output_file_prefix` *(str)*: Prefix for output files.

### Optional arguments

* `--output_save_loc` *(str, default: current directory)*: Output directory.
* `--censor_column_name` *(str, default: None)*: Column encoding censoring (`-1` left, `0` observed, `1` right). If provided, overrides threshold-based censoring.
* `--left_censor` *(float, default: -1)*: Left-censor threshold. Disabled if `-1` and no censor column is provided.
* `--right_censor` *(float, default: -1)*: Right-censor threshold. Disabled if `-1` and no censor column is provided.

**Notes on censoring**

* If thresholds are used, the script sets:

  * `cens = -1` when `phenos <= left_censor`
  * `cens =  1` when `phenos >= right_censor`
  * `cens =  0` otherwise
    and replaces censored phenotype values with the corresponding censor bound.

## Outputs

Two CSV files are written to `output_save_loc`:

### 1) `{prefix}_train_test_results.csv`

Per-variant results and design-matrix features.

Columns include:

* `true_kd`: observed (or censored-to-bound) (-\log_{10}(K_D))
* `predicted_kd`: Tobit model prediction (with censor-aware prediction)
* `geno`: `variant_id` string
* `train_type`: `"train"` or `"test"`
* `true_pred_del`: `true_kd - predicted_kd`
* plus one column per retained feature (e.g., `intercept`, `site3`, `site3,site7`, …)

### 2) `{prefix}_model_results.csv`

Fitted coefficients.

Columns:

* `beta_val`: estimated coefficient value
* `beta`: feature name (comma-separated interaction terms)
* `mat_index`: column index in the retained design matrix

---

## Important behavior / caveats

* **Genotypes must be coded as `"G"` and `"M"`** in the `site*` columns (hard-coded mapping `G→0`, `M→1`).
* The script checks for **rank-deficient feature matrices** (e.g., mutually exclusive mutations) and will error with guidance if interdependent columns are detected.
* Non-informative interaction terms (no variation) are automatically removed before fitting.
* Predictions use `tobit_model_funcs_infer_Kd.cens_predict` and (R^2) uses `tobit_model_funcs_infer_Kd.r2_score`.

---



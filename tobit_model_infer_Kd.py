# my_script.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import sys
import scipy.optimize
import scipy.linalg
import sklearn
import pickle as pkl
import os
import argparse
import sympy

import tobit_model_funcs_infer_Kd as tobit_functions

##########################################################################################
##########################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process TITE-Seq measurements and fit a linear model."
    )
    
    # ------------------------------------------------------------------
    # Required inputs
    # ------------------------------------------------------------------
    required = parser.add_argument_group("Required arguments")
    
    required.add_argument("--csv_loc", type=str, required=True,
                          help="Path to CSV file containing Kd input values.")
    
    required.add_argument("--antigen", type=str, required=True,
                          help="Name of the antigen used.")
    
    required.add_argument("--antigen_column", type=str, required=True,
                          help="Column containing antigen labels.")
    
    required.add_argument("--kd_column_name", type=str, required=True,
                          help="Column containing -log10(Kd) measurements.")
    
    required.add_argument("--order", type=int, required=True,
                          help="Order of the model.")
    
    required.add_argument("--num_folds", type=int, required=True,
                          help="Number of folds for cross-validation.")
    
    required.add_argument("--fold_idx", type=int, required=True,
                          help=("Specific fold to train on. "
                                "Use -1 to train on all data (no CV). "
                                "Overrides 'num_folds'."))
    
    required.add_argument("--penalization", type=float, required=True,
                          help="Regularization strength (lambda).")
    
    required.add_argument("--reg_type", type=str, required=True,
                          help="Regularization method (l1 or l2).")
    
    required.add_argument("--seed", type=int, required=True,
                          help="Random seed for optimization.")
    
    required.add_argument("--output_file_prefix", type=str, required=True,
                          help="Prefix for output file (e.g., {prefix}_coefficients.csv).")
    
    
    # ------------------------------------------------------------------
    # Optional arguments
    # ------------------------------------------------------------------
    optional = parser.add_argument_group("Optional arguments")
    
    optional.add_argument("--output_save_loc", type=str, default=None,
                          help=("Directory to save output file. "
                                "Defaults to current directory."))
    
    optional.add_argument("--censor_column_name", type=str, default=None,
                          help=("Binary column for censoring. "
                                "-1 = left-censored, 0 = valid measurement. 1=right censor "
                                "Overrides 'left_censor'."))
    
    optional.add_argument("--left_censor", type=float, required=False, default=-1,
                          help=("Left-censoring value. "
                                "If -1 this setting is disabled."))

    optional.add_argument("--right_censor", type=float, required=False,default=-1,
                          help=("Right-censoring value. "
                                "If -1 this setting is disabled."))


    return parser.parse_args()    
##########################################################################################
##########################################################################################
  
  
def main():
    
    args = parse_arguments()
    
    #saving information
    csv_loc = args.csv_loc
    example_antigen = args.antigen
    antigen_column = args.antigen_column
    output_file_prefix = args.output_file_prefix
    if args.output_save_loc is None:
        output_save_loc = os.getcwd()
    else:
        output_save_loc = args.output_save_loc

    # file names for use later 
    os.makedirs(output_save_loc, exist_ok=True)

    filename_tt = f'{output_save_loc}/{output_file_prefix}_train_test_results.csv'
    filename_model = f'{output_save_loc}/{output_file_prefix}_model_results.csv'

    if not os.path.isdir(output_save_loc):
        raise InvalidInputError("Output saving location does not exist.")

    # Model params
    order = int(args.order)
    penalization = float(args.penalization)
    num_folds= int(args.num_folds)
    fold_i = int(args.fold_idx)
    reg_type = args.reg_type
    
    # Censoring information
    left_censor = float(args.left_censor)
    right_censor= float(args.right_censor)
    censor_column_name = args.censor_column_name
    kd_column_name = args.kd_column_name
    
    # Seed
    seed = int(args.seed)
        
    np.random.seed(seed)

    print(f"Opening data from file {csv_loc}", flush=True)
    table_extension = "\t" if (".tsv" in csv_loc) else ","
    open_data = pd.read_csv(f"{csv_loc}" , sep=table_extension,dtype = {0:str})
    
    # Get only the data with antigen relevant measuremnts
    open_data = open_data.loc[open_data[antigen_column]==example_antigen]
    
    if open_data.shape[0]==0:
        raise InvalidInputError("Antigen selected is not present within dataset. Please check your spelling and raw data file")

    site_names = [x for x in open_data.columns if 'site' in x]
    
    open_data_shuffled = open_data.sample(frac=1, random_state=seed*2, replace=False).reset_index(drop=True)
    dfc = open_data_shuffled 
        
##########################################################################################
##########################################################################################

    # Get genotypes and censoring elements
    genos_tmp = dfc[site_names].to_numpy()
    
    binarize_mutation_dct = {"G":0,"M":1}
    geno_to_bin_func = np.vectorize(lambda k: binarize_mutation_dct[k])
    genos = geno_to_bin_func(genos_tmp)

    genos_string = dfc.variant_id.to_list()
    
    # Get phenotypes
    phenos = dfc[kd_column_name].to_numpy()

    # Censor based on input method choice
    if censor_column_name:
        cens = dfc[censor_column_name].to_numpy()
    else: 
        cens = np.zeros(dfc.shape[0])
        if left_censor>-1:
            cens[phenos<=left_censor] = -1
        if right_censor>-1:
            cens[phenos>=right_censor] = 1
        
    phenos[cens==-1]= left_censor      
    phenos[cens==1]= right_censor      

                
    if not (np.isin(cens, [-1, 0, 1]).all()):
        raise ValueError("Invalid number in Tobit model. Please check your input csv. This package cannot deal with NAN values.") # Raises a ValueError
    


    num_samples = dfc.shape[0]
    test_div= max(1, num_samples // num_folds)
    
    if fold_i>=0:
        k_split_start = test_div*fold_i
        
        # Chucks any remainder samples into the last test set 
        if fold_i==(test_div-1):
            k_split_stop = num_samples
        else:
            k_split_stop = test_div*(fold_i+1)

        test_x = genos[k_split_start:k_split_stop]
        test_y = phenos[k_split_start:k_split_stop]
        test_cens = cens[k_split_start:k_split_stop]
        geno_string_test  = genos_string[k_split_start:k_split_stop]

        train_tot_x  = np.vstack([genos[:k_split_start,:], genos[k_split_stop:,:]])
        train_tot_y  = np.concatenate([phenos[:k_split_start], phenos[k_split_stop:]])
        cens_train     = np.concatenate([cens[:k_split_start],   cens[k_split_stop:]])
        geno_string_train = genos_string[:k_split_start] + genos_string[k_split_stop:]

        train_size = num_samples - (k_split_stop-k_split_start)
        penalization_scaled = float(penalization) / train_size
        
                            # in format (test geno string, train geno string)

    
    else: # whole model parameters
        k_split_start = 0
        k_split_stop = num_samples
        
        test_x = genos
        test_y = phenos
        test_cens = cens
        geno_string_test = genos_string

        train_tot_x  = test_x
        train_tot_y  = test_y
        cens_train   = test_cens
        geno_string_train = genos_string
        
        penalization_scaled = float(penalization) / num_samples

    print("\n########################### Run details ###########################")
    print(f"Antigen: {example_antigen}", flush=True)
    print(f"Order: {order}", flush=True)
    print(f"Lambda (penalization): {penalization}", flush=True)
    print(f"Censoring cutoff left: {left_censor}", flush=True)
    print(f"Censoring cutoff right: {right_censor}", flush=True)
    
    if censor_column_name:
        print("\t>Censoring applied via boolean selected column.", flush=True)
    else:
        print("\t>Censoring accomplished via raw data.", flush=True)
    print(f"Processing mutation info from sites: {site_names}", flush=True)
    print(f"Number of samples used for model test+train: {open_data.shape[0]}")
    print(f"Number training samples: {len(train_tot_x)}")
    print(f"Number test samples: {len(test_x)}")
    if fold_i>0:
        print(f"Total fold number: {num_folds}. Running {fold_idx} fold.")
    else:
        print(f"Running model on all data.")
    print(f"Number of censored samples in total: {np.nansum(cens!=0)}")
    if fold_i < 0:
        print("Running: whole model (no CV fold)", flush=True)
    else:
        print(f"Running: cross-validation fold {fold_i}", flush=True)    

    print("#####################################################################\n")


    ##########################################################################################
    ##########################################################################################

   
    tr = tobit_functions.TobitModel(fit_intercept=False, 
                                    lower=left_censor, 
                                    upper=right_censor, 
                                    alpha=penalization_scaled, 
                                    penalize_intercept=False,
                                    l1_or_l2=reg_type)

    
    poly = sklearn.preprocessing.PolynomialFeatures(order,
                                                   interaction_only=True,
                                                   include_bias=True)  # ← Explicit
    
    # Check for mutual exclusivity (ex. two mutations K82I and K82D which 
    # by definition cannot both be present in a sequence ) 
    
    geno_check_rank_xform = poly.fit_transform(genos)
    rank = np.linalg.matrix_rank(geno_check_rank_xform)
    if geno_check_rank_xform.shape[1] > rank:
        print("WARNING: interdependent columns observed.")
        # test_mat = np.asarray([[0, 0,1,1,0],
        #                       [0, 0,0,0,1],
        #                       [1, 1,0,0,0],
        #                       [1, 1,1,1,1]])
        M=sympy.Matrix(geno_check_rank_xform)
        ns = M.nullspace()
        
        print(f"There are {len(ns)} problematic beta sites.")
        for i in range(len(ns)):
            ns_arr= np.asarray(ns[i]).flatten()
            sus_columns = [site_names[x] for x in np.argwhere(ns_arr!=0).flatten()]
            print(f"Columns {' '.join(sus_columns)} are completely interdependent.")
        raise Exception("Please consolidate your mutation sites and rerun this script.")

    # Remove redundant features
    poly_feature_names = poly.get_feature_names_out()

    # translate feature names    
    mut_to_featname = {f'x{x}':site_names[x] for x in range(0, len(site_names))}
    mut_to_featname['1'] = 'intercept'
    feat_name = trans_featname = [[mut_to_featname[y] for y in x.split(" ")] for x in poly_feature_names]
    
    # check for non informative sites
    interaction_sum = geno_check_rank_xform.sum(axis=0)

    non_represented_sites = np.argwhere((interaction_sum==0) | (interaction_sum==1)).flatten()
    
    if len(non_represented_sites)>0:
        print("WARNING: non-informative interaction terms observed.")
        sus_beta = [trans_featname[y] for y in non_represented_sites]
        print(f"Columns {' '.join(map(str, sus_beta))} are redundant, ...")
        print(f"Removing those terms.")
        


    num_beta = len(poly_feature_names)
    arr_beta = np.arange(num_beta)
    mask_redun_beta = np.isin(arr_beta, non_represented_sites)
    nonred_beta = arr_beta[~mask_redun_beta].astype(int)

           
    # Transform and scale, removing errant sites
    geno_train_xform = poly.fit_transform(train_tot_x)
    geno_test_xform = poly.transform(test_x)

    poly_train = geno_train_xform[:, nonred_beta]
    poly_test = geno_test_xform[:, nonred_beta]

    phenos_train = train_tot_y.copy()

    # Fit model
    regs = tr.fit(pd.DataFrame(poly_train), 
                  pd.Series(phenos_train), 
                  pd.Series(cens_train), 
                  verbose=False, 
                  alpha=penalization_scaled, 
                  penalize_intercept=False,
                  l1_or_l2=reg_type)

    coeffs= regs.coef_
    coeffs_lin = scipy.linalg.lstsq(poly_train, phenos_train, lapack_driver='gelsy')[0]
    coeffs_names = [feat_name[i] for i in nonred_beta]
    coeffs_names_joined = [",".join(x) for x in coeffs_names]


    test_res= tobit_functions.cens_predict(poly_test, coeffs, left_censor, right_censor) 
    train_res= tobit_functions.cens_predict(poly_train, coeffs, left_censor, right_censor) 

    r2_test = tobit_functions.r2_score(test_y,test_res)
    r2_train = tobit_functions.r2_score(phenos_train,train_res)

    print("\n########################### Model results ###########################")
    print("Finished model fitting.",flush=True)
    print(f"R2 test: {round(r2_test,7)}", flush=True)
    print(f"R2 train: {round(r2_train,7)}", flush=True)
    print("######################################################################\n")


    def build_output_df(phenos, preds, genos, poly, coeff_names, split_name):
        base_df = pd.DataFrame({
            'true_kd': phenos,
            'predicted_kd': preds,
            'geno': genos,
            'train_type': split_name
        })
        
        poly_df = pd.DataFrame(poly, columns=coeff_names)
        
        return pd.concat([base_df, poly_df], axis=1)
    
    
    pd_output_train = build_output_df(
        phenos_train,
        train_res,
        geno_string_train,
        poly_train,
        coeffs_names_joined,
        'train'
    )
    pd_output_test = build_output_df(
        test_y,
        test_res,
        geno_string_test,
        poly_test,
        coeffs_names_joined,
        'test'
    )


    pd_outputs_tt = pd.concat([pd_output_train,pd_output_test])
    pd_outputs_tt['true_pred_del'] = pd_outputs_tt.true_kd-pd_outputs_tt.predicted_kd
    
    	
    pd_outputs_model = pd.DataFrame({
    'beta_val': coeffs,
    'beta': coeffs_names_joined,
    'mat_index': list(range(len(coeffs_names_joined)))
    })


    pd_outputs_tt.to_csv(filename_tt,index=False)
    pd_outputs_model.to_csv(filename_model,index=False)

    

    print(f"Saving output structures to: ", flush=True)
    print(filename_tt, flush=True)
    print(filename_model, flush=True)

    
##########################################################################################
##########################################################################################

if __name__ == "__main__":
    main()

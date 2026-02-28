import numpy as np
import csv
import sys
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import math
import warnings
import pandas as pd
from scipy.optimize import minimize
import scipy.stats
from scipy.special import log_ndtr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from tobit_params import *

def normal_logpdf(z):
    # equals scipy.stats.norm.logpdf(z) but much faster
    return -LOG_SQRT_2PI - 0.5 * z*z

class InvalidInputError(Exception):
    """Raised when an input value is invalid for some parameter."""
    pass


def split_left_right_censored(x, y, cens, lower=None, upper=None):
    counts = cens.value_counts()
    if (-1 not in counts.index) and (1 not in counts.index):
        warnings.warn("No censored observations; use regression methods for uncensored data")
    xs = []
    ys = []

    for value in [-1, 0, 1]:
        if value in counts.index:
            split = cens == value
            x_split = x[split].values
            if value==0:
                y_split = np.squeeze(y[split].values)
            elif value==-1:
                y_split = np.ones(y[split].shape)*lower
            elif value==1:
                y_split = np.ones(y[split].shape)*upper

        else:
            y_split, x_split = None, None
        xs.append(x_split)
        ys.append(y_split)
    return xs, ys


def tobit_neg_log_likelihood_with_l1(xs, ys, params, alpha=0.0, penalize_intercept=False, l1_or_l2=None):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys
    

    b, log_s = params[:-1], params[-1]
    s = np.exp(log_s) # work with logS in order to remove issues of negative log numbers

    nll = tobit_neg_log_likelihood(xs, ys, params) # Apply normal log liklihood
    
    
    if l1_or_l2 == 'l2':
        penalty = alpha * (np.sum(b[1:]**2) if not penalize_intercept else  np.sum(b**2))
    elif l1_or_l2 == 'l1':
        smoothness_ep = 1e-6
        penalty = alpha * (
            np.sum(np.sqrt(b[1:]**2 + smoothness_ep)) if not penalize_intercept
            else np.sum(np.sqrt(b**2 + smoothness_ep))
        )
    elif l1_or_l2==None:
        penalty = 0
    else:
        raise ValueError("Unclear regularization method required.")

    return nll + penalty



def tobit_neg_log_likelihood(xs, ys, params):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys
    
    b, log_s = params[:-1], params[-1]
    s = np.exp(log_s) # work with logS in order to remove issues of negative log numbers

    to_cat = []

    cens_sum = 0.0
    if y_left is not None:
        z_left = (y_left - (x_left @ b)) / s
        cens_sum += scipy.stats.norm.logcdf(z_left).sum()
    if y_right is not None:
        z_right = ((x_right @ b) - y_right) / s
        cens_sum += scipy.stats.norm.logcdf(z_right).sum()

    mid_sum = 0.0
    if y_mid is not None:
        z_mid = (y_mid - np.dot(x_mid, b)) / s # log N(μ,σ^2) = log N(0,1) at z + (-log s)
        mid_sum = (normal_logpdf(z_mid) - np.log(s)).sum()

    loglik = cens_sum + mid_sum

    return - loglik


def tobit_neg_log_likelihood_der(xs, ys, params,alpha = 0.0, l1_or_l2=None, penalize_intercept=False):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b, log_s = params[:-1], params[-1]
    s = np.exp(log_s) # work with logS in order to remove issues of negative log numbers
    # s = math.exp(params[-1]) # in censReg, not using chain rule as below; they optimize in terms of log(s)
    
    # Chatgpt suggestion
    # log_s = params[-1]
    # s = np.exp(log_s)

    beta_jac = np.zeros(len(b))
    sigma_jac = 0

    if y_left is not None:
        left_stats = (y_left - np.dot(x_left, b)) / s
        l_pdf = normal_logpdf(left_stats)
        l_cdf = log_ndtr(left_stats)
        left_frac = np.exp(l_pdf - l_cdf)
        beta_left = np.dot(left_frac, x_left / s)
        beta_jac -= beta_left

        left_sigma = np.dot(left_frac, left_stats)
        sigma_jac -= left_sigma

    if y_right is not None:
        right_stats = (np.dot(x_right, b) - y_right) / s
        r_pdf = scipy.stats.norm.logpdf(right_stats)
        r_cdf = log_ndtr(right_stats)
        right_frac = np.exp(r_pdf - r_cdf)
        beta_right = np.dot(right_frac, x_right / s)
        beta_jac += beta_right

        right_sigma = np.dot(right_frac, right_stats)
        sigma_jac -= right_sigma

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        beta_mid = np.dot(mid_stats, x_mid / s)
        beta_jac += beta_mid

        mid_sigma = (np.square(mid_stats) - 1).sum()
        sigma_jac += mid_sigma
    

    if l1_or_l2 == 'l2':
        if penalize_intercept:
            beta_jac -= 2 * alpha * b
        else:
            beta_jac[1:] -= 2 * alpha * b[1:]
            
    elif l1_or_l2 == 'l1':  # this is a *smoothed* L1
        if penalize_intercept:
            denom = np.sqrt(b * b + smoothness_ep)
            reg_grad = alpha * (b / denom)
            beta_jac -= reg_grad
        else:
            denom = np.sqrt(b[1:] * b[1:] + smoothness_ep)
            reg_grad = alpha * (b[1:] / denom)
            beta_jac[1:] -= reg_grad
            # Note the minus sign is flipped in since this is penalization
          
    grad_log_s = sigma_jac 
    #grad_log_s = sigma_jac * s  OLD VERSION

    combo_jac = np.append(beta_jac, grad_log_s)


    #combo_jac = np.append(beta_jac, sigma_jac / s)  # by chain rule, since the expression above is dloglik/dlogsigma

    return -combo_jac


def cens_predict(x, coef, cens_val_l, cens_val_r):
    y_predd = np.dot(x, coef)
    if cens_val_l>0:
        y_predd[y_predd < cens_val_l] = cens_val_l
    if cens_val_r>0:
        y_predd[y_predd > cens_val_r] = cens_val_r

    return y_predd

def sigmoid(c, Kd, A, B):
    return np.log10(A * (10**c/((10**c)+(10**Kd))) + B)
    
class TobitModel:
    def __init__(self, fit_intercept=True, lower=6.0,upper=14.0, alpha=0.0, penalize_intercept=False, l1_or_l2=None):
        self.ols_coef_ = None
        self.ols_intercept = None
        self.ols_coef_complete = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None
        self.stderr_ = None
        
        self.fit_intercept = fit_intercept
        self.lower_ = lower
        self.upper_ = upper

        self.alpha = alpha
        self.penalize_intercept = penalize_intercept
        self.l1_or_l2 = l1_or_l2
        
        # pd.DataFrame(poly_train[order]), pd.Series(phenos_train), pd.Series(cens_train), 
        #                   verbose=False

    def fit(self, x, y, cens, alpha=0.0, penalize_intercept=False, verbose=False, l1_or_l2=None):
        """
        Fit a maximum-likelihood Tobit regression
        :param x: Pandas DataFrame (n_samples, n_features): Data
        :param y: Pandas Series (n_samples,): Target
        :param cens: Pandas Series (n_samples,): -1 indicates left-censored samples, 0 for uncensored, 1 for right-censored
        :param verbose: boolean, show info from minimization
        :return:
        """
        
        eff_alpha = self.alpha if alpha is None else alpha
        eff_penalize_intercept = self.penalize_intercept if penalize_intercept is None else penalize_intercept
        eff_l1_or_l2 = self.l1_or_l2 if l1_or_l2 is None else l1_or_l2
        # print(f"Parameters -- alpha used: {alpha} self.alpha: {self.alpha}")
        # print(f"Parameters -- alpha used: {l1_or_l2} self.alpha: {self.l1_or_l2}")
        # print(f"Parameters -- penalize_intercept used: {penalize_intercept} self.penalize_intercept: {self.penalize_intercept}")
        # print(f"Parameters -- intercept used: {self.intercept_}")


        x_copy = x.copy()
        if self.fit_intercept:
            x_copy.insert(0, 'intercept', 1.0)
        #else:
        #    x_copy.scale(with_mean=True, with_std=False, copy=False)
        init_reg = LinearRegression(fit_intercept=False).fit(x_copy, y)
        b0 = init_reg.coef_
        y_pred = init_reg.predict(x_copy)
        resid = y - y_pred
        resid_var = np.var(resid)
        #s0 = np.sqrt(resid_var)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, np.log(s0))
        xs, ys = split_left_right_censored(x_copy, y, cens, lower=self.lower_, upper=self.upper_)

        # print("Starting optimization...")
        n=x_copy.shape[1]
        bounds = [(lb_kd,ub_kd)] + [(-1*ub_kd, ub_kd)] * (n - 1) + [(-10, None)]  # last is log_s

        #result = minimize(lambda params: tobit_neg_log_likelihood_with_l1(xs, ys, params, alpha=eff_alpha, penalize_intercept=eff_penalize_intercept, l1_or_l2=eff_l1_or_l2), params0, method='L-BFGS-B',bounds=bounds,jac=lambda params: tobit_neg_log_likelihood_der(xs, ys, params, alpha=eff_alpha, penalize_intercept=eff_penalize_intercept, l1_or_l2=eff_l1_or_l2), options={'disp': verbose})
        result = minimize(
                lambda params: tobit_neg_log_likelihood_with_l1(
                    xs,
                    ys,
                    params,
                    alpha=eff_alpha,
                    penalize_intercept=eff_penalize_intercept,
                    l1_or_l2=eff_l1_or_l2,
                ),
                params0,
                method="L-BFGS-B",
                bounds=bounds,
                jac=lambda params: tobit_neg_log_likelihood_der(
                    xs,
                    ys,
                    params,
                    alpha=eff_alpha,
                    penalize_intercept=eff_penalize_intercept,
                    l1_or_l2=eff_l1_or_l2,
                ),
                options={"disp": verbose},
            )
        if verbose:
            print(result)
        self.ols_coef_ = b0[1:]
        self.ols_intercept = b0[0]
        self.ols_coef_complete = b0
        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_      = result.x[1:-1]

        else:
            self.coef_ = result.x[:-1]
            self.intercept_ = 0
        self.sigma_ = np.exp(result.x[-1])
        
        J = result.hess_inv # From original code
        if hasattr(J, "todense"):           # L-BFGS-B path
            Hinv = np.asarray(J.todense())
        else:                               # BFGS path (already array-like)
            Hinv = np.asarray(J)

        self.stderr_ = np.sqrt(np.diag(Hinv)) # Not really mathematically correct from my understanding with tobit model

        #cov = np.linalg.inv(J.T.dot(J))
        #self.stderr_ = np.sqrt(np.diagonal(J))
        return self


    def predict(self, x):
        y_predd = self.intercept_ + np.dot(x, self.coef_)
        y_predd[y_predd < self.lower_] = self.lower_
        return y_predd
    
    def unfiltered_predict(self, x):
        y_predd = self.intercept_ + np.dot(x, self.coef_)
        return y_predd

    def score(self, x, y, scoring_function=r2_score):
        y_pred = self.intercept_ + np.dot(x, self.coef_)
        y_pred[y_pred < self.lower_] = self.lower_
        return scoring_function(y, y_pred)
    
    def aic(self, x, y):
        y_pred = self.intercept_ + np.dot(x, self.coef_)
        y_pred[y_pred < self.lower_] = self.lower_
        n_sam = np.float64(len(y))
        sse = np.sum((y-y_pred)**2)
        print(sse)
        b0 = self.coef_
        n_par = np.float64(len(b0))
        return (n_sam * np.log(sse/n_sam) + 2*n_par)
        
    def ols_aic(self, x, y):
        
        #Below is a chatgpt suggestion - im not using this and not caring to check it
        x_with_1 = np.c_[np.ones((x.shape[0], 1)), x]
        y_pred = np.dot(x_with_1, self.ols_coef_complete)

        # y_pred = np.array(np.dot(x, self.ols_coef_complete))
        y_pred[y_pred < self.lower_] = self.lower_
        n_sam = np.float64(len(y))
        sse = np.sum((y-y_pred)**2)
        print(sse)
        

        b0 = self.ols_coef_
        n_par = np.float64(len(b0))
        return (n_sam * np.log(sse/n_sam) + 2*n_par)
    
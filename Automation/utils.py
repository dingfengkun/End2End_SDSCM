from collections.abc import Iterable

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def normalize_propensity_scores(data_df, treatment_index, possible_treatment_values):
    logprob_propensity_names = [f'logP(X{treatment_index}={treatment_value})' for treatment_value in possible_treatment_values]
    propensity_names = [f'P(X{treatment_index}={treatment_value})' for treatment_value in possible_treatment_values]
    propensity_df = pd.DataFrame(torch.softmax(torch.tensor(data_df[logprob_propensity_names].values), dim=1), columns=propensity_names)
    
    return propensity_df


def normalize_outcomes(data_df, outcome_index, possible_outcome_values, treatment_index, possible_treatment_values):
    logprob_outcome_names = [f'logP(X{outcome_index}={outcome_value})' for outcome_value in possible_outcome_values]
    prob_outcome_names = [f'P(X{outcome_index}={outcome_value})' for outcome_value in possible_outcome_values]
    outcome_df = pd.DataFrame(torch.softmax(torch.tensor(data_df[logprob_outcome_names].values), dim=1), columns=prob_outcome_names)

    for treatment_value in possible_treatment_values:
        logprob_outcome_names = [f'logP(X{outcome_index}={outcome_value})|do(X{treatment_index}={treatment_value})' for outcome_value in possible_outcome_values]
        prob_outcome_names = [f'P(X{outcome_index}={outcome_value})|do(X{treatment_index}={treatment_value})' for outcome_value in possible_outcome_values]
        outcome_df[prob_outcome_names] = torch.softmax(torch.tensor(data_df[logprob_outcome_names].values), dim=1)
    
    return outcome_df


def add_normalized_colums(data_df, outcome_index, possible_outcome_values, treatment_index, possible_treatment_values):
    propensity_colname_to_check = f'logP(X{treatment_index}={possible_treatment_values[0]})'
    dfs_to_concat = [data_df]
    if propensity_colname_to_check in data_df.columns:
        propensity_df = normalize_propensity_scores(data_df, treatment_index, possible_treatment_values)
        dfs_to_concat.append(propensity_df)
    outcomes_df = normalize_outcomes(data_df, outcome_index, possible_outcome_values, treatment_index, possible_treatment_values)
    dfs_to_concat.append(outcomes_df)
    return pd.concat(dfs_to_concat, axis=1)


# ========================================================================
# Variable-name based normalization functions (compatible with new format)
# ========================================================================

def normalize_propensity_scores_by_varname(data_df, treatment_name):
    """
    Normalize propensity scores using variable names instead of indices.
    
    Args:
        data_df: DataFrame containing logP columns
        treatment_name: Name of the treatment variable (e.g., 't')
    
    Returns:
        DataFrame with normalized propensity scores P(treatment_name=value)
    """
    # Find all treatment-related logP columns
    treatment_logP_cols = [col for col in data_df.columns if col.startswith(f'logP({treatment_name}=')]
    
    if len(treatment_logP_cols) == 0:
        return pd.DataFrame()
    
    # Extract possible values from column names
    treatment_values = []
    for col in treatment_logP_cols:
        # Extract value from 'logP(t=0)' -> '0'
        value = col.split('=')[1].rstrip(')')
        try:
            treatment_values.append(int(value))
        except:
            treatment_values.append(float(value))
    
    treatment_values = sorted(treatment_values)
    
    # Extract logP values and apply softmax
    logP_treatment = data_df[treatment_logP_cols].values
    prob_treatment = torch.softmax(torch.tensor(logP_treatment, dtype=torch.float32), dim=1).numpy()
    
    # Create DataFrame with normalized probabilities
    prob_cols = [f'P({treatment_name}={val})' for val in treatment_values]
    propensity_df = pd.DataFrame(prob_treatment, columns=prob_cols, index=data_df.index)
    
    return propensity_df


def normalize_outcomes_by_varname(data_df, outcome_name, treatment_name=None):
    """
    Normalize outcome probabilities using variable names instead of indices.
    
    Args:
        data_df: DataFrame containing logP columns
        outcome_name: Name of the outcome variable (e.g., 'yf')
        treatment_name: Name of the treatment variable (optional, for counterfactual probabilities)
    
    Returns:
        DataFrame with normalized outcome probabilities P(outcome_name=value)
    """
    # Find all outcome-related logP columns
    outcome_logP_cols = [col for col in data_df.columns if col.startswith(f'logP({outcome_name}=')]
    
    if len(outcome_logP_cols) == 0:
        return pd.DataFrame()
    
    # Extract possible values from column names
    outcome_values = []
    for col in outcome_logP_cols:
        # Extract value from 'logP(yf=0)' -> '0'
        value = col.split('=')[1].rstrip(')')
        try:
            outcome_values.append(int(value))
        except:
            outcome_values.append(float(value))
    
    outcome_values = sorted(outcome_values)
    
    # Extract logP values and apply softmax
    logP_outcome = data_df[outcome_logP_cols].values
    prob_outcome = torch.softmax(torch.tensor(logP_outcome, dtype=torch.float32), dim=1).numpy()
    
    # Create DataFrame with normalized probabilities
    prob_cols = [f'P({outcome_name}={val})' for val in outcome_values]
    outcome_df = pd.DataFrame(prob_outcome, columns=prob_cols, index=data_df.index)
    
    # If treatment_name is provided, also normalize counterfactual outcome probabilities
    if treatment_name is not None:
        # Find treatment values
        treatment_logP_cols = [col for col in data_df.columns if col.startswith(f'logP({treatment_name}=')]
        if len(treatment_logP_cols) > 0:
            treatment_values = []
            for col in treatment_logP_cols:
                value = col.split('=')[1].rstrip(')')
                try:
                    treatment_values.append(int(value))
                except:
                    treatment_values.append(float(value))
            
            treatment_values = sorted(treatment_values)
            
            # Process counterfactual probabilities for each treatment value
            for treatment_value in treatment_values:
                cf_logP_cols = [col for col in data_df.columns 
                               if col.startswith(f'logP({outcome_name}=') and f'|do({treatment_name}={treatment_value})' in col]
                
                if len(cf_logP_cols) > 0:
                    logP_cf = data_df[cf_logP_cols].values
                    prob_cf = torch.softmax(torch.tensor(logP_cf, dtype=torch.float32), dim=1).numpy()
                    
                    prob_cf_cols = [f'P({outcome_name}={val})|do({treatment_name}={treatment_value})' for val in outcome_values]
                    outcome_df[prob_cf_cols] = prob_cf
    
    return outcome_df


def add_normalized_colums_by_varname(data_df, outcome_name, treatment_name, verbose=False):
    """
    Add normalized columns using variable names instead of indices.
    This function is compatible with data that uses variable names in logP columns.
    
    Args:
        data_df: DataFrame containing original data and logP columns
        outcome_name: Name of the outcome variable (e.g., 'yf')
        treatment_name: Name of the treatment variable (e.g., 't')
        verbose: If True, print diagnostic information
    
    Returns:
        DataFrame with original data + normalized probability columns
    """
    dfs_to_concat = [data_df]
    
    # Check if data uses variable names or indices in logP columns
    sample_logP_cols = [col for col in data_df.columns if col.startswith('logP(')]
    if len(sample_logP_cols) > 0 and verbose:
        uses_varnames = not sample_logP_cols[0].startswith('logP(X')
        if uses_varnames:
            print(f"  Detected variable-name based logP columns (e.g., '{sample_logP_cols[0]}')")
        else:
            print(f"  Detected index-based logP columns (e.g., '{sample_logP_cols[0]}')")
    
    # ================================================================
    # Part 1: Normalize treatment propensity scores
    # ================================================================
    treatment_logP_cols = [col for col in data_df.columns if col.startswith(f'logP({treatment_name}=')]
    
    if len(treatment_logP_cols) > 0:
        if verbose:
            print(f"  Found {len(treatment_logP_cols)} treatment logP columns")
        
        propensity_df = normalize_propensity_scores_by_varname(data_df, treatment_name)
        
        if not propensity_df.empty:
            dfs_to_concat.append(propensity_df)
            if verbose:
                print(f"  Added {len(propensity_df.columns)} normalized treatment probability columns")
    elif verbose:
        print(f"  No treatment logP columns found for '{treatment_name}'")
    
    # ================================================================
    # Part 2: Normalize outcome probabilities
    # ================================================================
    outcome_logP_cols = [col for col in data_df.columns if col.startswith(f'logP({outcome_name}=')]
    
    if len(outcome_logP_cols) > 0:
        if verbose:
            print(f"  Found {len(outcome_logP_cols)} outcome logP columns")
        
        outcome_df = normalize_outcomes_by_varname(data_df, outcome_name, treatment_name)
        
        if not outcome_df.empty:
            dfs_to_concat.append(outcome_df)
            if verbose:
                print(f"  Added {len(outcome_df.columns)} normalized outcome probability columns")
    elif verbose:
        print(f"  No outcome logP columns found for '{outcome_name}'")
    
    # ================================================================
    # Part 3: Concatenate all DataFrames
    # ================================================================
    result_df = pd.concat(dfs_to_concat, axis=1)
    
    if verbose:
        new_cols = len(result_df.columns) - len(data_df.columns)
        print(f"  Total columns added: {new_cols}")
    
    return result_df


def verify_normalization(data_df, outcome_name, treatment_name):
    """
    Verify that normalized probabilities sum to 1.
    
    Args:
        data_df: DataFrame with normalized probability columns
        outcome_name: Name of the outcome variable
        treatment_name: Name of the treatment variable
    
    Returns:
        dict with verification results
    """
    results = {}
    
    # Check treatment probabilities
    prob_t_cols = [col for col in data_df.columns if col.startswith(f'P({treatment_name}=')]
    if prob_t_cols:
        prob_t_sum = data_df[prob_t_cols].sum(axis=1)
        results['treatment'] = {
            'columns': prob_t_cols,
            'sum_min': prob_t_sum.min(),
            'sum_max': prob_t_sum.max(),
            'sum_mean': prob_t_sum.mean(),
            'valid': np.allclose(prob_t_sum, 1.0, atol=1e-5)
        }
    
    # Check outcome probabilities
    prob_y_cols = [col for col in data_df.columns if col.startswith(f'P({outcome_name}=') and '|do(' not in col]
    if prob_y_cols:
        prob_y_sum = data_df[prob_y_cols].sum(axis=1)
        results['outcome'] = {
            'columns': prob_y_cols,
            'sum_min': prob_y_sum.min(),
            'sum_max': prob_y_sum.max(),
            'sum_mean': prob_y_sum.mean(),
            'valid': np.allclose(prob_y_sum, 1.0, atol=1e-5)
        }
    
    return results


# ========================================================================
# Original formatting functions (unchanged)
# ========================================================================

def format_estimation_df(predictions_df):
    index_columns = ['df_path', 'model_name', 'dataset_size', 'covariate_names', 'treatment_name', 'y_name']
    estimator_interval_tuples = [('ate', 'ate_lower_bound', 'ate_upper_bound'), ('cate', 'cate_lower_bounds', 'cate_upper_bounds')]
    estimand_for_estimator_lookup = dict(ate='sate', cate='ite')
    estimator_columns = [name for interval_tuple in estimator_interval_tuples for name in interval_tuple]
    estimator_prefixes = [f'{name}_' for name in estimator_columns]
    method_names = predictions_df['method_name'].unique()

    estimation_df = predictions_df.pivot(index=index_columns, columns='method_name', values=estimator_columns)
    estimation_df.columns = estimation_df.columns.map('_'.join)
    estimation_df = estimation_df.reset_index(drop=False)
    true_values_df = predictions_df.groupby(index_columns).take([0])[['y0', 'y1', 'y', 'treatment']].reset_index(level=0)
    true_values_df['ite'] = true_values_df['y1'] - true_values_df['y0']
    true_values_df['sate'] = [np.mean(values) for values in true_values_df['ite']]
    true_values_df['obs_ate'] = [
        y_obs[t_obs == 1].mean() - y_obs[t_obs == 0].mean()
        for y_obs, t_obs in zip(true_values_df['y'], true_values_df['treatment'])
    ]
    estimation_df = estimation_df.merge(true_values_df, on='df_path', how='left')

    for method_name in method_names:
        for prefix in estimator_prefixes:
            column_name = prefix + method_name
            estimation_df[column_name] = [np.squeeze(item) if not isinstance(item, float) else item for item in estimation_df[column_name]]
            
        for (estimator_name, estimator_lower_name, estimator_upper_name) in estimator_interval_tuples:
            point_name = f'{estimator_name}_{method_name}'
            lower_name = f'{estimator_lower_name}_{method_name}'
            upper_name = f'{estimator_upper_name}_{method_name}'
            
            target_estimand = estimand_for_estimator_lookup[estimator_name]
        
            if isinstance(estimation_df[target_estimand][0], Iterable):
                estimation_df[f'RMSE_{estimator_name}_{method_name}'] = [
                    np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)) 
                    for y_true, y_pred in zip(estimation_df[point_name], estimation_df[target_estimand])
                ]
                estimation_df[f'R2_{estimator_name}_{method_name}'] = [
                    r2_score(y_true=y_true, y_pred=y_pred)
                    for y_true, y_pred in zip(estimation_df[point_name], estimation_df[target_estimand])
                ]
            else:
                estimation_df[f'error_{estimator_name}_{method_name}'] = estimation_df[target_estimand] - estimation_df[point_name]
                
            coverage_results = []
            average_widths = []
            for y_true, y_lower, y_upper in zip(estimation_df[target_estimand], estimation_df[lower_name], estimation_df[upper_name]):
                if np.any(y_lower == None) or np.any(y_upper == None):
                    coverage = np.nan
                    width = np.nan
                elif np.any(np.isnan(y_lower)) or np.any(np.isnan(y_upper)):
                    coverage = np.nan
                    width = np.nan
                else:
                    coverage = np.mean((y_true > y_lower) * (y_true < y_upper)) 
                    width = np.mean(np.abs((y_upper - y_lower)))
                coverage_results.append(coverage)
                average_widths.append(width)
            estimation_df[f'coverage_{estimator_name}_{method_name}'] = coverage_results
            estimation_df[f'mean_width_{estimator_name}_{method_name}'] = average_widths
    
    return estimation_df


def join_r_estimation_df(estimation_df, r_predictions_df):
    joined_estimation_df = estimation_df.merge(r_predictions_df, how='inner', on=['df_path', 'covariate_names'])
    nan_columns = [
        'ate_lower_bound_BART', 'ate_upper_bound_BART', 
        'ate_Conformal', 'ate_lower_bound_Conformal', 'ate_upper_bound_Conformal', 
        'cate_Conformal', 'cate_lower_bounds_Conformal', 'cate_upper_bounds_Conformal', 
        'ite_Conformal',
        'ate_BART-ITE', 'ate_lower_bound_BART-ITE', 'ate_upper_bound_BART-ITE',
        'ite_BART-ITE', 'ite_lower_bounds_BART-ITE', 'ite_upper_bounds_BART-ITE',
    ]
    joined_estimation_df[nan_columns] = np.nan
    joined_estimation_df['ate_BART'] = [item[0] for item in joined_estimation_df['ate_BART']]
    joined_estimation_df['ate_BART-ITE'] = 0
    joined_estimation_df['ate_Conformal'] = 0
    joined_estimation_df['cate_lower_bounds_Conformal'] = joined_estimation_df['ite_lower_bounds_Conformal']
    joined_estimation_df['cate_upper_bounds_Conformal'] = joined_estimation_df['ite_upper_bounds_Conformal']
    joined_estimation_df['cate_lower_bounds_BART-ITE'] = joined_estimation_df['ite_lower_bounds_BART']
    joined_estimation_df['cate_upper_bounds_BART-ITE'] = joined_estimation_df['ite_upper_bounds_BART']
    joined_estimation_df['cate_BART-ITE'] = joined_estimation_df['ite_BART']
    estimator_interval_tuples = [('ate', 'ate_lower_bound', 'ate_upper_bound'), ('cate', 'cate_lower_bounds', 'cate_upper_bounds'), ('ite', 'ite_lower_bounds', 'ite_upper_bounds')]
    estimator_columns = [name for interval_tuple in estimator_interval_tuples for name in interval_tuple]
    estimator_prefixes = [f'{name}_' for name in estimator_columns]
    estimand_for_estimator_lookup = dict(ate='sate', cate='ite', ite='ite')
    for method_name in ['BART', 'Conformal', 'BART-ITE']:
        for prefix in estimator_prefixes:
            column_name = prefix + method_name
            joined_estimation_df[column_name] = [np.squeeze(item) for item in joined_estimation_df[column_name]]
            
        for (estimator_name, estimator_lower_name, estimator_upper_name) in estimator_interval_tuples:
            point_name = f'{estimator_name}_{method_name}'
            lower_name = f'{estimator_lower_name}_{method_name}'
            upper_name = f'{estimator_upper_name}_{method_name}'     
            target_estimand = estimand_for_estimator_lookup[estimator_name]

            if method_name == 'Conformal' or (method_name == 'BART-ITE' and estimator_name == 'ite'):  
                joined_estimation_df[f'RMSE_{estimator_name}_{method_name}'] = np.nan
                joined_estimation_df[f'R2_{estimator_name}_{method_name}'] = np.nan
            elif isinstance(joined_estimation_df[target_estimand][0], Iterable):
                joined_estimation_df[f'RMSE_{estimator_name}_{method_name}'] = [
                    np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
                    for y_true, y_pred in zip(joined_estimation_df[point_name], joined_estimation_df[target_estimand])
                ]
                joined_estimation_df[f'R2_{estimator_name}_{method_name}'] = [
                    r2_score(y_true=y_true, y_pred=y_pred)
                    for y_true, y_pred in zip(joined_estimation_df[point_name], joined_estimation_df[target_estimand])
                ]
            else:
                joined_estimation_df[f'error_{estimator_name}_{method_name}'] = joined_estimation_df[target_estimand] - joined_estimation_df[point_name]
                
            coverage_results = []
            average_widths = []
            for y_true, y_lower, y_upper in zip(joined_estimation_df[target_estimand], joined_estimation_df[lower_name], joined_estimation_df[upper_name]):
                if np.any(y_lower == None) or np.any(y_upper == None):
                    coverage = np.nan
                    width = np.nan
                elif np.any(np.isnan(y_lower)) or np.any(np.isnan(y_upper)):
                    coverage = np.nan
                    width = np.nan
                else:
                    coverage = np.mean((y_true > y_lower) * (y_true < y_upper)) 
                    width = np.mean(np.abs((y_upper - y_lower)))
                coverage_results.append(coverage)
                average_widths.append(width)
            joined_estimation_df[f'coverage_{estimator_name}_{method_name}'] = coverage_results
            joined_estimation_df[f'mean_width_{estimator_name}_{method_name}'] = average_widths
    return joined_estimation_df


def format_ate_performance_df(estimation_df):
    model_names = estimation_df['model_name'].unique()
    possible_covariate_names = estimation_df['covariate_names'].unique()
    method_prefix = 'ate_lower_bound_'
    method_column_names = estimation_df.columns[estimation_df.columns.str.startswith(method_prefix)]
    method_names = [name[len(method_prefix):] for name in method_column_names]

    y_true_name = f'sate'
    ate_results_list = []
    for model_name in model_names:
        for covariate_names in possible_covariate_names:
            for method_name in method_names:
                subset_df = estimation_df[(estimation_df['model_name'] == model_name) * (estimation_df['covariate_names'] == covariate_names)]
                
                y_pred_name = f'ate_{method_name}'
                y_pred = subset_df[y_pred_name]
                
                y_true = subset_df[y_true_name]

                r2_value = r2_score(y_true=y_true, y_pred=y_pred)
                rmse_value = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
                
                ate_results_list.append({
                    'method_name': method_name,
                    'model_name': model_name,
                    'covariate_names': covariate_names,
                    'R2': r2_value,
                    'RMSE': rmse_value
                })

    ate_performance_df = pd.DataFrame(ate_results_list).round(4)
    ate_performance_df['R2'][ate_performance_df['R2'] < 0] = '<0'
    ate_performance_df = ate_performance_df.pivot(
        index=['method_name'], 
        columns=['model_name', 'covariate_names']
    )
    ate_performance_df.columns = ate_performance_df.columns.swaplevel(0, 1)
    ate_performance_df = ate_performance_df.sort_index(axis=1, level=0)
    
    return ate_performance_df


def format_cate_performance_df(estimation_df):
    est_df = estimation_df.copy(deep=True)
    est_df['ite'] = est_df['y1'] - est_df['y0']
    est_df['sd_y'] = est_df['y'].apply(np.std)
    est_df['sd_ite'] = est_df['ite'].apply(np.std)

    group_columns = ['df_path', 'model_name', 'covariate_names', 'sd_y', 'sd_ite']
    r2_columns = list(est_df.columns[est_df.columns.str.startswith('R2_cate')])
    rmse_columns = list(est_df.columns[est_df.columns.str.startswith('RMSE_cate')])
    coverage_columns = list(est_df.columns[est_df.columns.str.startswith('coverage_cate')])
    width_columns = list(est_df.columns[est_df.columns.str.startswith('mean_width_cate')])
    selected_columns = group_columns + r2_columns + rmse_columns + coverage_columns + width_columns
    cate_performance_df = est_df[selected_columns]

    cate_performance_df = pd.wide_to_long(
        cate_performance_df,
        stubnames=['R2_cate', 'RMSE_cate', 'coverage_cate', 'mean_width_cate'],
        i=group_columns, 
        j='method_name',
        sep='_',
        suffix=r'.+'
    ).reset_index()
    cate_performance_df['sd_width_cate'] = cate_performance_df['mean_width_cate'] / cate_performance_df['sd_y']
    cate_performance_df['sd_PEHE'] = cate_performance_df['RMSE_cate'] / cate_performance_df['sd_y']
    cate_performance_df['sd_ite_width_cate'] = cate_performance_df['mean_width_cate'] / cate_performance_df['sd_ite']
    cate_performance_df['sd_ite_PEHE'] = cate_performance_df['RMSE_cate'] / cate_performance_df['sd_ite']
    cate_performance_df['num_covariates'] = cate_performance_df['covariate_names'].map(len)
    cate_performance_df['setting'] = [r'All Covariates' if num == 12 else r'Hidden $\mathbf{U}$' for num in cate_performance_df['num_covariates']]
    cate_performance_df['R2'] = cate_performance_df['R2_cate'].clip(lower=0)
    
    return cate_performance_df
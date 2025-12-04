import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
from econml.grf import CausalForest

from econml.dml import LinearDML, CausalForestDML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import TLearner, SLearner
from econml.dr import ForestDRLearner, LinearDRLearner
from catenets.models.jax import SNet, FlexTENet, OffsetNet, TNet, SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet

from .utils import add_normalized_colums, add_normalized_colums_by_varname


def format_prediction_results(
    estimated_cates,
    cate_lower_bounds,
    cate_upper_bounds,
    estimated_ate,
    ate_lower_bound,
    ate_upper_bound,
    estimated_ites,
    ite_lower_bounds,
    ite_upper_bounds
):
    estimate_dict = {
        'ate': estimated_ate,
        'ate_lower_bound': ate_lower_bound,
        'ate_upper_bound': ate_upper_bound,
        'cate': estimated_cates,
        'cate_lower_bounds': cate_lower_bounds,
        'cate_upper_bounds': cate_upper_bounds,
        'ite': estimated_ites,
        'ite_lower_bounds': ite_lower_bounds,
        'ite_upper_bounds': ite_upper_bounds,
    }
    return estimate_dict


def predict_forest_tlearner(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.metalearners.TLearner.html
    model = TLearner(models=RandomForestRegressor())
    model.fit(Y=y, T=z, X=x)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = None, None
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = None, None
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_tlearner(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.metalearners.TLearner.html
    model = TLearner(models=LinearRegression())
    model.fit(Y=y, T=z, X=x)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = None, None
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = None, None
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_forest_slearner(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.metalearners.SLearner.html
    model = SLearner(overall_model=RandomForestRegressor())
    model.fit(Y=y, T=z, X=x)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = None, None
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = None, None
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_slearner(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.metalearners.SLearner.html
    model = SLearner(overall_model=LinearRegression())
    model.fit(Y=y, T=z, X=x)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = None, None
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = None, None
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_forest_dr(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dr.LinearDRLearner.html
    model = ForestDRLearner(discrete_outcome=False, n_jobs=None)
    model.fit(Y=y, T=z, X=x, W=None)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_dr(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dr.LinearDRLearner.html
    model = LinearDRLearner(discrete_outcome=False)
    model.fit(Y=y, T=z, X=x, W=None)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_dml(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dml.LinearDML.html
    model = LinearDML(discrete_treatment=True, discrete_outcome=False)
    model.fit(Y=y, T=z, X=x, W=None)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_causalforest_dml(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html
    model = CausalForestDML(discrete_treatment=True, discrete_outcome=False, n_jobs=None)
    model.fit(Y=y, T=z, X=x, W=None)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_causal_forest(x, z, y):
    model = CausalForest(n_jobs=None)
    model.fit(x, z, y)
    estimated_cates, cate_lower_bounds, cate_upper_bounds = model.predict(X=x, interval=True)
    estimated_ate = estimated_cates.mean()
    ate_lower_bound = None
    ate_upper_bound = None
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_naive_linear_regression(x, z, y):
    x_0 = np.expand_dims(np.zeros_like(z), -1)
    x_1 = np.expand_dims(np.ones_like(z), -1)
    model = LinearRegression().fit(X=np.expand_dims(z, -1), y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = model.coef_[0]
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_regression(x, z, y):
    x_full = np.c_[x, z]
    x_0 = np.c_[x, np.zeros_like(z)]
    x_1 = np.c_[x, np.ones_like(z)]
    model = LinearRegression().fit(X=x_full, y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = model.coef_[-1]
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_naive_random_forest(x, z, y):
    x_0 = np.expand_dims(np.zeros_like(z), -1)
    x_1 = np.expand_dims(np.ones_like(z), -1)
    model = RandomForestRegressor().fit(X=np.expand_dims(z, -1), y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = np.mean(estimated_cates)
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_random_forest(x, z, y):
    x_full = np.c_[x, z]
    x_0 = np.c_[x, np.zeros_like(z)]
    x_1 = np.c_[x, np.ones_like(z)]
    model = RandomForestRegressor().fit(X=x_full, y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = np.mean(estimated_cates)
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_catenet(model_name, x, z, y):
    catenet_lookup = dict(
        SNet=SNet,
        FlexTENet=FlexTENet,
        OffsetNet=OffsetNet,
        TNet=TNet,
        SNet1=SNet1,
        SNet2=SNet2,
        SNet3=SNet3,
        DRNet=DRNet,
        RANet=RANet,
        PWNet=PWNet,
        RNet=RNet,
        XNet=XNet
    )
    model = catenet_lookup[model_name]()
    model.fit(X=x, y=y, w=z)

    estimated_cates = model.predict(X=x)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = np.mean(estimated_cates)
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


METHOD_LOOKUP = dict(
    SNet=lambda x, z, y: predict_catenet('SNet', x, z, y),
    FlexTENet=lambda x, z, y: predict_catenet('FlexTENet', x, z, y),
    OffsetNet=lambda x, z, y: predict_catenet('OffsetNet', x, z, y),
    TNet=lambda x, z, y: predict_catenet('TNet', x, z, y),
    SNet1=lambda x, z, y: predict_catenet('SNet1', x, z, y),
    SNet2=lambda x, z, y: predict_catenet('SNet2', x, z, y),
    SNet3=lambda x, z, y: predict_catenet('SNet3', x, z, y),
    DRNet=lambda x, z, y: predict_catenet('DRNet', x, z, y),
    RANet=lambda x, z, y: predict_catenet('RANet', x, z, y),
    PWNet=lambda x, z, y: predict_catenet('PWNet', x, z, y),
    RNet=lambda x, z, y: predict_catenet('RNet', x, z, y),
    XNet=lambda x, z, y: predict_catenet('XNet', x, z, y),
    CausalForest=predict_causal_forest,
    NaiveLinReg=predict_naive_linear_regression,
    NaiveRF=predict_naive_random_forest,
    LinReg=predict_linear_regression,
    RF=predict_random_forest,
    CausalForestDML=predict_causalforest_dml,
    LinearDML=predict_linear_dml,
    ForestDR=predict_forest_dr,
    LinearDR=predict_linear_dr,
    ForestTLearner=predict_forest_tlearner,
    ForestSLearner=predict_forest_slearner,
    LinearTLearner=predict_linear_tlearner,
    LinearSLearner=predict_linear_slearner,
)


# ========================================================================
# Index-based version (original)
# ========================================================================

def run_method(args):
    """
    Run a single estimation method on a dataset (index-based version).
    
    Args:
        args: Tuple containing (df_path, method_name, covariate_names, treatment_name, 
              y_name, outcome_index, possible_outcome_values, treatment_index, 
              possible_treatment_values)
    
    Returns:
        dict: Prediction results including estimates and ground truth values
    """
    df_path, method_name, covariate_names, treatment_name, y_name, outcome_index, possible_outcome_values, treatment_index, possible_treatment_values = args
    data_df = pd.read_csv(df_path)
    data_df = add_normalized_colums(data_df, outcome_index, possible_outcome_values, treatment_index, possible_treatment_values)

    x = data_df[covariate_names].values
    z = data_df[treatment_name].values
    y = data_df[y_name].values

    prediction_result = METHOD_LOOKUP[method_name](x=x, z=z, y=y)

    y1 = data_df[f'{y_name}|do({treatment_name}=1)'].values
    y0 = data_df[f'{y_name}|do({treatment_name}=0)'].values

    prediction_result.update({
        'df_path': df_path,
        'method_name': method_name,
        'covariate_names': tuple(covariate_names),
        'treatment_name': treatment_name,
        'y_name': y_name,
        'dataset_size': len(y),
        'treatment': z,
        'y0': y0,
        'y1': y1,
        'y': y
    })

    return prediction_result


def run_estimation_methods(
    dataset_paths,
    method_names,
    covariate_names,
    treatment_name,
    y_name,
    outcome_index,
    possible_outcome_values,
    treatment_index,
    possible_treatment_values,
    num_processes=6
):
    """
    Run multiple estimation methods on multiple datasets in parallel (index-based version).
    
    Args:
        dataset_paths: List of paths to dataset CSV files
        method_names: List of method names to run
        covariate_names: List of covariate column names
        treatment_name: Name of treatment variable
        y_name: Name of outcome variable
        outcome_index: Index position of outcome variable
        possible_outcome_values: List of possible outcome values
        treatment_index: Index position of treatment variable
        possible_treatment_values: List of possible treatment values
        num_processes: Number of parallel processes
    
    Returns:
        DataFrame: Results for all method-dataset combinations
    """
    tasks = (
        (
            df_path,
            method_name,
            covariate_names,
            treatment_name,
            y_name,
            outcome_index,
            possible_outcome_values,
            treatment_index,
            possible_treatment_values
        )
        for df_path in dataset_paths
        for method_name in method_names
    )

    with mp.Pool(processes=num_processes) as pool:
        result = [
            item
            for item in tqdm(pool.imap_unordered(run_method, tasks), total=len(dataset_paths)*len(method_names))
        ]
        result_df = pd.DataFrame(result)

    return result_df


# ========================================================================
# Variable-name based version (new)
# ========================================================================

def run_method_by_varname(args):
    """
    Run a single estimation method on a dataset (variable-name based version).
    Compatible with data that uses variable names in logP columns.
    
    Args:
        args: Tuple containing (df_path, method_name, covariate_names, treatment_name, 
              y_name, use_normalization, verbose)
    
    Returns:
        dict: Prediction results including estimates and ground truth values
    """
    df_path, method_name, covariate_names, treatment_name, y_name, use_normalization, verbose = args
    
    data_df = pd.read_csv(df_path)
    
    # Apply normalization if requested
    if use_normalization:
        data_df = add_normalized_colums_by_varname(
            data_df, 
            outcome_name=y_name, 
            treatment_name=treatment_name,
            verbose=verbose
        )
    
    # Extract data for estimation
    x = data_df[covariate_names].values
    z = data_df[treatment_name].values
    y = data_df[y_name].values

    # Run estimation method
    prediction_result = METHOD_LOOKUP[method_name](x=x, z=z, y=y)

    # Extract ground truth counterfactuals (must exist in data)
    y1 = data_df[f'{y_name}|do({treatment_name}=1)'].values
    y0 = data_df[f'{y_name}|do({treatment_name}=0)'].values

    # Add metadata
    prediction_result.update({
        'df_path': df_path,
        'method_name': method_name,
        'covariate_names': tuple(covariate_names),
        'treatment_name': treatment_name,
        'y_name': y_name,
        'dataset_size': len(y),
        'treatment': z,
        'y0': y0,
        'y1': y1,
        'y': y
    })

    return prediction_result


def run_estimation_methods_by_varname(
    dataset_paths,
    method_names,
    covariate_names,
    treatment_name,
    y_name,
    use_normalization=False,
    verbose=False,
    num_processes=6
):
    """
    Run multiple estimation methods on multiple datasets in parallel (variable-name based version).
    Compatible with data that uses variable names instead of indices in logP columns.
    
    Args:
        dataset_paths: List of paths to dataset CSV files
        method_names: List of method names to run
        covariate_names: List of covariate column names
        treatment_name: Name of treatment variable (e.g., 't')
        y_name: Name of outcome variable (e.g., 'yf')
        use_normalization: Whether to apply normalization (default: False, assumes data already normalized)
        verbose: Whether to print diagnostic information
        num_processes: Number of parallel processes
    
    Returns:
        DataFrame: Results for all method-dataset combinations
    
    Example:
        >>> results_df = run_estimation_methods_by_varname(
        ...     dataset_paths=['data1.csv', 'data2.csv'],
        ...     method_names=['CausalForest', 'LinearDML'],
        ...     covariate_names=['x1', 'x2', 'x3'],
        ...     treatment_name='t',
        ...     y_name='yf',
        ...     use_normalization=False,
        ...     num_processes=4
        ... )
    """
    tasks = (
        (
            df_path,
            method_name,
            covariate_names,
            treatment_name,
            y_name,
            use_normalization,
            verbose
        )
        for df_path in dataset_paths
        for method_name in method_names
    )

    with mp.Pool(processes=num_processes) as pool:
        result = [
            item
            for item in tqdm(pool.imap_unordered(run_method_by_varname, tasks), 
                           total=len(dataset_paths)*len(method_names),
                           desc="Running estimation methods")
        ]
        result_df = pd.DataFrame(result)

    return result_df
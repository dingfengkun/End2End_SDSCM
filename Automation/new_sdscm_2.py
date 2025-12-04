import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Automation.new_sdscm import (
    sample_sequences,
    sample_counterfactual_sequences,\
    sample_sequence
)
from typing import List, Dict, Any, Optional
import json
import torch

def generate_counterfactual_datasets(model, tokenizer, config, num_samples=1000, 
                                     evidence_indices=None, verbose=True):
    """
    Generate observational and counterfactual datasets without calculating ATE.
    
    Returns:
        Dictionary containing:
        - samples_obs: Original observational samples
        - samples_cf_x0: Counterfactual samples with do(X=0)
        - samples_cf_x1: Counterfactual samples with do(X=1)
        - df_obs: DataFrame of observational samples
        - df_cf_x0: DataFrame of counterfactuals do(X=0)
        - df_cf_x1: DataFrame of counterfactuals do(X=1)
    """
    
    sequence_sample_space = config['setup_sequence_sample_space']
    index_of_intervention = config['index_of_intervention']
    intervention_name = sequence_sample_space[index_of_intervention]['variable_name']
    
    # Set default evidence indices
    if evidence_indices is None:
        evidence_indices = list(range(index_of_intervention))
        if verbose:
            print(f"Using evidence indices: {evidence_indices}")
            evidence_vars = [sequence_sample_space[i]['variable_name'] for i in evidence_indices]
            print(f"Evidence variables: {evidence_vars}")
    
    # Step 1: Generate observational samples
    if verbose:
        print(f"\n Generating {num_samples} observational samples...")
    
    samples_obs = sample_sequences(
        model=model,
        tokenizer=tokenizer,
        sequence_sample_space=sequence_sample_space,
        num_samples=num_samples,
        verbose=verbose
    )
    
    # Step 2: Generate counterfactuals do(X=0)
    if verbose:
        print(f"\n Generating counterfactuals: do({intervention_name}=0)...")
    
    samples_cf_x0 = sample_counterfactual_sequences(
        model=model,
        tokenizer=tokenizer,
        sequence_sample_space=sequence_sample_space,
        samples=samples_obs,
        evidence_indices=evidence_indices,
        index_of_intervention=index_of_intervention,
        intervention_choice=0,
        verbose=verbose
    )
    
    # Step 3: Generate counterfactuals do(X=1)
    if verbose:
        print(f"\n Generating counterfactuals: do({intervention_name}=1)...")
    
    samples_cf_x1 = sample_counterfactual_sequences(
        model=model,
        tokenizer=tokenizer,
        sequence_sample_space=sequence_sample_space,
        samples=samples_obs,
        evidence_indices=evidence_indices,
        index_of_intervention=index_of_intervention,
        intervention_choice=1,
        verbose=verbose
    )
    
    # Create DataFrames
    variable_names = [setup['variable_name'] for setup in sequence_sample_space]
    
    df_obs = pd.DataFrame([s['sampled_indices'] for s in samples_obs], columns=variable_names)
    df_cf_x0 = pd.DataFrame([s['sampled_indices'] for s in samples_cf_x0], columns=variable_names)
    df_cf_x1 = pd.DataFrame([s['sampled_indices'] for s in samples_cf_x1], columns=variable_names)
    
    # Add sample IDs for matching
    df_obs['sample_id'] = range(num_samples)
    df_cf_x0['sample_id'] = range(num_samples)
    df_cf_x1['sample_id'] = range(num_samples)
    
    # Add data type labels
    df_obs['data_type'] = 'Observational'
    df_cf_x0['data_type'] = f'CF: do({intervention_name}=0)'
    df_cf_x1['data_type'] = f'CF: do({intervention_name}=1)'
    
    return {
        'samples_obs': samples_obs,
        'samples_cf_x0': samples_cf_x0,
        'samples_cf_x1': samples_cf_x1,
        'df_obs': df_obs,
        'df_cf_x0': df_cf_x0,
        'df_cf_x1': df_cf_x1,
        'variable_names': variable_names,
        'intervention_name': intervention_name,
        'evidence_indices': evidence_indices,
        'config': config
    }


def visualize_distributions_comparison(results, figsize=(20, 12)):
    """
    Visualize distributions of all variables comparing observational vs counterfactual data.
    Each variable gets its own subplot showing all three distributions.
    """
    df_obs = results['df_obs']
    df_cf_x0 = results['df_cf_x0']
    df_cf_x1 = results['df_cf_x1']
    variable_names = results['variable_names']
    intervention_name = results['intervention_name']
    
    # Determine number of subplots
    n_vars = len(variable_names)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    fig.suptitle(f'Variable Distributions: Observational vs Counterfactual\nIntervention: {intervention_name}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    colors = {
        'Observational': '#2E86AB',
        f'CF: do({intervention_name}=0)': '#A23B72',
        f'CF: do({intervention_name}=1)': '#F18F01'
    }
    
    for idx, var_name in enumerate(variable_names):
        ax = axes[idx]
        
        # Get unique values and their labels
        all_values = pd.concat([df_obs[var_name], df_cf_x0[var_name], df_cf_x1[var_name]]).unique()
        all_values = sorted(all_values)
        
        # Calculate proportions for each dataset
        obs_counts = df_obs[var_name].value_counts(normalize=True).sort_index()
        cf0_counts = df_cf_x0[var_name].value_counts(normalize=True).sort_index()
        cf1_counts = df_cf_x1[var_name].value_counts(normalize=True).sort_index()
        
        # Prepare data for grouped bar chart
        x = np.arange(len(all_values))
        width = 0.25
        
        # Plot bars
        obs_vals = [obs_counts.get(v, 0) for v in all_values]
        cf0_vals = [cf0_counts.get(v, 0) for v in all_values]
        cf1_vals = [cf1_counts.get(v, 0) for v in all_values]
        
        ax.bar(x - width, obs_vals, width, label='Observational', 
               color=colors['Observational'], alpha=0.8)
        ax.bar(x, cf0_vals, width, label=f'do({intervention_name}=0)', 
               color=colors[f'CF: do({intervention_name}=0)'], alpha=0.8)
        ax.bar(x + width, cf1_vals, width, label=f'do({intervention_name}=1)', 
               color=colors[f'CF: do({intervention_name}=1)'], alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_values, rotation=45 if len(all_values) > 3 else 0)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight if this is the intervention variable
        if var_name == intervention_name:
            ax.patch.set_facecolor('#FFFACD')
            ax.patch.set_alpha(0.3)
            ax.set_title(f'{var_name} (INTERVENTION)', fontsize=12, fontweight='bold', color='red')
    
    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def visualize_individual_changes(results, sample_ids=None, n_samples=5):
    """
    Visualize how individual samples change across counterfactuals.
    Shows the actual values for each variable in a paired comparison.
    """
    df_obs = results['df_obs']
    df_cf_x0 = results['df_cf_x0']
    df_cf_x1 = results['df_cf_x1']
    variable_names = results['variable_names']
    intervention_name = results['intervention_name']
    config = results['config']
    
    # Select samples to visualize
    if sample_ids is None:
        sample_ids = np.random.choice(len(df_obs), min(n_samples, len(df_obs)), replace=False)
    
    fig, axes = plt.subplots(len(sample_ids), len(variable_names), 
                             figsize=(len(variable_names) * 2, len(sample_ids) * 2))
    
    if len(sample_ids) == 1:
        axes = axes.reshape(1, -1)
    if len(variable_names) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Individual Sample Changes: Observational → Counterfactual', 
                 fontsize=14, fontweight='bold')
    
    for i, sample_id in enumerate(sample_ids):
        for j, var_name in enumerate(variable_names):
            ax = axes[i, j]
            
            # Get values
            obs_val = df_obs.iloc[sample_id][var_name]
            cf0_val = df_cf_x0.iloc[sample_id][var_name]
            cf1_val = df_cf_x1.iloc[sample_id][var_name]
            
            # Plot
            positions = [0, 1, 2]
            values = [obs_val, cf0_val, cf1_val]
            colors_list = ['#2E86AB', '#A23B72', '#F18F01']
            labels = ['Obs', f'do(0)', f'do(1)']
            
            bars = ax.bar(positions, values, color=colors_list, alpha=0.7)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val)}', ha='center', va='bottom', fontsize=9)
            
            # Check if value changed
            if var_name == intervention_name:
                ax.patch.set_facecolor('#FFE4E1')
            elif cf0_val != obs_val or cf1_val != obs_val:
                ax.patch.set_facecolor('#E6F3FF')
            
            # Labels
            if i == 0:
                ax.set_title(var_name, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Sample {sample_id}', fontsize=10)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def create_transition_matrix(results, var_name):
    """
    Create transition matrices showing how values change from observational to counterfactual.
    """
    df_obs = results['df_obs']
    df_cf_x0 = results['df_cf_x0']
    df_cf_x1 = results['df_cf_x1']
    
    # Get unique values
    all_values = sorted(pd.concat([df_obs[var_name], df_cf_x0[var_name], df_cf_x1[var_name]]).unique())
    n_values = len(all_values)
    
    # Create transition matrices
    trans_matrix_0 = np.zeros((n_values, n_values))
    trans_matrix_1 = np.zeros((n_values, n_values))
    
    for i in range(len(df_obs)):
        obs_val = df_obs.iloc[i][var_name]
        cf0_val = df_cf_x0.iloc[i][var_name]
        cf1_val = df_cf_x1.iloc[i][var_name]
        
        obs_idx = all_values.index(obs_val)
        cf0_idx = all_values.index(cf0_val)
        cf1_idx = all_values.index(cf1_val)
        
        trans_matrix_0[obs_idx, cf0_idx] += 1
        trans_matrix_1[obs_idx, cf1_idx] += 1
    
    # Normalize to get probabilities
    trans_matrix_0 = trans_matrix_0 / trans_matrix_0.sum(axis=1, keepdims=True)
    trans_matrix_1 = trans_matrix_1 / trans_matrix_1.sum(axis=1, keepdims=True)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap for do(X=0)
    sns.heatmap(trans_matrix_0, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_values, yticklabels=all_values,
                cbar_kws={'label': 'Transition Probability'},
                ax=ax1)
    ax1.set_xlabel('Counterfactual Value', fontsize=11)
    ax1.set_ylabel('Observational Value', fontsize=11)
    ax1.set_title(f'{var_name}: Obs → do({results["intervention_name"]}=0)', fontsize=12, fontweight='bold')
    
    # Heatmap for do(X=1)
    sns.heatmap(trans_matrix_1, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=all_values, yticklabels=all_values,
                cbar_kws={'label': 'Transition Probability'},
                ax=ax2)
    ax2.set_xlabel('Counterfactual Value', fontsize=11)
    ax2.set_ylabel('Observational Value', fontsize=11)
    ax2.set_title(f'{var_name}: Obs → do({results["intervention_name"]}=1)', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Transition Matrices for {var_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return trans_matrix_0, trans_matrix_1



# try
def sample_sequences_from_multiple_files(
    model, 
    tokenizer, 
    json_files: List[str],
    num_samples_per_file: int = 100,
    verbose: bool = True
) -> Dict[str, List]:

    all_samples = {}
    
    for json_file in tqdm(json_files, disable=not verbose, desc="Processing files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            sequence_sample_space = json.load(f)
        
        samples = []
        for _ in tqdm(range(num_samples_per_file), disable=not verbose, 
                     desc=f"Sampling from {json_file}", leave=False):
            sample = sample_sequence(model, tokenizer, sequence_sample_space)
            samples.append(sample)
        
        all_samples[json_file] = samples
    
    return all_samples


def format_multiple_samples_as_dataframe(
    samples_dict: Dict[str, List],
    format_function,
    intervention_vars: Optional[List[str]] = None,
    outcome_vars: Optional[List[str]] = None,
    save_logprobs: bool = True
) -> Dict[str, pd.DataFrame]:

    formatted_dfs = {}
    
    for file_name, sequences in samples_dict.items():
        df = format_function(
            sequences=sequences,
            name_of_intervention=intervention_vars[0] if intervention_vars else None,
            name_of_outcome=outcome_vars[0] if outcome_vars else None,
            save_logprobs=save_logprobs
        )
        formatted_dfs[file_name] = df
    
    return formatted_dfs


def compare_variables_across_files(
    dfs_dict: Dict[str, pd.DataFrame],
    variables: List[str],
    file_labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    if file_labels is None:
        file_labels = {k: k for k in dfs_dict.keys()}
    
    comparison_data = []
    
    for file_name, df in dfs_dict.items():
        label = file_labels.get(file_name, file_name)
        for var in variables:
            if var in df.columns:
                value_counts = df[var].value_counts()
                for value, count in value_counts.items():
                    comparison_data.append({
                        'File': label,
                        'Variable': var,
                        'Value': value,
                        'Count': count,
                        'Percentage': (count / len(df)) * 100
                    })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def visualize_variable_comparison(
    comparison_df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    figsize: tuple = (15, 10),
    plot_type: str = 'bar'
) -> None:
    if variables is None:
        variables = comparison_df['Variable'].unique()
    
    num_vars = len(variables)
    num_cols = min(3, num_vars)
    num_rows = (num_vars + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if num_vars > 1 else [axes]
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        var_data = comparison_df[comparison_df['Variable'] == var]
        
        if plot_type == 'bar':
            pivot_data = var_data.pivot_table(
                index='Value', 
                columns='File', 
                values='Percentage', 
                fill_value=0
            )
            pivot_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylabel('Percentage (%)')
        
        elif plot_type == 'box':
            files = var_data['File'].unique()
            data_by_file = [var_data[var_data['File'] == f]['Percentage'].values for f in files]
            ax.boxplot(data_by_file, labels=files)
            ax.set_ylabel('Percentage (%)')
        
        ax.set_title(f'Variable: {var}')
        ax.set_xlabel('Value' if plot_type == 'bar' else 'File')
        ax.legend(title='File', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for idx in range(num_vars, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def create_summary_statistics(
    dfs_dict: Dict[str, pd.DataFrame],
    variables: List[str],
    file_labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:

    if file_labels is None:
        file_labels = {k: k for k in dfs_dict.keys()}
    
    stats_list = []
    
    for file_name, df in dfs_dict.items():
        label = file_labels.get(file_name, file_name)
        
        for var in variables:
            if var in df.columns:
                stats_list.append({
                    'File': label,
                    'Variable': var,
                    'Count': len(df),
                    'Unique_Values': df[var].nunique(),
                    'Most_Common': df[var].mode()[0] if len(df[var].mode()) > 0 else None,
                    'Most_Common_Freq': df[var].value_counts().iloc[0] if len(df[var].value_counts()) > 0 else 0
                })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df

def format_sequences_as_dataframe(sequences, name_of_intervention, name_of_outcome, save_logprobs=False):
    """
    Convert sequences to a pandas DataFrame with log probabilities and probabilities.
    
    Args:
        sequences: List of sequence samples
        name_of_intervention: Name of the intervention variable
        name_of_outcome: Name of the outcome variable
        save_logprobs: If True, save log probabilities for all variables; if False, only for intervention and outcome
    
    Returns:
        DataFrame with sampled values, log probabilities, and probabilities
    """
    df_rows = []
    possible_treatment_values = None
    possible_outcome_values = None
    
    for sequence in sequences:
        df_row = {
            'sampled_text': sequence['sampled_text']
        }
        df_row.update(dict(zip(sequence['variable_names'], sequence['sampled_indices'])))
        
        for i, name in enumerate(sequence['variable_names']):
            if save_logprobs or name in [name_of_intervention, name_of_outcome]:
                if name == name_of_outcome:
                    possible_outcome_values = range(len(sequence['sampled_logprobs'][i]))
                if name == name_of_intervention:
                    possible_treatment_values = range(len(sequence['sampled_logprobs'][i]))
                logprob_names = [f'logP({name}={value})' for value in range(len(sequence['sampled_logprobs'][i]))]
                df_row.update(dict(zip(logprob_names, sequence['sampled_logprobs'][i])))
        
        df_rows.append(df_row)
    
    data_df = pd.DataFrame(df_rows)
    
    # Convert log probabilities to probabilities for treatment
    logprob_treatment_names = [f'logP({name_of_intervention}={treatment_value})' for treatment_value in possible_treatment_values]
    prob_treatment_names = [f'P({name_of_intervention}={treatment_value})' for treatment_value in possible_treatment_values]
    data_df[prob_treatment_names] = torch.softmax(torch.tensor(data_df[logprob_treatment_names].values), dim=1)
    
    # Convert log probabilities to probabilities for outcome
    logprob_outcome_names = [f'logP({name_of_outcome}={outcome_value})' for outcome_value in possible_outcome_values]
    prob_outcome_names = [f'P({name_of_outcome}={outcome_value})' for outcome_value in possible_outcome_values]
    data_df[prob_outcome_names] = torch.softmax(torch.tensor(data_df[logprob_outcome_names].values), dim=1)
    
    return data_df
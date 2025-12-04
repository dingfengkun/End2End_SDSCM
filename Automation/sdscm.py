import json
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd



def chunk_continuation(model, tokenizer, context, candidate_set, sum=False, verbose=False):
    """
    Modified for complete sentences without prefix/suffix.
    """
    if context:
        input_ids = tokenizer(context, return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            past_key_values = outputs.past_key_values
    else:
        past_key_values = None

    log_probs = []
    for candidate in candidate_set:
        candidate_ids = tokenizer(candidate, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            if past_key_values is not None:
                outputs = model(candidate_ids, past_key_values=past_key_values)
            else:
                outputs = model(candidate_ids)
            logits = outputs.logits
            token_log_probs = logits[0, -1, :].log_softmax(dim=-1)
            
            if sum:
                log_probs.append(token_log_probs.index_select(0, candidate_ids[0, :]).sum().item())
            else:
                log_probs.append(token_log_probs.index_select(0, candidate_ids[0, :]).mean().item())

    if verbose:
        for candidate, log_prob in zip(candidate_set, log_probs):
            print(f"{candidate}: {log_prob}")

    probs = torch.softmax(torch.tensor(log_probs), dim=0)
    sampled_cand = torch.multinomial(probs, 1).item()
    
    if verbose:
        print(f"Sampled: {candidate_set[sampled_cand]}")

    sample = {
        'sampled_text': candidate_set[sampled_cand],
        'full_text': f'{context} {candidate_set[sampled_cand]}' if context else candidate_set[sampled_cand],
        'sampled_index': sampled_cand,
        'candidate_logprobs': log_probs
    }

    return sample


def sample_sequence(model, tokenizer, sequence_sample_space):
    """
    Sample a sequence based on the causal structure.
    """
    sampled_indices = []
    sampled_logprobs = []
    sampled_text_list = []
    variable_names = []
    sequence_sample_space = sequence_sample_space["setup_sequence_sample_space"]
    for setup_dict in sequence_sample_space:
        variable_names.append(setup_dict['variable_name'])
        
        # Build context from parent variables
        context = ''
        if setup_dict['parent_indices'] is not None:
            for parent_index in setup_dict['parent_indices']:
                if parent_index < len(sampled_text_list):
                    context = context + ' ' + sampled_text_list[parent_index]
        context = context.strip()
        
        # Sample or intervene
        if setup_dict['intervention_choice'] is None:
            # Natural sampling
            chunk_sample = chunk_continuation(
                model=model,
                tokenizer=tokenizer,
                context=context,
                candidate_set=setup_dict['candidate_set'],
                sum=False,
                verbose=False
            )
            sampled_text = chunk_sample['sampled_text']
            sampled_index = chunk_sample['sampled_index']
            candidate_logprobs = chunk_sample['candidate_logprobs']
            
        elif setup_dict['intervention_choice'] == 'uniform':
            # Uniform random intervention
            candidate_set = setup_dict['candidate_set']
            sampled_index = np.random.randint(low=0, high=len(candidate_set))
            sampled_text = candidate_set[sampled_index]
            candidate_logprobs = (np.ones(len(candidate_set)) * float('-inf')).tolist()
            candidate_logprobs[sampled_index] = 0
            
        elif isinstance(setup_dict['intervention_choice'], int):
            # Fixed intervention
            sampled_index = setup_dict['intervention_choice']
            candidate_set = setup_dict['candidate_set']
            sampled_text = candidate_set[sampled_index]
            candidate_logprobs = (np.ones(len(candidate_set)) * float('-inf')).tolist()
            candidate_logprobs[sampled_index] = 0
        else:
            raise ValueError(f"Unsupported intervention_choice: {setup_dict['intervention_choice']}")
        
        sampled_text_list.append(sampled_text)
        sampled_indices.append(sampled_index)
        sampled_logprobs.append(candidate_logprobs)

    sample = {
        'sampled_text': ' '.join(sampled_text_list),
        'sampled_indices': sampled_indices,
        'sampled_logprobs': sampled_logprobs,
        'variable_names': variable_names,
    }
    
    return sample


def sample_sequences(model, tokenizer, sequence_sample_space, num_samples, verbose=True):
    """Sample multiple sequences."""
    samples = []
    for _ in tqdm(range(num_samples), disable=not verbose):
        sample = sample_sequence(model, tokenizer, sequence_sample_space)
        samples.append(sample)
    return samples


def sample_interventional_sequence(model, tokenizer, sequence_sample_space, index_of_intervention, intervention_choice):
    """Sample with intervention do(X=x)."""
    modified_sample_space = deepcopy(sequence_sample_space)
    
    if modified_sample_space[index_of_intervention]['exogenous']:
        raise ValueError('Cannot intervene on exogenous variable')
    
    for i in range(len(sequence_sample_space)):
        modified_sample_space[i]['intervention_choice'] = None
    
    modified_sample_space[index_of_intervention]['intervention_choice'] = intervention_choice
    
    return sample_sequence(model, tokenizer, modified_sample_space)


def sample_interventional_sequences(model, tokenizer, sequence_sample_space, num_samples, 
                                   index_of_intervention, intervention_choice, verbose=True):
    """Sample multiple interventional sequences."""
    samples = []
    for _ in tqdm(range(num_samples), disable=not verbose):
        sample = sample_interventional_sequence(
            model, tokenizer, sequence_sample_space, 
            index_of_intervention, intervention_choice
        )
        samples.append(sample)
    return samples


def sample_counterfactual_sequence(model, tokenizer, sequence_sample_space, sampled_indices, 
                                  evidence_indices, index_of_intervention, intervention_choice):
    """Sample counterfactual: what if X had been x?"""
    modified_sample_space = deepcopy(sequence_sample_space)
    
    # Fix exogenous and evidence variables
    for i in range(len(sequence_sample_space)):
        if modified_sample_space[i]['exogenous'] or i in evidence_indices:
            modified_sample_space[i]['intervention_choice'] = sampled_indices[i]
        else:
            modified_sample_space[i]['intervention_choice'] = None
    
    # Apply intervention
    if modified_sample_space[index_of_intervention]['exogenous']:
        raise ValueError('Cannot intervene on exogenous variable')
    modified_sample_space[index_of_intervention]['intervention_choice'] = intervention_choice
    
    return sample_sequence(model, tokenizer, modified_sample_space)


def sample_counterfactual_sequences(model, tokenizer, sequence_sample_space, samples, 
                                   evidence_indices, index_of_intervention, intervention_choice, verbose=True):
    """Sample counterfactuals for multiple samples."""
    cf_samples = []
    for sample in tqdm(samples, disable=not verbose):
        cf_sample = sample_counterfactual_sequence(
            model, tokenizer, sequence_sample_space, 
            sample['sampled_indices'], evidence_indices, 
            index_of_intervention, intervention_choice
        )
        cf_samples.append(cf_sample)
    return cf_samples


def plot_dag_from_sample_space(sequence_sample_space, title=None):
    """Visualize the causal DAG."""
    graph = nx.DiGraph()
    
    for i, setup_dict in enumerate(sequence_sample_space):
        node_name = setup_dict['variable_name']
        graph.add_node(node_name)
        
        if setup_dict['parent_indices'] is not None:
            for parent_index in setup_dict['parent_indices']:
                parent_name = sequence_sample_space[parent_index]['variable_name']
                graph.add_edge(parent_name, node_name)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                          node_size=3000, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, width=2, alpha=0.6)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def calculate_ate(model, tokenizer, config, num_samples=1000, verbose=True):
    """
    Calculate Average Treatment Effect (ATE).
    ATE = E[Y|do(X=1)] - E[Y|do(X=0)]
    """
    sequence_sample_space = config['setup_sequence_sample_space']
    index_of_intervention = config['index_of_intervention']
    index_of_outcome = config['index_of_outcome']
    
    # Sample under do(X=0) do counterfactual
    samples_x0 = sample_counterfactual_sequences(
        model, tokenizer, sequence_sample_space, 
        num_samples, index_of_intervention, 0, verbose
    )
    
    # Sample under do(X=1)
    samples_x1 = sample_counterfactual_sequences(
        model, tokenizer, sequence_sample_space,
        num_samples, index_of_intervention, 1, verbose
    )
    
    # Calculate outcome probabilities
    y1_given_x0 = np.mean([s['sampled_indices'][index_of_outcome] for s in samples_x0])
    y1_given_x1 = np.mean([s['sampled_indices'][index_of_outcome] for s in samples_x1])
    
    ate = y1_given_x1 - y1_given_x0
    
    if verbose:
        intervention_name = sequence_sample_space[index_of_intervention]['variable_name']
        outcome_name = sequence_sample_space[index_of_outcome]['variable_name']
        print(f"\n=== Average Treatment Effect ===")
        print(f"Intervention: {intervention_name}")
        print(f"Outcome: {outcome_name}")
        print(f"P({outcome_name}=1 | do({intervention_name}=0)): {y1_given_x0:.3f}")
        print(f"P({outcome_name}=1 | do({intervention_name}=1)): {y1_given_x1:.3f}")
        print(f"ATE: {ate:.3f}")
    
    return ate, samples_x0, samples_x1


def format_sequences_as_dataframe(sequences, name_of_intervention, name_of_outcome, save_logprobs=False):
    """
    Convert sequences to a pandas DataFrame with log probabilities and probabilities.
    
    Args:
        sequences: List of sequence samples
        name_of_intervention: Name of the intervention variable (can be None)
        name_of_outcome: Name of the outcome variable (can be None)
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
    
    # Convert log probabilities to probabilities for treatment (if intervention variable exists)
    if name_of_intervention is not None and possible_treatment_values is not None:
        logprob_treatment_names = [f'logP({name_of_intervention}={treatment_value})' for treatment_value in possible_treatment_values]
        prob_treatment_names = [f'P({name_of_intervention}={treatment_value})' for treatment_value in possible_treatment_values]
        # Check if columns exist before computing softmax
        if all(col in data_df.columns for col in logprob_treatment_names):
            data_df[prob_treatment_names] = torch.softmax(torch.tensor(data_df[logprob_treatment_names].values), dim=1).numpy()
    
    # Convert log probabilities to probabilities for outcome (if outcome variable exists)
    if name_of_outcome is not None and possible_outcome_values is not None:
        logprob_outcome_names = [f'logP({name_of_outcome}={outcome_value})' for outcome_value in possible_outcome_values]
        prob_outcome_names = [f'P({name_of_outcome}={outcome_value})' for outcome_value in possible_outcome_values]
        # Check if columns exist before computing softmax
        if all(col in data_df.columns for col in logprob_outcome_names):
            data_df[prob_outcome_names] = torch.softmax(torch.tensor(data_df[logprob_outcome_names].values), dim=1).numpy()
    
    return data_df
import pandas as pd
import re
from copy import deepcopy
import json
def discrete_candidates(df, threshold=3):
    """
    Convert continuous numeric columns into discrete categories (3-class bins),
    generating unique, column-specific labels (e.g., 'age_low', 'age_mid', 'age_high').

    Parameters:
        df (pd.DataFrame): Input dataset.
        threshold (int): Minimum number of unique values required 
                         to consider binning a numeric variable.

    Returns:
        pd.DataFrame: Transformed DataFrame with "_3class" columns added (original columns kept).
    """
    df_transformed = df.copy()
    new_cols = []

    for col in df.columns:
        series = df[col]

        # Only process numeric columns
        if not pd.api.types.is_numeric_dtype(series):
            continue

        unique_count = series.nunique(dropna=True)
        if unique_count <= threshold:
            continue

        new_col = f"{col}_3class"

        try:
            # Generate unique labels for this column
            labels = [f"{col}_low", f"{col}_mid", f"{col}_high"]
            df_transformed[new_col] = pd.qcut(
                series, 3, labels=labels, duplicates="drop"
            )

        except ValueError:
            # Handle columns with too few distinct quantile edges
            try:
                bins = pd.qcut(series, 3, duplicates="drop", retbins=True)[1]
                n_bins = len(bins) - 1
                if n_bins <= 1:
                    continue  
                new_labels = [f"{col}_{i}" for i in range(n_bins)]
                df_transformed[new_col] = pd.qcut(
                    series, n_bins, labels=new_labels, duplicates="drop"
                )
            except Exception:
                continue
        df_transformed.drop(columns=[col], inplace=True)
        new_cols.append(new_col)

    return df_transformed




def extract_variables(df)-> list:
    # Variables that are not treatment or outcomes
    vars = [col for col in df.columns]
    return vars



def build_long_candidates_set(
    tokenizer,
    model,
    df,
    description=None,
    few_examples = [
        "Sentence: The passenger is male and is recorded as having a masculine gender identity.",
        "Sentence: The passenger did not survive the disaster and was listed among those who perished.",
        "Sentence: The passenger traveled in third class, indicating a lower-cost ticket category for the voyage."
    ]

    ,
    max_new_tokens=40,
    deterministic=True  
):
    """
    Generate concise English sentences for each unique variable value.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        df (pd.DataFrame): Input dataset
        description (str, optional): Dataset description
        few_examples (list[str]): Example sentences to guide style
        max_new_tokens (int): Generation length
        deterministic (bool): If True, use deterministic decoding (greedy)

    Returns:
        dict[str, list[str]]: {column_name: [generated sentences]}
    """
    all_candidates = {}
    device = model.device

    # Format few-shot examples
    style_examples = "\n".join(f"- {ex}" for ex in few_examples)

    for col in df.columns:

        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) == 0:
            continue

        col_candidates = []
        desc_text = f"Dataset context: {description}\n" if description else ""

        for val in unique_vals:

            # ---------- 1. Stronger and cleaner prompt ----------
            prompt = (
                f"Write one short factual English sentence describing:\n"
                f"{desc_text}"
                f"Variable: {col}\n"
                f"Value: {val}\n\n"
                f"Write one slightly longer descriptive sentence (12–18 words).\n"
                f"Follow the style of these examples:\n"
                f"{style_examples}\n\n"
                f"Sentence:"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,         
                temperature=0.0,          
                repetition_penalty=1.1,   
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
            # Raw decoded text
            gen_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
            decoded = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            print(f"[Raw output]{decoded}")

            # ---------- 3. Keep only the FIRST sentence ----------
            first_sentence = re.split(r'(?<=[.!?])\s+', decoded)[0].strip()
            print(f"[True output]{first_sentence}")

            col_candidates.append(first_sentence)

        # ---------- 7. Save if any results ----------
        if col_candidates:
            all_candidates[col] = list(dict.fromkeys(col_candidates))
            print(f"Added column: {col} ({len(col_candidates)} sentences)")

    return all_candidates



def fill_candidate_sets(json_framework, sentences_dict, output_path=None):
    """
    Fill the candidate_set fields in JSON framework with sentences from dictionary.
    
    Args:
        json_framework: JSON config with empty candidate_sets (can be file path or dict)
        sentences_dict: Dictionary with variable names as keys and list of sentences as values
        output_path: Optional path to save the filled configuration
        
    Returns:
        Updated JSON configuration with filled candidate_sets
    
    # Method 1: Load from file
    filled_config = fill_candidate_sets('titanic_framework.json', sentences_data)
    
    # Method 2: Use dict directly
    with open('titanic_framework.json', 'r') as f:
        framework = json.load(f)
    filled_config = fill_candidate_sets(framework, sentences_data)
    
    # Method 3: Save to file
    filled_config = fill_candidate_sets('titanic_framework.json', 
                                        sentences_data, 
                                        'titanic_filled.json')
    """
    # Load JSON if it's a file path
    if isinstance(json_framework, str):
        with open(json_framework, 'r') as f:
            config = json.load(f)
    else:
        config = deepcopy(json_framework)
    
    # Fill in the candidate_sets
    for variable_config in config['setup_sequence_sample_space']:
        variable_name = variable_config['variable_name']
        if variable_name in sentences_dict:
            variable_config['candidate_set'] = sentences_dict[variable_name]
            print(f"Filled {variable_name}: {len(sentences_dict[variable_name])} candidates")
        else:
            print(f"⚠ Warning: No candidates found for variable '{variable_name}'")
    
    # Update intervention_choices based on actual candidate count
    intervention_idx = config['index_of_intervention']
    intervention_var_name = config['setup_sequence_sample_space'][intervention_idx]['variable_name']
    
    if intervention_var_name in sentences_dict:
        config['intervention_choices'] = list(range(len(sentences_dict[intervention_var_name])))
        print(f"Updated intervention_choices for {intervention_var_name}: {config['intervention_choices']}")
    
    # Update possible_outcome_choices based on actual candidate count  
    outcome_idx = config['index_of_outcome']
    outcome_var_name = config['setup_sequence_sample_space'][outcome_idx]['variable_name']
    
    if outcome_var_name in sentences_dict:
        config['possible_outcome_choices'] = list(range(len(sentences_dict[outcome_var_name])))
        print(f"Updated possible_outcome_choices for {outcome_var_name}: {config['possible_outcome_choices']}")
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nFilled configuration saved to {output_path}")
    
    return config


def fill_DAG(json_framework, dag_edges, exogenous_list, output_path=None):
    """
    Fill the DAG edges and exogenous values in JSON framework.

    Args:
        json_framework: dict or file path of JSON
        dag_edges: dict, variable -> list of parent variable names
        exogenous_list: list of bool, one per variable in order
        output_path: optional save path
    """

    # Load JSON if file path
    if isinstance(json_framework, str):
        with open(json_framework, "r") as f:
            config = json.load(f)
    else:
        config = json_framework

    nodes = config["setup_sequence_sample_space"]

    # Check exogenous list length
    if len(exogenous_list) != len(nodes):
        raise ValueError("exogenous_list length must match number of variables.")

    # Build mapping: variable_name -> index
    name_to_idx = {node["variable_name"]: i for i, node in enumerate(nodes)}

    # Fill DAG
    for i, node in enumerate(nodes):
        var = node["variable_name"]

        # Convert parent names -> indices
        if var in dag_edges:
            parent_indices = dag_edges[var]
        else:
            parent_indices = []

        node["parent_indices"] = parent_indices
        node["exogenous"] = bool(exogenous_list[i])

    # Save if needed
    if output_path:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    return config

   
def init_framework_from_df(df, description=None,save_path=None):
    """
    Initialize JSON framework from DataFrame.

    Args:
        df (pd.DataFrame): Input dataset
        description (str, optional): Dataset description

    Returns:
        dict: JSON framework
    """
    variables = extract_variables(df)

    config = {
        "description": description or "No description provided.",
        "setup_sequence_sample_space": [
            {
                "variable_name": var,
                "candidate_set": [],
                "parent_indices": [],
                "exogenous": False,
                "intervention_choice": None
            }
            for var in variables
        ],
        "index_of_intervention": 0,
        "intervention_choices": [],
        "index_of_outcome": len(variables) - 1,
        "possible_outcome_choices": []
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        print(f"Saved to {save_path}") 
    return config
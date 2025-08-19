import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional, Dict
import unicodedata
from circuit_tracer import ReplacementModel

from .utils import *


def load_feature_analysis(
    feature, 
    model, 
    save_dir, 
    analyze_if_not_found=True, 
    save=False, 
    all_results=None
):
    """
    Load or compute the analysis for a specific feature in a model.

    This function tries to retrieve the analysis of a given feature from:
      1. An in-memory cache (`all_results`) if provided.
      2. A JSON file on disk corresponding to the feature's layer.
      3. If not found and `analyze_if_not_found=True`, it computes the analysis using `analyze_feature`.

    Optionally, it can save new analyses to disk for future reuse.

    Parameters
    ----------
    feature : object
        The feature to analyze. Must have attributes `layer` and `feature_idx`.
    model : object
        The model containing the feature.
    save_dir : Path
        Directory where feature analysis JSON files are stored.
    analyze_if_not_found : bool, default True
        If True, will compute analysis if it is not found in memory or on disk.
    save : bool, default False
        If True, will save newly computed results to disk.
    all_results : dict or None, default None
        Optional in-memory cache of previously computed results organized as:
            {layer: [feature_results_dicts...]}

    Returns
    -------
    dict
        The analysis result for the requested feature, including numeric attributions,
        tokens, and optionally a human-readable description.

    Raises
    ------
    ValueError
        If the feature is not found in memory or disk and `analyze_if_not_found=False`.
    TypeError
        If there is an error decoding the JSON file for the layer.
    """
    
    # Check in-memory cache first
    if all_results is not None:
        if feature.layer in all_results.keys():
            if feature.feature_idx in [x['feature']['feature_idx'] for x in all_results[feature.layer]]:
                return np.array(all_results[feature.layer])[
                    [x['feature']['feature_idx'] == feature.feature_idx for x in all_results[feature.layer]]
                ][0]
    
    # Path to JSON file storing results for this layer
    save_path = save_dir / f"feature_analysis_layer_{feature.layer}.json"
    
    # Load existing results from disk if available
    results = []
    if save_path.exists():
        try: 
            with open(save_path, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            raise TypeError(f"Error decoding JSON from {save_path}, trying to load feature {feature}")
    
    # Determine whether to compute the feature analysis
    if feature.feature_idx not in [x['feature']['feature_idx'] for x in results] and analyze_if_not_found:
        result = analyze_feature(model, feature, save_dir=save_dir, save_result=save)
    elif feature.feature_idx in [x['feature']['feature_idx'] for x in results]:
        result = np.array(results)[
            [x['feature']['feature_idx'] == feature.feature_idx for x in results]
        ][0]
    else: 
        raise ValueError(f"Feature {feature.feature_idx} not found in {save_path}.")
    
    # Save newly computed result to disk if requested
    if feature.feature_idx not in [x['feature']['feature_idx'] for x in results] and save:
        results.append(result)
        with open(save_path, "w") as f:
            results_sorted = sorted(results, key=lambda x: x['feature']['feature_idx'])
            json.dump(results_sorted, f, indent=2)
    
    return result


def generate_description(
    model: ReplacementModel, 
    feature: Feature, 
    save_dir: Union[Path, str] = "results/", 
    result: Optional[Dict] = None, 
    save_result: bool = False, 
    all_results = None
) -> Dict:
    """
    Generate a textual description of a model feature based on its activations
    and contributions from other features. Optionally saves the updated result.

    This function:
    1. Loads the feature analysis if not provided.
    2. Constructs a 'description' and 'description_soft' for the feature
       based on the top positive tokens.
    3. Incorporates contributions from other features recursively.
    4. Validates whether the generated tokens activate the target feature.
    5. Optionally saves the updated result to disk.

    Parameters
    ----------
    model : ReplacementModel
        The model containing the feature to describe.
    feature : Feature
        Feature to generate description for. Must have attributes 'layer' and 'feature_idx'.
    save_dir : Path or str, default "results/"
        Directory to load/save feature analysis JSON files.
    result : dict, optional
        Precomputed feature analysis result. If None, it will be loaded.
    save_result : bool, default False
        Whether to save the updated result to disk.
    all_results : dict, optional
        Optional in-memory cache of previously computed results for efficiency.

    Returns
    -------
    dict
        Updated feature analysis with keys:
        - 'description': list of validated tokens activating the feature.
        - 'description_soft': list of all positive tokens including contributions.
        - 'validation': bool indicating if the description activates the feature.
        - 'validation_soft': proportion of 'description_soft' tokens activating the feature.

    Notes
    -----
    - Handles different model types (Gemma, GPT, Llama) to determine the BOS token.
    - Recursively includes descriptions from contributing features.
    - Uses get_activation_with_stop to validate token contributions at the feature's layer.
    """
    
    # Load feature analysis if not already provided
    if not result:
        result = load_feature_analysis(feature, model, Path(save_dir), all_results=all_results)
    
    # Only generate description if it does not exist
    if "description" not in result:
        # Determine BOS token depending on the model type
        if "gemma" in model.cfg.model_name:
            bos_token = 2
        elif "gpt" in model.cfg.model_name:
            bos_token = None
        elif "Llama" in model.cfg.model_name:
            bos_token = 128000

        # Initialize descriptions from top positive tokens
        positive_mask = np.array(result['embedding']['positive_tokens_activation']) > 0
        result['description_soft'] = np.array(result['embedding']['top_positive_tokens'])[positive_mask].tolist()
        result['description'] = result['description_soft'].copy()
        
        # Recursively add descriptions from contributing features
        for f in result['feature_to_feature_contributions']:
            if (f['from_layer'] != 0) and ("gpt" in model.cfg.model_name):
                continue
            
            feature_result = load_feature_analysis(
                Feature(f['from_layer'], -1, f['from_feature_idx']),
                model,
                save_dir=save_dir,
                save=save_result,
                all_results=all_results
            )
            
            if "description" not in feature_result:
                feature_result = generate_description(
                    model,
                    Feature(f['from_layer'], -1, f['from_feature_idx']),
                    result=feature_result,
                    save_dir=save_dir,
                    save_result=save_result
                )
            
            if feature_result['description'] is None:
                continue
            
            result['description_soft'].extend(feature_result['description'])
            
            # Validate which contributing tokens actually activate this feature
            input_ids = [model.tokenizer(x).input_ids[-1] for x in feature_result['description']]
            if bos_token is not None:
                input_ids = [bos_token] + input_ids
            acts = get_activation_with_stop(model, torch.tensor(input_ids), stop_at_layer=feature.layer)
            activations = acts[feature.layer, 1 if bos_token is not None else 0:, feature.feature_idx]
            
            added_description = np.array(feature_result['description'])[activations.to(torch.float32).cpu().numpy() > 0].tolist()
            result['description'].extend(added_description)
        
        # Deduplicate descriptions and validate
        if result['description']:
            result['description'] = np.unique(result['description']).tolist()
            result['description_soft'] = np.unique(result['description_soft']).tolist()
            
            input_ids = [model.tokenizer(x).input_ids[-1] for x in result['description']]
            if bos_token is not None:
                input_ids = [bos_token] + input_ids
            acts = get_activation_with_stop(model, torch.tensor(input_ids), stop_at_layer=feature.layer)
            result['validation'] = acts[feature.layer, 1 if bos_token is not None else 0:, feature.feature_idx].sum().item() > 0
        else:
            result['description'] = None
            result['validation'] = False

        # Compute soft validation (proportion of activating tokens)
        if result['description_soft']:
            input_ids = [model.tokenizer(x).input_ids[-1] for x in result['description_soft']]
            if bos_token is not None:
                input_ids = [bos_token] + input_ids
            acts = get_activation_with_stop(model, torch.tensor(input_ids), stop_at_layer=feature.layer)
            if len(acts[feature.layer, 1 if bos_token is not None else 0:, feature.feature_idx]) != 0:
                result['validation_soft'] = (acts[feature.layer, 1 if bos_token is not None else 0:, feature.feature_idx] > 0).sum().item() / len(acts[feature.layer, 1 if bos_token is not None else 0:, feature.feature_idx])
            else:
                result['validation_soft'] = 0.0
        elif result['description'] is not None:
            result['validation_soft'] = 1.0
        else:
            result['description_soft'] = None
            result['validation_soft'] = 0.0
        
        # Optionally save the updated result
        if save_result:
            results = []
            save_path = Path(save_dir) / f"feature_analysis_layer_{feature.layer}.json"
            if save_path.exists():
                with open(save_path, "r") as f:
                    results = json.load(f)
            else:
                os.makedirs(save_dir, exist_ok=True)
            results.append(result)
            results_sorted = sorted(results, key=lambda x: x['feature']['feature_idx'])
            with open(save_path, "w") as f:
                json.dump(results_sorted, f, indent=2)
    
    return result

def analyze_feature(
    model: "ReplacementModel",
    feature: "Feature",
    threshold_tokens: float = 5.5,
    threshold_features: float = 4,
    save_dir: str = "results/",
    save_result: bool = False,
    all_results: Optional[dict] = None
) -> dict:
    """
    Analyze a single feature in a model, computing:
        - Top embedding tokens contributing to the feature
        - Top output logits influenced by the feature
        - Feature-to-feature contributions across previous layers

    Args:
        model: ReplacementModel instance
        feature: Feature object specifying layer and index
        threshold_tokens: Z-score threshold for token contribution outliers
        threshold_features: Z-score threshold for feature-to-feature contributions
        save_dir: Directory to save descriptive results
        save_result: Whether to save intermediate description results
        all_results: Previously computed results (used for hierarchical analysis)

    Returns:
        result: Dictionary with embeddings, output logits, feature-to-feature contributions,
                and descriptive information.
    """
    result = {
        "feature": {"layer": feature.layer, "feature_idx": feature.feature_idx},
        "embedding": {
            "top_positive_tokens": [],
            "top_negative_tokens": [],
            "positive_tokens_activation": [],
        },
        "output_logits": {"top_positive_tokens": [], "top_negative_tokens": []},
        "feature_to_feature_contributions": [],
    }

    # --- Embedding Analysis ---
    # Project the feature vector to token embeddings
    emb_top, emb_bottom, embeddings = get_projection_to_embeddings(
        model, feature, k=1000, return_embeddings=True
    )

    # Identify positive outlier tokens (most activating)
    outliers_top, _ = get_outliers(emb_top[0], threshold=threshold_tokens)
    result["embedding"]["top_positive_tokens"] = [
        model.tokenizer.decode(x) for x in emb_top[1][outliers_top].tolist()
    ]
    result["embedding"]["top_positive_token_ids"] = emb_top[1][outliers_top].tolist()
    result["embedding"]["top_positive_tokens_contribution"] = emb_top[0][outliers_top].tolist()

    # For each top positive token, compute activation at the feature layer
    for token_id in result["embedding"]["top_positive_token_ids"]:
        acts = get_activation_with_stop(
            model, torch.tensor([2, token_id], dtype=torch.long), stop_at_layer=feature.layer
        )
        result["embedding"]["positive_tokens_activation"].append(
            acts[feature.layer, -1, feature.feature_idx].item()
        )

    # Negative embeddings (mostly placeholder; not meaningful in practice)
    _, outliers_bot = get_outliers(emb_bottom[0], threshold=threshold_tokens)
    result["embedding"]["top_negative_tokens"] = [
        model.tokenizer.decode(x) for x in emb_bottom[1][outliers_bot].tolist()
    ]
    result["embedding"]["top_negative_tokens_ids"] = emb_bottom[1][outliers_bot].tolist()
    result["embedding"]["top_negative_tokens_contribution"] = emb_bottom[0][outliers_bot].tolist()

    # --- Output Logits Analysis ---
    unemb_top, unemb_bot = get_unembeddings(model, feature, k=1000)
    outliers_top, _ = get_outliers(unemb_top[0], threshold=threshold_tokens)
    result["output_logits"]["top_positive_tokens"] = [
        model.tokenizer.decode(x) for x in unemb_top[1][outliers_top].tolist()
    ]
    _, outliers_bot = get_outliers(unemb_bot[0], threshold=threshold_tokens)
    result["output_logits"]["top_negative_tokens"] = [
        model.tokenizer.decode(x) for x in unemb_bot[1][outliers_bot].tolist()
    ]

    # --- Feature-to-Feature Contributions ---
    total_absolute_contribution = embeddings.abs().sum().item()
    feature_contrib_list = []

    for layer in range(feature.layer):
        # Compute top contributing features from earlier layers
        top_contrib, _, contrib_vec = get_input_independent_features(model, layer, feature, k=100)
        total_absolute_contribution += contrib_vec.abs().sum().item()

        # Identify outliers above threshold
        outliers, _ = get_outliers(top_contrib[0], threshold=threshold_features)
        for score, idx in zip(top_contrib[0][outliers], top_contrib[1][outliers]):
            feature_contrib_list.append({
                "from_layer": layer,
                "from_feature_idx": idx.item(),
                "contribution": score.item()
            })

    # Compute relative contributions
    for entry in feature_contrib_list:
        entry["relative_contribution"] = abs(entry["contribution"]) / total_absolute_contribution

    result["feature_to_feature_contributions"] = sorted(
        feature_contrib_list, key=lambda x: x["relative_contribution"], reverse=True
    )

    # Compute relative contribution of top positive/negative embedding tokens
    result["embedding"]["top_positive_tokens_relative_contribution"] = (
        torch.Tensor(result["embedding"]["top_positive_tokens_contribution"]).abs() / total_absolute_contribution
    ).tolist()
    result["embedding"]["top_negative_tokens_relative_contribution"] = (
        torch.Tensor(result["embedding"]["top_negative_tokens_contribution"]).abs() / total_absolute_contribution
    ).tolist()

    # Total contribution detected via input-invariant analysis
    result["total_input_invariant_detected_contribution"] = (
        sum(result["embedding"]["top_positive_tokens_relative_contribution"]) +
        sum(x["relative_contribution"] for x in result["feature_to_feature_contributions"])
    )
    result["total_input_invariant_contribution"] = total_absolute_contribution

    # Generate a human-readable description for this feature
    result = generate_description(
        model, feature, save_dir=Path(save_dir), result=result,
        save_result=save_result, all_results=all_results
    )

    return result


def analyze_all_features(
    model: "ReplacementModel",
    save_dir: Union[str, Path] = "results/",
    checkpoint: int = 1000
):
    """
    Run analysis for all features in a model layer by layer.

    Saves JSON files per layer containing feature analysis results.

    Args:
        model: ReplacementModel instance
        save_dir: Directory to store analysis JSON files
        checkpoint: Frequency of saving intermediate results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Initialize/load results for each layer
    for layer in range(model.cfg.n_layers):
        save_path = save_dir / f"feature_analysis_layer_{layer}.json"
        if save_path.exists():
            with open(save_path, "r") as f:
                all_results[layer] = json.load(f)
        else:
            all_results[layer] = []

    # Iterate layers
    for layer in range(model.cfg.n_layers):
        save_path = save_dir / f"feature_analysis_layer_{layer}.json"
        print(f"Layer {layer} feature analysis:")
        existing_feature_indices = set(
            entry["feature"]["feature_idx"] for entry in all_results[layer]
        )

        # Iterate features in current layer
        for feature_idx in tqdm(range(model.transcoders[0].W_enc.shape[1])):
            if feature_idx in existing_feature_indices:
                continue
            feature = Feature(layer=layer, feature_idx=feature_idx, pos=-1)
            result = analyze_feature(
                model, feature,
                save_dir=save_dir,
                all_results={ll: v for ll, v in all_results.items() if ll < layer}
            )
            all_results[layer].append(result)

            # Periodically save checkpoint
            if feature_idx % checkpoint == 0:
                results_sorted = sorted(all_results[layer], key=lambda x: x["feature"]["feature_idx"])
                with open(save_path, "w") as f:
                    json.dump(results_sorted, f, indent=2)

        # Save final layer results
        results_sorted = sorted(all_results[layer], key=lambda x: x["feature"]["feature_idx"])
        with open(save_path, "w") as f:
            json.dump(results_sorted, f, indent=2)

        print(f"[✓] Finished Layer {layer}\n")


# --- Utility Functions ---

def sanitize_token(token: str) -> str:
    """
    Convert a token to a human-readable printable string.

    Replaces control characters with escaped forms.
    """
    safe = ""
    for ch in token:
        if ch.isprintable() and not unicodedata.category(ch).startswith("C"):
            safe += ch
        else:
            safe += repr(ch).strip("'")
    return safe


def print_feature_analysis(
    feature_or_entry: Union["Feature", dict],
    save_dir: Union[str, Path] = "results/",
    model: Optional["ReplacementModel"] = None
):
    """
    Pretty-print analysis of a single feature.

    Args:
        feature_or_entry: Either a Feature object or a previously computed result dict
        save_dir: Directory where JSON analysis files are stored
        model: Optional model instance (used to compute missing feature analyses)
    """
    if isinstance(feature_or_entry, dict):
        entry = feature_or_entry
        layer = entry["feature"]["layer"]
        feature_idx = entry["feature"]["feature_idx"]
    else:
        feature = feature_or_entry
        layer = feature.layer
        feature_idx = feature.feature_idx
        save_path = Path(save_dir) / f"feature_analysis_layer_{layer}.json"
        if not save_path.exists():
            print(f"[!] No analysis file found for layer {layer}.")
            return
        with open(save_path, "r") as f:
            all_results = json.load(f)
        entry = next((e for e in all_results if e["feature"]["feature_idx"] == feature_idx), None)
        if entry is None:
            print(f"[!] Feature L{layer} F{feature_idx} not found in analysis.")
            return

    print(f"\n[✓] Feature analysis for L{layer} F{feature_idx}:\n")

    # --- Embedding contributions ---
    emb = entry["embedding"]
    pos_emb = emb["top_positive_tokens"]
    pos_contribs = emb.get("top_positive_tokens_contribution", [])
    pos_rel = emb.get("top_positive_tokens_relative_contribution", [])
    pos_act = emb.get("positive_tokens_activation", [])
    pos_tokens = emb.get("top_positive_token_ids", [])

    print(f"📝 Description: {'   |   '.join(entry['description']) if entry['description'] else None}")

    if not pos_emb or (np.array(pos_act) > 0).sum() == 0:
        print("No directly influencing tokens found for this feature.")
    else:
        print("🔍 Top contributing embedding tokens:")
        for tok_id, tok, val, rel, act in zip(pos_tokens, pos_emb, pos_contribs, pos_rel, pos_act):
            if act > 0:
                print(
                    f" - Token ID {tok_id:>5} | {sanitize_token(tok):>20} | Activation: {act:6.4f} | "
                    f"Contribution: {val:8.3f} | Rel: {rel:6.8%}"
                )

    # --- Output logits ---
    pos_logits = entry["output_logits"]["top_positive_tokens"]
    neg_logits = entry["output_logits"]["top_negative_tokens"]
    print("\n📤 Output logits:")
    print(" + Top positive tokens: " + ('   |   '.join(sanitize_token(tok) for tok in pos_logits) if pos_logits else "None"))
    print(" - Top negative tokens: " + ('   |   '.join(sanitize_token(tok) for tok in neg_logits) if neg_logits else "None"))

    # --- Feature-to-feature contributions ---
    contribs = entry.get("feature_to_feature_contributions", [])
    if contribs:
        print("\n🧩 Top contributing features:")
        sorted_contribs = sorted(contribs, key=lambda x: x["relative_contribution"], reverse=True)
        for c in sorted_contribs:
            if c["from_layer"] != 0 and model and "gpt" in model.cfg.model_name:
                continue            

            save_path = Path(save_dir) / f"feature_analysis_layer_{c['from_layer']}.json"
            try:
                f_entry = None
                if save_path.exists():
                    with open(save_path, "r") as f:
                        all_results = json.load(f)
                    f_entry = next(
                        (x for x in all_results if x["feature"]["feature_idx"] == c["from_feature_idx"]),
                        None
                    )
                elif model is not None:
                    f_entry = analyze_feature(model, Feature(c["from_layer"], -1, c["from_feature_idx"]))

                if f_entry and f_entry["description"] is not None:
                    print(
                        f" → From L{c['from_layer']} F{c['from_feature_idx']:>4} | "
                        f"Contribution: {c['contribution']:+.4f} | "
                        f"Relative: {c['relative_contribution']:.8%}"
                    )
                    print("   └─ Description:", "    |   ".join(f_entry['description']) if f_entry['description'] is not None else None)
            except Exception as e:
                print(f"[!] Error while analyzing L{c['from_layer']} F{c['from_feature_idx']}: {e}")
    else:
        print("\n[•] No significant feature-to-feature contributions found.")

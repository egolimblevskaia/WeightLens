import torch
import numpy as np
from tqdm import tqdm
from typing import NamedTuple, Tuple
from circuit_tracer import ReplacementModel
from transformer_lens.utils import to_numpy


class Feature(NamedTuple):
    """Represents a specific feature in a transcoder."""
    layer: int          # Layer index where the feature is located
    pos: int            # Position in the sequence
    feature_idx: int    # Index of the feature within the layer


def get_outliers(
    input_array: torch.Tensor, 
    threshold: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identify outliers in an array based on z-scores.

    Args:
        input_array: A 1D tensor of values.
        threshold: Z-score threshold for identifying outliers.

    Returns:
        top_outlier_indices: Indices of values greater than +threshold.
        bottom_outlier_indices: Indices of values less than -threshold.
    """
    mean = input_array.mean()
    std = input_array.std()

    z_scores = (input_array - mean) / std
    bottom_outlier_indices = (z_scores < -threshold).nonzero(as_tuple=True)[0]
    top_outlier_indices = (z_scores > threshold).nonzero(as_tuple=True)[0]

    return top_outlier_indices, bottom_outlier_indices


def get_unembeddings(
    model: ReplacementModel, 
    feature: Feature, 
    k: int = 5
):
    """
    Compute top and bottom k token unembeddings for a given transcoder feature.

    Args:
        model: The ReplacementModel containing transcoders.
        feature: The feature to analyze.
        k: Number of top/bottom tokens to return.

    Returns:
        (val_top, ind_top), (val_bottom, ind_bottom): Values and indices for top/bottom tokens.
    """
    unembeddings = model.transcoders[feature.layer].W_dec[feature.feature_idx, :] @ model.W_U
    val_top, ind_top = torch.topk(unembeddings, k=k, sorted=True)
    val_bottom, ind_bottom = torch.topk(unembeddings, k=k, sorted=True, largest=False)

    return (val_top, ind_top), (val_bottom, ind_bottom)


def get_input_independent_features(
    model: ReplacementModel, 
    base_layer_idx: int, 
    feature: Feature, 
    k: int = 7
):
    """
    Project a transcoder feature onto features of a base layer.

    Args:
        model: The ReplacementModel containing transcoders.
        base_layer_idx: Index of the layer to project onto.
        feature: The target feature to project.
        k: Number of top/bottom features to return.

    Returns:
        (top_vals, top_idxs), (bot_vals, bot_idxs): Top and bottom projections.
    """
    feature_projection = (
        model.transcoders[base_layer_idx].W_dec 
        @ model.transcoders[feature.layer].W_enc[:, feature.feature_idx]
    )
    top_vals, top_idxs = torch.topk(feature_projection, k=k)
    bot_vals, bot_idxs = torch.topk(feature_projection, k=k, largest=False)

    return (top_vals, top_idxs), (bot_vals, bot_idxs), feature_projection


def get_projection_to_embeddings(
    model: ReplacementModel, 
    feature: Feature, 
    k: int = 5,
    return_embeddings: bool = False
):
    """
    Project a transcoder feature directly into the model’s embedding space.

    Args:
        model: The ReplacementModel containing embeddings.
        feature: The feature to project.
        k: Number of top/bottom embeddings to return.

    Returns:
        (top_vals, top_idxs), (bot_vals, bot_idxs): Top and bottom embedding matches.
    """
    embedding_projection = model.W_E @ model.transcoders[feature.layer].W_enc[:, feature.feature_idx]
    top_vals, top_idxs = torch.topk(embedding_projection, k=k, sorted=True)
    bot_vals, bot_idxs = torch.topk(embedding_projection, k=k, sorted=True, largest=False)

    if return_embeddings:
        return (top_vals, top_idxs), (bot_vals, bot_idxs), embedding_projection
    return (top_vals, top_idxs), (bot_vals, bot_idxs)


def get_activation_with_stop(model, input, stop_at_layer, requires_grad=False, return_logits=False):
    """
    Get the activation of the model at a specific layer.
    """
    activation_cache, activation_hooks = model._get_activation_caching_hooks(
        sparse=False,
        zero_bos=False,
        apply_activation_function=True,
    )
    if not requires_grad:
        with torch.inference_mode(), model.hooks(activation_hooks):
            logits = model.forward(input, stop_at_layer=stop_at_layer+1)
    else: 
        with model.hooks(activation_hooks):
            logits = model.forward(input, stop_at_layer=stop_at_layer+1)
    if return_logits:
        return torch.stack(activation_cache[:stop_at_layer+1]), logits
    else: 
        return torch.stack(activation_cache[:stop_at_layer+1])
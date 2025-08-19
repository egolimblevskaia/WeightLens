"""
Partial extensions and utilities for circuit_tracer's ReplacementModel.

This file heavily builds on the original `ReplacementModel` implementation 
from the `circuit_tracer` package. Most functions, hooks, and transcoder 
loading logic are adapted from circuit_tracer, with minor extensions 
for partial replacement and custom activation caching.

References:
    - circuit_tracer.replacement_model.ReplacementModel
    - circuit_tracer.transcoder.SingleLayerTranscoder
"""
import torch
import yaml
from torch import nn
from huggingface_hub import hf_hub_download
from collections import namedtuple
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union


from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder
from circuit_tracer.utils.hf_utils import download_hf_uris, parse_hf_uri
from circuit_tracer.transcoder.single_layer_transcoder import load_relu_transcoder, load_gemma_scope_transcoder
from circuit_tracer.replacement_model import ReplacementMLP, ReplacementUnembed



def safe_deduplicate_attention_buffers(self) -> None:
    """
    Safely deduplicate attention buffers across transformer blocks.

    This method collects attention masks and rotary embeddings once,
    then reassigns the shared buffers to all attention modules in the model.

    It avoids errors when some blocks have rotary embeddings and others do not.
    """
    attn_masks = {}

    # Collect attention masks and rotary embeddings from all blocks
    for block in self.blocks:
        attn_masks[block.attn.attn_type] = block.attn.mask
        if hasattr(block.attn, "rotary_sin"):
            attn_masks["rotary_sin"] = block.attn.rotary_sin
        if hasattr(block.attn, "rotary_cos"):
            attn_masks["rotary_cos"] = block.attn.rotary_cos

    # Assign shared buffers back to each block's attention module
    for block in self.blocks:
        block.attn.mask = attn_masks[block.attn.attn_type]
        if "rotary_sin" in attn_masks:
            block.attn.rotary_sin = attn_masks["rotary_sin"]
        if "rotary_cos" in attn_masks:
            block.attn.rotary_cos = attn_masks["rotary_cos"]


def load_replacement_model_from_yaml(yaml_path: str, device: str = "cpu") -> ReplacementModel:
    """
    Load a ReplacementModel configured from a YAML file, including its transcoders.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        device (str): Device to load model and transcoders onto (default "cpu").

    Returns:
        ReplacementModel: The loaded ReplacementModel instance.

    Raises:
        NameError: If the model_name in YAML is not 'gpt2' 
                   (for gemma and llama there are internal implementations in ReplacementModel)
    """
    # Load YAML configuration
    with open(yaml_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Check supported model
    if config_data['model_name'] != "gpt2":
        raise NameError(f"Unsupported model_name: {config_data['model_name']}. Only 'gpt2' is supported.")

    # Patch ReplacementModel method for safe attention buffer deduplication
    ReplacementModel._deduplicate_attention_buffers = safe_deduplicate_attention_buffers

    # Load transcoders per layer
    transcoders = {}
    for transcoder_entry in config_data["transcoders"]:
        layer = transcoder_entry["layer"]
        file_path = hf_hub_download(
            repo_id=config_data['repo_id'],
            filename=transcoder_entry['filepath']
        )

        # Load state dict, filter out unexpected keys
        state_dict = torch.load(file_path, map_location=device, weights_only=False)
        state_dict = state_dict.get("state_dict", state_dict)
        filtered_state_dict = {k: v for k, v in state_dict.items() if k != "b_dec_out"}

        # Create and load SingleLayerTranscoder
        transcoder = SingleLayerTranscoder(
            d_model=config_data['d_model'],
            d_transcoder=config_data['d_transcoder'],
            activation_function=torch.nn.ReLU(),
            layer_idx=layer,
            skip_connection='W_skip' in state_dict
        )
        transcoder.load_state_dict(filtered_state_dict)
        transcoders[layer] = transcoder

    # Instantiate ReplacementModel with config and transcoders
    model_pretrained = ReplacementModel.from_pretrained_and_transcoders(config_data['model_name'], transcoders)

    return model_pretrained


TranscoderSettings = namedtuple(
    "TranscoderSettings", ["transcoders", "feature_input_hook", "feature_output_hook", "scan"]
)

def load_partial_transcoder_set(
    transcoder_config_file: str,
    device: Optional[torch.device] = torch.device("cuda"),
    dtype: Optional[torch.dtype] = torch.float32,
) -> TranscoderSettings:
    """
    Function, adjusted from circuit_tracer.transcoder.single_layer_transcoder
    for loading Llama transcoders partially, i.e. for a few specified in the config layers. 

    Loads either a preset set of transformers, or a set specified by a file.

    Args:
        transcoder_config_file (str): _description_
        device (Optional[torch.device], optional): _description_. Defaults to torch.device('cuda').

    Returns:
        TranscoderSettings: A namedtuple consisting of the transcoder dict,
        and their feature input hook, feature output hook and associated scan.
    """

    scan = None
    # try to match a preset, and grab its config
    if transcoder_config_file == "llama":
        transcoder_config_file = "configs/llama-relu.yaml"
        scan = "llama-3-131k-relu"

    with open(transcoder_config_file, "r") as file:
        config = yaml.safe_load(file)

    sorted_transcoder_configs = sorted(config["transcoders"], key=lambda x: x["layer"])
    if scan is None:
        # the scan defaults to a list of transcoder ids, preceded by the model's name
        model_name_no_slash = config["model_name"].split("/")[-1]
        scan = [
            f"{model_name_no_slash}/{transcoder_config['id']}"
            for transcoder_config in sorted_transcoder_configs
        ]

    hf_paths = [
        t["filepath"] for t in sorted_transcoder_configs if t["filepath"].startswith("hf://")
    ]
    local_map = download_hf_uris(hf_paths)

    transcoders = {}
    for transcoder_config in sorted_transcoder_configs:
        path = transcoder_config["filepath"]
        if path.startswith("hf://"):
            local_path = local_map[path]
            repo_id = parse_hf_uri(path).repo_id
            if "gemma-scope" in repo_id:
                transcoder = load_gemma_scope_transcoder(
                    local_path, transcoder_config["layer"], device=device, dtype=dtype
                )
            else:
                transcoder = load_relu_transcoder(
                    local_path, transcoder_config["layer"], device=device, dtype=dtype
                )
        else:
            transcoder = load_relu_transcoder(
                path, transcoder_config["layer"], device=device, dtype=dtype
            )
        assert transcoder.layer_idx not in transcoders, (
            f"Got multiple transcoders for layer {transcoder.layer_idx}"
        )
        transcoders[transcoder.layer_idx] = transcoder

    # we don't know how many layers the model has, but we need all layers from 0 to max covered
    assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
        f"Each layer should have a transcoder, but got transcoders for layers "
        f"{set(transcoders.keys())}"
    )
    feature_input_hook = config["feature_input_hook"]
    feature_output_hook = config["feature_output_hook"]
    return TranscoderSettings(transcoders, feature_input_hook, feature_output_hook, scan)

def get_partial_activation_caching_hooks(
    model: ReplacementModel,
    zero_bos: bool = False,
    sparse: bool = False,
    apply_activation_function: bool = True,
) -> Tuple[List[Optional[torch.Tensor]], List[Tuple[str, Callable]]]:
    """
    Get activation caching hooks for a model partially replaced with transcoders.
    """
    activation_cache: List[Optional[torch.Tensor]] = [None] * len(model.transcoders)

    def cache_activations(acts, hook, layer, zero_bos):
        transcoder_acts = (
            model.transcoders[layer]
            .encode(acts, apply_activation_function=apply_activation_function)
            .detach()
            .squeeze(0)
        )
        if zero_bos:
            transcoder_acts[0] = 0
        activation_cache[layer] = transcoder_acts.to_sparse() if sparse else transcoder_acts

    activation_hooks = [
        (
            f"blocks.{layer}.{model.feature_input_hook}",
            partial(cache_activations, layer=layer, zero_bos=zero_bos),
        )
        for layer in range(len(model.transcoders))
    ]
    return activation_cache, activation_hooks


def configure_partial_replacement(
    model: ReplacementModel,
    transcoders: Dict[int, SingleLayerTranscoder],
    feature_input_hook: str,
    feature_output_hook: str,
    scan: Optional[Union[str, List[str]]],
):
    """
    Function from original ReplacementModel class in circuit_tracer.replacement_model.py, 
    named "_configure_replacement_model", adjusted for partial replacement of the model with transcoders.
    """
    for transcoder in transcoders.values():
        transcoder.to(model.cfg.device, model.cfg.dtype)

    model.add_module(
        "transcoders",
        nn.ModuleList([transcoders[i] for i in range(len(transcoders))]),
    )
    model.d_transcoder = transcoder.d_transcoder
    model.feature_input_hook = feature_input_hook
    model.original_feature_output_hook = feature_output_hook
    model.feature_output_hook = feature_output_hook + ".hook_out_grad"
    model.skip_transcoder = transcoder.W_skip is not None
    model.scan = scan

    for block in model.blocks:
        block.mlp = ReplacementMLP(block.mlp)

    model.unembed = ReplacementUnembed(model.unembed)

    model._configure_gradient_flow()
    model._deduplicate_attention_buffers()
    model.setup()


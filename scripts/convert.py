from os.path import join
from json import load
import warnings

import torch
from sae_lens.sae import SAE, SAEConfig
import sae_lens

from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE

def convert(folder: str):
    # Load configuration
    cfg_path = join(folder, 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = load(f)
        cfg['dtype'] = getattr(torch, cfg['dtype'].split('.')[-1])
    
    # Initialize SAE and load state
    type = cfg["sae_type"]
    if type == "vanilla":
        original = VanillaSAE(cfg)
    elif type == "topk":
        original = TopKSAE(cfg)
    elif type == "batchtopk":
        original = BatchTopKSAE(cfg)
    elif type == "jumprelu":
        original = JumpReLUSAE(cfg)
    elif type == "nnet":
        original = NNetSAE(cfg)
    elif type == "sym":
        original = SymSAE(cfg)
    original.load_state_dict(torch.load(join(folder, 'sae.pt')))
    
    # Configure lens SAE
    cfg_lens = {
        "d_in": cfg["act_size"],
        "d_sae": cfg["dict_size"],
        "device": cfg["device"],
        "dtype": str(cfg["dtype"]),
        "architecture": "topk",
        "hook_name": cfg["hook_point"],
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "normalize_activations": "none",
        "model_name": f'{cfg["model_name"]}-{cfg["sae_type"]}-sae-{cfg["dict_size"]}-{cfg["dataset_path"].split("/")[-1]}',
        "activation_fn_str": "topk",
        "context_size": cfg["seq_len"],
        "hook_layer": cfg["layer"],
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": cfg["dataset_path"],
        "dataset_trust_remote_code": True,
        "activation_fn_kwargs": {"k": cfg["top_k"]},
        "sae_lens_training_version": sae_lens.__version__
    }
    
    sae_config = SAEConfig(**cfg_lens)
    sae = SAE(sae_config).to(cfg_lens["device"])
    
    # Transfer weights from original SAE
    sae.W_enc = original.W_enc
    sae.W_dec = original.W_dec
    sae.b_enc = original.b_enc
    sae.b_dec = original.b_dec

    # If input unit normalization is enabled, set up reshape functions
    if original.cfg.get("input_unit_norm", False):
        warnings.warn("input_unit_norm is not supported by SAE-Lens but returned SAE only supports unified encode and decode.")
        state = {}
        def reshape_fn_in(x):
            x, x_mean, x_std = original.preprocess_input(x)
            state['mean'], state['std'] = x_mean, x_std
            return x
        def reshape_fn_out(x, d_head=None):
            return original.postprocess_output(x, state['mean'], state['std'])
        sae.reshape_fn_in = reshape_fn_in
        sae.reshape_fn_out = reshape_fn_out

    return sae, original

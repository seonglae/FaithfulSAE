'''
Place this file in the BatTopK directory.
This file is used to run the training of the SAEs on the Llama-3.1-8B model.
https://github.com/bartbussmann/BatchTopK
'''
import torch
from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils

for sae_type in ['topk']:
    for top_k in [64]:
      for dataset in ['seonglae/true-synthetic-llama3.1-8b']:
        cfg = get_default_cfg()
        cfg["sae_type"] = "topk"
        cfg["model_name"] = "meta-llama/Llama-3.1-8B"
        cfg["layer"] = 10
        cfg["site"] = "resid_pre"
        cfg["dataset_path"] = dataset
        cfg["aux_penalty"] = (1/32)
        cfg["lr"] = 3e-4
        cfg["dtype"] = torch.bfloat16
        cfg["input_unit_norm"] = False
        cfg["top_k"] = top_k
        cfg["seq_len"] = 512
        cfg["act_size"] = 4096
        cfg["dict_size"] = 4096 * 4
        cfg["num_tokens"] = int(1e7)
        cfg["batch_size"] = 512
        cfg["model_batch_size"] = 16
        cfg['wandb_project'] = 'intrinsic_sae'
        cfg['l1_coeff'] = 0.
        cfg['device'] = 'cuda'
        if cfg["sae_type"] == "vanilla":
            sae = VanillaSAE(cfg)
        elif cfg["sae_type"] == "topk":
            sae = TopKSAE(cfg)
        elif cfg["sae_type"] == "batchtopk":
            sae = BatchTopKSAE(cfg)
        elif cfg["sae_type"] == 'jumprelu':
            sae = JumpReLUSAE(cfg)
        cfg = post_init_cfg(cfg)
 
        model = HookedTransformer.from_pretrained(cfg["model_name"], dtype=cfg["dtype"], device=cfg["device"])
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)

# Replace the following function in the BatchTopK config.py
def post_init_cfg(cfg):
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["name"] = f"{cfg['model_name'].split('/')[-1]}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}_{cfg['seed']}_{cfg['dataset_path'].split('/')[-1]}_{cfg['seq_len']}"
    if cfg["sae_type"] == "nnet":
        cfg["name"] = f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['intermediate_size']}_{cfg['dict_size'] - cfg['intermediate_size']}_{cfg['sae_type']}_{cfg['intermediate_top_k']}_{cfg['top_k']}_{cfg['lr']}"
    if cfg["l1_coeff"] != 0:
        cfg["name"] += f"_{cfg['l1_coeff']}"
    return cfg

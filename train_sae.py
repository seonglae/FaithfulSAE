import torch
import os
import argparse
from glob import glob
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import wandb
from sparsify.sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.sparsify.data import chunk_and_tokenize

from constants import ELEUTHERAI, PYTHIA_1_4B, PYTHIA_2_8B

models_to_expansion_factors = {
    PYTHIA_1_4B: 20,
    PYTHIA_2_8B: 16
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoders on language models")
    parser.add_argument("--org_name", type=str, default=ELEUTHERAI, help="Organization name for model loading")
    parser.add_argument("--target_models", nargs="+", default=[PYTHIA_1_4B, PYTHIA_2_8B], 
                        help="Target models to train SAEs on")
    parser.add_argument("--source_models", nargs="+", default=[PYTHIA_1_4B],
                        help="Source models that generated the datasets")
    parser.add_argument("--dataset_seed", type=int, default=4,
                        help="Seed used when generating the source dataset. We can just remove this in the future by changing the dataset naming convention.")
    parser.add_argument("--dataset_temperature", type=float, default=1.0,
                        help="Temperature used when generating the source dataset (default: 1.0)")
    parser.add_argument("--dataset_top_p", type=float, default=0.9,
                        help="Top-p used when generating the source dataset (default: 0.9)")
    parser.add_argument("--num_seeds", type=int, default=2, 
                        help="Number of seeds for initializing the SAE weights (default: 2)")
    parser.add_argument("--layers", nargs="+", type=int, default=[7, 15], 
                        help="Layers to sparsify in the target models (default: [7, 15])")
    parser.add_argument("--project_name", type=str, default="sae-training", 
                        help="WandB project name (default: sae-training)")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Training batch size (default: 2)")
    return parser.parse_args()


def setup_wandb(project_name: str, run_name: str, config: Dict[str, Any]) -> None:
    if wandb.run is not None:
        wandb.finish()
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True
    )


def load_model(org_name: str, model_name: str) -> tuple:
    config = AutoConfig.from_pretrained(f"{org_name}/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        f"{org_name}/{model_name}",
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
    )
    return model, config


def prepare_dataset(dataset_file: str, org_name: str, model_name: str) -> tuple:
    """Load and tokenize dataset"""
    dataset_path = f"datasets/{dataset_file}"
    print(f"Training on {dataset_path}")
    
    dataset_source_model = dataset_file.split("_")[0]
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(f"{org_name}/{model_name}")
    tokenized = chunk_and_tokenize(dataset, tokenizer)
    
    return tokenized, dataset_source_model


def train_sae(
    model_name: str, 
    dataset_file: str, 
    layers_to_sparsify: List[int], 
    model, 
    tokenized_data, 
    dataset_source_model: str,
    seed: int,
    project_name: str,
    batch_size: int = 2
) -> None:
    """Train the Sparse Autoencoder with the given parameters"""
    run_name = f"{model_name}_from_{dataset_source_model}_seed_{seed}"
    
    setup_wandb(
        project_name=project_name,
        run_name=run_name,
        config={
            "model": model_name,
            "source_model": dataset_source_model,
            "seed": seed,
            "dataset": dataset_file,
            "layers_to_sparsify": layers_to_sparsify,
        }
    )
    
    expansion_factor = models_to_expansion_factors[model_name]
    
    cfg = TrainConfig(
        SaeConfig(expansion_factor=expansion_factor),
        batch_size=batch_size,
        layers=layers_to_sparsify,
        run_name=f"{model_name}_from_{dataset_source_model}_seed_{seed}",
        init_seeds=[seed],
    )
    
    trainer = Trainer(cfg, tokenized_data, model)
    trainer.fit()
    
    # Clean up checkpoint files
    for file in glob(f"checkpoints/*/*.pt"):
        os.remove(file)


def get_dataset_filename(model_name: str, seed: int = 4, temperature: float = 1.0, top_p: float = 0.9) -> str:
    return f"{model_name}_{seed}_{temperature}_{top_p}.jsonl"


def main():
    args = parse_args()
    
    # Construct dataset filenames from source models
    dataset_files = []
    for source_model in args.source_models:
        dataset_file = get_dataset_filename(
            source_model, 
            args.dataset_seed,
            args.dataset_temperature,
            args.dataset_top_p
        )
        dataset_files.append(dataset_file)
    
    # Seeds to use for training
    seeds = list(range(args.num_seeds))
    
    for model_name in args.target_models:
        # Load model and config
        model, config = load_model(args.org_name, model_name)
        layers_to_sparsify = args.layers
        print(f"Targeting layers: {layers_to_sparsify}")
        
        # Initial wandb run for this model
        setup_wandb(
            project_name=args.project_name,
            run_name=f"{model_name}-training",
            config={
                "model": model_name,
                "source_models": args.source_models,
                "datasets": dataset_files,
                "num_seeds": args.num_seeds,
                "layers_to_sparsify": layers_to_sparsify,
            }
        )
        
        for dataset_file in dataset_files:
            tokenized, dataset_source_model = prepare_dataset(dataset_file, args.org_name, model_name)
            
            for seed in seeds:
                train_sae(
                    model_name=model_name,
                    dataset_file=dataset_file,
                    layers_to_sparsify=layers_to_sparsify,
                    model=model,
                    tokenized_data=tokenized,
                    dataset_source_model=dataset_source_model,
                    seed=seed,
                    project_name=args.project_name,
                    batch_size=args.batch_size
                )
    
    # Make sure to close the final wandb run
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
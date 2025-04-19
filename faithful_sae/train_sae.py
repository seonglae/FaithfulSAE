import torch
import os
import argparse
from glob import glob
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import wandb
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

models_to_expansion_factors = {"pythia-1.4b": 20, "pythia-2.8b": 16}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoders on language models")
    parser.add_argument(
        "--org_name", type=str, default="EleutherAI", help="Organization name for model loading"
    )
    parser.add_argument(
        "--target_models",
        nargs="+",
        default=["pythia-1.4b", "pythia-2.8b"],
        help="Target models to train SAEs on",
    )
    parser.add_argument(
        "--source_models",
        nargs="+",
        default=["natural-instruction"],
        help="Source models that generated the datasets",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=4,
        help="Seed used when generating the source dataset. We can just remove this in the future by changing the dataset naming convention.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle Seed (default: 100_000_000)",
    )
    parser.add_argument(
        "--use_shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the dataset or not (default: True)",
    )
    parser.add_argument(
        "--dataset_temperature",
        type=float,
        default=1.0,
        help="Temperature used when generating the source dataset (default: 1.0)",
    )
    parser.add_argument(
        "--dataset_top_p",
        type=float,
        default=0.9,
        help="Top-p used when generating the source dataset (default: 0.9)",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=2,
        help="Number of seeds for initializing the SAE weights (default: 2)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100_000_000,
        help="Number of tokens for training the SAE (default: 100_000_000)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[7, 15],
        help="Layers to sparsify in the target models (default: [7, 15])",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="sae-training",
        help="WandB project name (default: sae-training)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Training batch size (default: 2)"
    )
    return parser.parse_args()


def setup_wandb(project_name: str, run_name: str, config: Dict[str, Any]) -> None:
    if wandb.run is not None:
        wandb.finish()

    wandb.init(project=project_name, name=run_name, config=config, reinit=True)


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
    dataset_path = dataset_file
    print(f"Training on {dataset_path}")

    dataset_source_model = dataset_file.split("_")[0]
    if dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(f"{org_name}/{model_name}")
    if dataset_path.endswith(".jsonl"):
        tokenized = chunk_and_tokenize(dataset, tokenizer)
    else:
        tokenized = chunk_and_tokenize(dataset, tokenizer, text_key="chosen")

    return tokenized, dataset_source_model


def trim_dataset(tokenized_dataset, target_token_size):
    """Trim tokenized dataset upto a target_token_size"""
    dataset_shape = tokenized_dataset["input_ids"].shape
    num_tokes = tokenized_dataset["input_ids"].numel()
    if num_tokes > target_token_size:
        max_idx = (target_token_size // dataset_shape[1]) + 1
        trimmed_dataset = tokenized_dataset.select(range(max_idx))
        return trimmed_dataset
    else:
        return tokenized_dataset


def train_sae(
    model_name: str,
    dataset_file: str,
    layers_to_sparsify: List[int],
    model,
    tokenized_data,
    dataset_source_model: str,
    seed: int,
    project_name: str,
    batch_size: int = 2,
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
        },
    )

    expansion_factor = models_to_expansion_factors[model_name]

    cfg = TrainConfig(
        SaeConfig(expansion_factor=expansion_factor),
        batch_size=batch_size,
        layers=layers_to_sparsify,
        run_name=run_name,
        init_seeds=[seed],
    )

    trainer = Trainer(cfg, tokenized_data, model)
    trainer.fit()

    # Clean up checkpoint files
    for file in glob(f"checkpoints/*/*/*.pt"):
        os.remove(file)


def main():
    args = parse_args()

    dataset_files = ["datasets/better-natural-instructions_dataset.jsonl"]

    # Seeds to use for training
    seeds = list(range(args.num_seeds))

    for model_name in args.target_models:
        # Load model and config
        model, config = load_model(args.org_name, model_name)
        layers_to_sparsify = args.layers
        print(f"Targeting layers: {layers_to_sparsify}")

        # Initial wandb run for this model
        config = {
            "model": model_name,
            "source_models": args.source_models,
            "datasets": dataset_files,
            "num_seeds": args.num_seeds,
            "layers_to_sparsify": layers_to_sparsify,
            "use_suffle": args.use_shuffle,
        }
        if args.use_shuffle:
            config["shuffle_seed"] = args.shuffle_seed
        setup_wandb(
            project_name=args.project_name, run_name=f"{model_name}-training", config=config
        )

        for dataset_file in dataset_files:
            tokenized, dataset_source_model = prepare_dataset(
                dataset_file, args.org_name, model_name
            )

            if args.use_shuffle:
                tokenized = tokenized.shuffle(seed=args.shuffle_seed)

            trimmed_tokenized = trim_dataset(tokenized, args.max_tokens)

            for seed in seeds:
                train_sae(
                    model_name=model_name,
                    dataset_file=dataset_file,
                    layers_to_sparsify=layers_to_sparsify,
                    model=model,
                    tokenized_data=trimmed_tokenized,
                    dataset_source_model=dataset_source_model,
                    seed=seed,
                    project_name=args.project_name,
                    batch_size=args.batch_size,
                )

    # Make sure to close the final wandb run
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

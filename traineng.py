#english GRPO training. Model needs to be updated for desired size. 

import typer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_soft_overlong_punishment

from utils import (
    print_trainable_parameters,
    format_reward_func_qa_any,
    qa_raw_correctness_reward,
    qa_reward_normalized_en,
)
from gsm8k import GSM8K

def train(
    format: str = typer.Option("qa", help="Only QA prompting is supported.", case_sensitive=False),
    num_shots: int = typer.Option(4, help="Number of few-shot examples."),
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID."),
    subset_size: int = typer.Option(1000, help="Subset of GSM8K to train on."),
    num_generations: int = typer.Option(4, help="Generations per GRPO step."),
    lr: float = typer.Option(5e-6, help="Learning rate."),
    epochs: int = typer.Option(1, help="Number of training epochs."),
    use_lora: bool = typer.Option(True, help="True: train LoRA adapters. False: full-model GRPO."),
    short_instructions: bool = typer.Option(False, help="Use shorter instruction block to reduce truncation."),
    reward_mode: str = typer.Option(
        "raw",
        help="raw = (format + correctness + overlong). norm = (normalized reward + overlong).",
        case_sensitive=False,
    ),
):
    import random
    import numpy as np

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)

    typer.echo(f"Starting training with: {model_name}")
    typer.echo(f"Format: {format}, Shots: {num_shots}, Subset: {subset_size}")
    typer.echo(f"use_lora: {use_lora}, short_instructions: {short_instructions}, reward_mode: {reward_mode}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Dataset
    few_shot_flag = (num_shots > 0)
    dataset = (
        GSM8K(
            split="train",
            data_path="gsm8k_golddaen.csv",
            question_field="question",
            answer_field="answer",
            include_answer=False,
            include_reasoning=True,
            few_shot=few_shot_flag,
            num_shots=num_shots,
            seed=42,
            template="qa",
            lang="en",
            tokenizer=tokenizer,
            use_chat_template=True,
            system_prompt=None,
            short_instructions=short_instructions,
        )
        .dataset.shuffle(seed=42)
        .select(range(subset_size))
    )

    finetune_mode = "lora" if use_lora else "full"
    output_dir = (
        f"outputs/split/GRPO/qa/English/{model_name}/"
        f"{finetune_mode}_subset{subset_size}_gen{num_generations}_lr{lr}_{reward_mode}"
    )
    typer.echo(f"Saving model to: {output_dir}")

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f"GRPO-GSM8K-English-qa-{model_name.split('/')[-1]}-{finetune_mode}",
        learning_rate=lr,
        logging_steps=5,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=768,
        max_completion_length=200,  
        num_train_epochs=epochs,
        save_steps=500,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        temperature=0.6,
        top_p=0.9,
    )

    typer.echo("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if use_lora:
        rank = 16
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type="CAUSAL_LM",
            bias="none",
            lora_dropout=0.05,
        )
        typer.echo("Wrapping base model with LoRA...")
        model = get_peft_model(model, peft_config)
    else:
        typer.echo("Training full model (no LoRA).")

    model.config.pad_token_id = tokenizer.pad_token_id
    print_trainable_parameters(model)


    reward_mode = reward_mode.lower()
    if reward_mode == "raw":
        reward_funcs = [format_reward_func_qa_any, qa_raw_correctness_reward]
    elif reward_mode == "norm":
        reward_funcs = [qa_reward_normalized_en]
    else:
        raise ValueError("reward_mode must be 'raw' or 'norm'")

    print("Reward funcs:", [f.__name__ for f in reward_funcs])

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    typer.echo("Starting GRPO training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    typer.echo(f"Saved to: {output_dir}")


if __name__ == "__main__":
    typer.run(train)

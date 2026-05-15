# Code to run English SFT. Models needs to be changed according the desired model size. 
import os
import re
import typer
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

app = typer.Typer()


def extract_final_answer(ans: str) -> str:
    """
    Your 'answer' field is GSM8K-style with a final line like: '#### 72'
    We extract the '72'. Fallback: last number-like token.
    """
    if ans is None:
        return ""
    ans = str(ans)

    m = re.search(r"####\s*([^\n\r]+)", ans)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
    return nums[-1].strip() if nums else ans.strip()


def build_prompt(question: str, allow_reasoning: bool) -> str:
    """
    For TRL's accuracy_reward, boxed answers are the safest format.
    """
    if allow_reasoning:
        return (
            "Solve the question step-by-step.\n"
            "End your response with a final line EXACTLY in this format:\n"
            "\\boxed{<number>}\n"
            "Write only the number inside the box (no text, no units, no extra lines)."
            f"Question: {question}\n"
            "Answer:\n"
        )
    else:
        return (
            f"{question}\n\n"
            "Return ONLY the final answer on a single line EXACTLY in this format:\n"
            "\\boxed{<number>}"
        )


def path_safe_name(x: str) -> str:
    return x.replace("/", "_").replace("\\", "_").replace(":", "_")


@app.command()
def train(
    csv_path: str = typer.Option(
        "gsm8k_goldaen.csv",
        help="Path to CSV file with columns: id,question,answer,da_question,da_answer",
    ),
    model_name: str = typer.Option(
        "outputs/SFT/english_only_reasoning_boxed_qwen2p5_1p5b_lora",
        help="HF model id or local SFT checkpoint path",
    ),
    base_model_name: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        help="Base model id used by the SFT LoRA checkpoint",
    ),
    use_danish: bool = typer.Option(False, help="Use da_question/da_answer instead of question/answer"),
    allow_reasoning: bool = typer.Option(True, help="Allow reasoning text before the boxed final answer"),
    train_rows: int = typer.Option(7473, help="Use the first N rows as training data"),
    subset_size: int = typer.Option(4000, help="Train on first N rows"),
    num_generations: int = typer.Option(8, help="GRPO generations per step"),
    lr: float = typer.Option(3e-6, help="Learning rate"),
    epochs: int = typer.Option(3, help="Epochs"),
    use_lora: bool = typer.Option(
        False,
        help="Set True only when starting from a plain base model. Keep False for SFT LoRA checkpoints.",
    ),
    bf16: bool = typer.Option(True, help="Use bfloat16 (set False if unsupported)"),
):
    model_tag = path_safe_name(os.path.basename(model_name.rstrip("/")) or model_name)
    output_dir = (
        f"outputs/GRPO2/English/{model_tag}/"
        f"epochs{epochs}_subset{subset_size}_gen{num_generations}_lr{lr}_lora{use_lora}"
    )

    q_col = "da_question" if use_danish else "question"
    a_col = "da_answer" if use_danish else "answer"

    typer.echo(f"Loading CSV: {csv_path}")
    typer.echo(f"Using columns: question='{q_col}', answer='{a_col}'")
    typer.echo(f"Model source: {model_name}")
    typer.echo(f"Base model: {base_model_name}")
    typer.echo(f"use_lora={use_lora} | allow_reasoning={allow_reasoning}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("csv", data_files=csv_path, split="train")

    if train_rows is not None:
        ds = ds.select(range(min(train_rows, len(ds))))

    if subset_size and subset_size < len(ds):
        ds = ds.select(range(subset_size))

    typer.echo(f"Using {len(ds)} rows for training.")

    def to_grpo(ex):
        q = ex[q_col]
        gold = extract_final_answer(ex[a_col])
        return {
            "prompt": [{"role": "user", "content": build_prompt(q, allow_reasoning)}],
            "solution": rf"\boxed{{{gold}}}",
        }

    ds = ds.map(to_grpo, remove_columns=ds.column_names)

    typer.echo(f"Processed columns: {ds.column_names}")
    typer.echo(f"Example row: {ds[0]}")

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f"GRPO-RLVR-CSV-{model_tag}",
        learning_rate=lr,
        logging_steps=5,
        per_device_train_batch_size=1,
        generation_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_completion_length=256,
        num_train_epochs=epochs,
        save_steps=500,
        report_to="wandb",
        bf16=bf16,
        fp16=(not bf16),
        temperature=0.6,
        top_p=0.9,
    )

    typer.echo("Loading model...")
    dtype = torch.bfloat16 if bf16 else torch.float16

    is_local_path = os.path.isdir(model_name)
    has_adapter_config = os.path.exists(os.path.join(model_name, "adapter_config.json")) if is_local_path else False

    if has_adapter_config:
        typer.echo("Detected PEFT/LoRA checkpoint. Loading with AutoPeftModelForCausalLM...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            is_trainable=True,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        typer.echo("Detected base model. Loading with AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

        if use_lora:
            rank = 16
            peft_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "up_proj", "down_proj", "gate_proj",
                ],
                task_type="CAUSAL_LM",
                bias="none",
                lora_dropout=0.05,
            )
            typer.echo("Wrapping base model with LoRA...")
            model = get_peft_model(model, peft_config)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=ds,
    )

    typer.echo("Starting GRPO training...")
    trainer.train()

    typer.echo(f"Saving to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    typer.echo("Done.")


if __name__ == "__main__":
    app()
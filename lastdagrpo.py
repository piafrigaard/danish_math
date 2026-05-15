#Danish GRPO code. Models needs to be changed according to which models/adapters are used. 

import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward
from peft import PeftModel
from rewards_da import danish_minus_english_reward


CSV_PATH = "gsm8k_golddaen.csv"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# point this to SFT adapter dir, or set to None
SFT_ADAPTER_PATH = "outputs/SFT/bilingual_reasoning_boxed_qwen2p5_1p5b_lora"
sft_name = "paired"

OUTPUT_ROOT = "outputs/GRPO/final"

ALLOW_REASONING = True

TRAIN_ROWS = 7473
SUBSET_SIZE = 2000

NUM_GENERATIONS = 4
LR = 3e-6
BF16 = True

TEMPERATURE = 0.7
TOP_P = 0.9
MAX_COMPLETION_LENGTH = 128

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GENERATION_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 4

MAX_STEPS = 100000
EPOCHS = 1   # ignored when MAX_STEPS is not None

LOGGING_STEPS = 5
SAVE_STEPS = 500
REPORT_TO = "wandb"

USE_VLLM = False
VLLM_MODE = "colocate"


def extract_final_answer(ans: str) -> str:
    """Extract final GSM8K number (handles comma decimals)."""
    if ans is None:
        return ""
    s = str(ans)

    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        tail = m.group(1).strip()
        m2 = re.search(r"-?\d+(?:[.,]\d+)?", tail)
        if m2:
            return m2.group(0).replace(",", ".").strip()

    nums = re.findall(r"-?\d+(?:[.,]\d+)?", s)
    return nums[-1].replace(",", ".").strip() if nums else s.strip()


def build_prompt_da(question: str, allow_reasoning: bool) -> str:
    if allow_reasoning:
        return (
            "Løs opgaven trin for trin på dansk.\n"
            "Afslut altid med en sidste linje præcis på formatet:\n"
            "\\boxed{<tal>}\n\n"
            f"Spørgsmål: {question}\nSvar:\n"
        )
    else:
        return (
            f"{question}\n\n"
            "Returnér KUN det endelige svar på én linje PRÆCIS i dette format:\n"
            "\\boxed{<tal>}"
        )


def make_run_name() -> str:
    model_tag = MODEL_NAME.split("/")[-1]
    adapter_tag = sft_name if SFT_ADAPTER_PATH else "frombase"
    mode_tag = "reason" if ALLOW_REASONING else "short"
    data_rows = min(TRAIN_ROWS, SUBSET_SIZE) if SUBSET_SIZE else TRAIN_ROWS
    budget_tag = f"steps{MAX_STEPS}" if MAX_STEPS is not None else f"ep{EPOCHS}"
    vllm_tag = "vllm" if USE_VLLM else "novllm"
    return (
        f"grpo_da_{model_tag}_{adapter_tag}_{mode_tag}_"
        f"comp{MAX_COMPLETION_LENGTH}_g{NUM_GENERATIONS}_"
        f"rows{data_rows}_{budget_tag}_{vllm_tag}_lr{LR:.0e}"
    )


def main():
    run_name = make_run_name()
    output_dir = os.path.join(OUTPUT_ROOT, run_name)

    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir}")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    ds = load_dataset("csv", data_files=CSV_PATH, split="train")
    ds = ds.select(range(min(TRAIN_ROWS, len(ds))))
    if SUBSET_SIZE and SUBSET_SIZE < len(ds):
        ds = ds.select(range(SUBSET_SIZE))

    print(f"Using {len(ds)} rows for GRPO training (Danish).")

    def to_grpo(ex):
        q = ex["da_question"]
        gold = extract_final_answer(ex["da_answer"])
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": build_prompt_da(q, ALLOW_REASONING),
                }
            ],
            "solution": rf"\boxed{{{gold}}}",
        }

    ds = ds.map(to_grpo, remove_columns=ds.column_names)
    print(f"Example row: {ds[0]}")

    use_bf16 = BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using dtype: {'bfloat16' if use_bf16 else 'float16'}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere+
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    print(f"Loading base model: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    base.config.pad_token_id = tok.pad_token_id
    if getattr(base, "generation_config", None) is not None:
        base.generation_config.pad_token_id = tok.pad_token_id
        base.generation_config.eos_token_id = tok.eos_token_id
        base.generation_config.bos_token_id = tok.bos_token_id

    if SFT_ADAPTER_PATH:
        print(f"Loading SFT LoRA adapter (trainable) from: {SFT_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(
            base,
            SFT_ADAPTER_PATH,
            is_trainable=True,
        )
    else:
        print("No SFT adapter provided; training from base model.")
        model = base

    model.train()

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    grpo_kwargs = dict(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        report_to=REPORT_TO,

        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=False,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        generation_batch_size=GENERATION_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,

        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        reward_weights=[1.0, 0.2],

        loss_type="dapo",
        mask_truncated_completions=True,

        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE,
    )

    if MAX_STEPS is not None:
        grpo_kwargs["max_steps"] = MAX_STEPS
    else:
        grpo_kwargs["num_train_epochs"] = EPOCHS

    training_args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[accuracy_reward, danish_minus_english_reward],
        args=training_args,
        train_dataset=ds,
    )

    print("Starting Danish GRPO from SFT (accuracy_reward)...")
    trainer.train()

    print(f"Saving to: {output_dir}")
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
# Code to run English SFT. Models needs to be changed according the desired model size. 
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# Settings
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "outputs/english_sft"

MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
SEED = 42

# Prompt formatting
def format_example(example):
    question = example["question"]
    answer = example["answer"]

    # Extract final answer from GSM8K format
    if "####" in answer:
        reasoning, final_answer = answer.split("####")
        final_answer = final_answer.strip()
    else:
        reasoning = answer
        final_answer = ""

    prompt = (
        "Solve the question step by step.\n"
        "End your response with a final line EXACTLY in this format:\n"
        "\\boxed{<number>}\n"
        "Write only the number inside the box (no text, no units, no extra lines).\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = reasoning.strip() + f"\n\\boxed{{{final_answer}}}"

    # Qwen chat template format
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    return {"messages": messages}


def main():
    torch.manual_seed(SEED)

    print("Loading dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].map(format_example)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="wandb",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    print("Starting English SFT...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved English SFT adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

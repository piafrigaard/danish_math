# BiPaired code.

import re
import typer
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

app = typer.Typer()


def extract_gold_number(ans: str) -> str:
    """
    Extract a numeric gold from GSM8K-style answers.
    Prefer '#### <num>' if present, else last number in the string.
    Returns the number as a string (with '.' decimal).
    """
    if ans is None:
        return ""
    s = str(ans)

    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        tail = m.group(1).strip()
        m2 = re.search(r"-?\d+(?:[.,]\d+)?", tail)
        if m2:
            return m2.group(0).replace(",", ".")

    nums = re.findall(r"-?\d+(?:[.,]\d+)?", s)
    if nums:
        return nums[-1].replace(",", ".")
    return s.strip()


def convert_answer_to_reasoning_plus_boxed(full_answer_text: str) -> str:
    """
    Keep the original GSM8K reasoning text, but replace the final '#### ...' with '\boxed{gold}'.
    If no #### line exists, we append a boxed line using extracted gold.
    """
    if full_answer_text is None:
        full_answer_text = ""
    s = str(full_answer_text).strip()

    gold = extract_gold_number(s)
    boxed_line = rf"\boxed{{{gold}}}"

    s_no_hash = re.sub(r"####\s*[^\n\r]+", "", s).strip()

    if s_no_hash:
        return s_no_hash + "\n" + boxed_line
    return boxed_line


def build_system_prompt(lang: str) -> str:
    if lang == "da":
        return "Du er en hjælpsom assistent, der løser matematikopgaver."
    return "You are a helpful assistant that solves math problems."


def build_user_prompt(question: str, lang: str, allow_reasoning: bool) -> str:
    if lang == "da":
        if allow_reasoning:
            return (
                "Løs opgaven trin for trin.\n"
                "Afslut altid med en sidste linje præcis på formatet:\n"
                "\\boxed{<tal>}\n"
                "Skriv kun tallet i boksen (ingen tekst, ingen enheder).\n\n"
                f"Spørgsmål: {question}\n"
                "Svar:\n"
            )
        else:
            return (
                "Returner kun den endelige løsning på én linje præcis på formatet:\n"
                "\\boxed{<tal>}\n"
                "Skriv kun tallet i boksen (ingen tekst, ingen enheder).\n\n"
                f"Spørgsmål: {question}\n"
                "Svar:\n"
            )

    
    if allow_reasoning:
        return (
            "Solve the problem step by step.\n"
            "Always end with a final line EXACTLY in the format:\n"
            "\\boxed{<number>}\n"
            "Write only the number in the box (no text, no units).\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    else:
        return (
            "Return only the final answer on one line EXACTLY in the format:\n"
            "\\boxed{<number>}\n"
            "Write only the number in the box (no text, no units).\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )

@app.command()
def train(
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", help="HF model id"),
    data_path: str = typer.Option("gsm8k_golddaen.csv", help="CSV with question/answer + da_question/da_answer"),
    output_dir: str = typer.Option("outputs/SFT/bilingual_reasoning_boxed_qwen2p5_1p5b_lora", help="Save dir"),
    lr: float = typer.Option(1e-4, help="Learning rate (recommend <=1e-4 for this setup)"),
    epochs: int = typer.Option(1, help="Epochs"),
    batch_size: int = typer.Option(1, help="Per-device batch size"),
    grad_accum: int = typer.Option(8, help="Gradient accumulation"),
    seed: int = typer.Option(42, help="Seed"),
    bf16: bool = typer.Option(True, help="Use bfloat16 (else fp16)"),
    use_lora: bool = typer.Option(True, help="Train LoRA adapters"),
    allow_reasoning: bool = typer.Option(True, help="Include step-by-step instructions in the prompt"),
    train_rows: int = typer.Option(7473, help="Train split size (first N rows)"),
    max_len: int = typer.Option(1024, help="Max token length (truncation)"),
):
    
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    
    dtype = torch.bfloat16 if bf16 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    base.config.pad_token_id = tok.pad_token_id

    model = base
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
        model = get_peft_model(base, peft_config)

    
    ds_all = load_dataset("csv", data_files={"data": data_path})["data"]
    ds_all = ds_all.select(range(0, min(train_rows, len(ds_all)))).shuffle(seed=seed)
    print(f"Train rows (original): {len(ds_all)} (rows [0:{min(train_rows, len(ds_all))}))")

    def expand_bilingual(batch):
        langs, qs, ans = [], [], []
        for q_en, a_en, q_da, a_da in zip(
            batch["question"], batch["answer"], batch["da_question"], batch["da_answer"]
        ):
            # Danish example
            langs.append("da")
            qs.append(q_da)
            ans.append(convert_answer_to_reasoning_plus_boxed(a_da))

            # English example
            langs.append("en")
            qs.append(q_en)
            ans.append(convert_answer_to_reasoning_plus_boxed(a_en))

        return {"lang": langs, "q": qs, "a": ans}

    ds = ds_all.map(expand_bilingual, batched=True, remove_columns=ds_all.column_names)
    print(f"Train rows (expanded bilingual): {len(ds)} (2x)")
    print("Example expanded row:", {k: ds[0][k] for k in ["lang", "q", "a"]})

    def build_text_and_tokenize(batch):
        texts = []
        for lang, q, a in zip(batch["lang"], batch["q"], batch["a"]):
            messages = [
                {"role": "system", "content": build_system_prompt(lang)},
                {"role": "user", "content": build_user_prompt(q, lang, allow_reasoning)},
                {"role": "assistant", "content": a},
            ]
            txt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(txt)

        enc = tok(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds_tok = ds.map(build_text_and_tokenize, batched=True, remove_columns=ds.column_names)
    print(f"Tokenized columns: {ds_tok.column_names}")

    args = SFTConfig(
        output_dir=output_dir,
        run_name="SFT-GSM8K-Bilingual-EN-DA-REASONING-BOXED-LoRA",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=bf16,
        fp16=(not bf16),
        logging_steps=10,
        save_steps=500,
        report_to="wandb",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        processing_class=tok, 
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving to: {output_dir}")
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    app()

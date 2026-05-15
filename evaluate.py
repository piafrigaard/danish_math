# Code to evaluate models. Models needs to be changed according to which models/adapters are used. 
import re
import csv
import json
import math
import os
import torch
from typing import Optional, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from lingua import Language, LanguageDetectorBuilder

CSV_PATH = "gsm8k_golddaen.csv"
BASE_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

MODELS = [
    {
        "name": "32b sft",
        "adapter": "outputs/family_SFT/32B",
    },
]

SPLIT = "test"
TRAIN_ROWS = 7473
SUBSET_SIZE = 0
SEED = 42

BATCH_SIZE = 24
MAX_PROMPT_LEN = 768
MAX_NEW_TOKENS = 512
ALLOW_REASONING = True
BF16 = True

EVAL_MODES = ["da","en"]

OUT_CSV = "eval_results_boxed_bilingual.csv"
OUT_JSON = "eval_results_boxed_bilingual.json"

LANG_DETECT_MIN_CHARS = 15
LANG_DETECT_THRESHOLD = 0.30

_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.DANISH, Language.ENGLISH
).build()


def extract_boxed_number(text: str) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"\\boxed\s*\{\s*([^\}]+)\s*\}", str(text))
    if not m:
        return None
    inner = m.group(1)
    m2 = re.search(r"-?\d+(?:[.,]\d+)?", inner)
    if not m2:
        return None
    return float(m2.group(0).replace(",", "."))


def boxed_compliance(text: str) -> bool:
    return re.search(r"\\boxed\s*\{", str(text)) is not None


def extract_gold_number(ans: str) -> Optional[float]:
    if ans is None:
        return None
    s = str(ans)

    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        m2 = re.search(r"-?\d+(?:[.,]\d+)?", m.group(1))
        if m2:
            return float(m2.group(0).replace(",", "."))

    nums = re.findall(r"-?\d+(?:[.,]\d+)?", s)
    return float(nums[-1].replace(",", ".")) if nums else None


def _clean_for_lang_detect(text: str) -> str:
    s = text or ""

    # Remove boxed answer
    s = re.sub(r"\\boxed\s*\{.*?\}", " ", s, flags=re.DOTALL)

    # Remove other latex commands
    s = re.sub(r"\\[a-zA-Z]+", " ", s)

    # Remove digits
    s = re.sub(r"\d+", " ", s)

    # Keep letters incl. Danish chars and whitespace
    s = re.sub(r"[^A-Za-zÆØÅæøå\s]", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def danish_english_eval_signal(
    text: str,
    exclude_final_answer_line: bool = True,
    min_chars: int = LANG_DETECT_MIN_CHARS,
    threshold: float = LANG_DETECT_THRESHOLD,
) -> Dict[str, object]:
    content = text or ""

    if exclude_final_answer_line:
        lines = content.splitlines()
        if lines:
            last = lines[-1].strip()
            if last.startswith(r"\boxed{") and last.endswith("}"):
                content = "\n".join(lines[:-1])

    cleaned = _clean_for_lang_detect(content)

    if len(cleaned) < min_chars:
        return {
            "score": 0.0,
            "label": "unclear",
            "conf_da": 0.0,
            "conf_en": 0.0,
            "cleaned_len": len(cleaned),
        }

    confidences = _DETECTOR.compute_language_confidence_values(cleaned)

    conf_da = 0.0
    conf_en = 0.0
    for c in confidences:
        if c.language == Language.DANISH:
            conf_da = float(c.value)
        elif c.language == Language.ENGLISH:
            conf_en = float(c.value)

    score = conf_da - conf_en

    if score >= threshold:
        label = "danish"
    elif score <= -threshold:
        label = "english"
    else:
        label = "mixed"

    return {
        "score": score,
        "label": label,
        "conf_da": conf_da,
        "conf_en": conf_en,
        "cleaned_len": len(cleaned),
    }


def system_prompt(lang: str) -> str:
    return (
        "Du er en hjælpsom assistent, der løser matematikopgaver."
        if lang == "da"
        else "You are a helpful assistant that solves math problems."
    )


def user_prompt_en(question: str, allow_reasoning: bool) -> str:
    return (
        "Solve the problem step by step in English.\n"
        "End with a final line EXACTLY in the format:\n"
        "\\boxed{<number>}\n\n"
        f"Question: {question}\nAnswer:\n"
    )


def user_prompt_da(question: str, allow_reasoning: bool) -> str:
    return (
        "Løs opgaven trin for trin på dansk.\n"
        "Afslut altid med en sidste linje præcis på formatet:\n"
        "\\boxed{<tal>}\n\n"
        f"Spørgsmål: {question}\nSvar:\n"
    )


def user_prompt_da_en(da_question: str, en_question: str, allow_reasoning: bool) -> str:
    return (
        "Du får både en dansk opgave du skal løse, samt en engelsk referenceoversættelse.\n"
        "Brug kun den engelske tekst til at afklare betydningen af problemet.\n"
        "Løs opgaven trin for trin på dansk.\n"
        "Afslut altid med en sidste linje præcis på formatet:\n"
        "\\boxed{<tal>}\n"
        "Skriv kun tallet i boksen (ingen tekst, ingen enheder, ingen ekstra linjer).\n\n"
        f"DA: {da_question}\n\n"
        f"EN: {en_question}\n\n"
        "Svar:\n"
    )


def build_chat(tok, mode: str, row: dict) -> str:
    if mode == "en":
        msgs = [
            {"role": "system", "content": system_prompt("en")},
            {"role": "user", "content": user_prompt_en(row["question"], ALLOW_REASONING)},
        ]
    elif mode == "da":
        msgs = [
            {"role": "system", "content": system_prompt("da")},
            {"role": "user", "content": user_prompt_da(row["da_question"], ALLOW_REASONING)},
        ]
    elif mode == "da_en":
        msgs = [
            {"role": "system", "content": system_prompt("da")},
            {
                "role": "user",
                "content": user_prompt_da_en(
                    row["da_question"], row["question"], ALLOW_REASONING
                ),
            },
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _has_adapter_config(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "adapter_config.json"))


def _looks_like_full_model_checkpoint(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False

    if not os.path.isfile(os.path.join(path, "config.json")):
        return False

    filenames = set(os.listdir(path))

    if "model.safetensors" in filenames or "pytorch_model.bin" in filenames:
        return True

    # sharded checkpoints
    if any(
        f.startswith("model-") and f.endswith(".safetensors")
        for f in filenames
    ):
        return True
    if any(
        f.startswith("pytorch_model-") and f.endswith(".bin")
        for f in filenames
    ):
        return True

    return False


def load_model(base_model_id, adapter_path):
    if adapter_path and _has_adapter_config(adapter_path):
        tok_source = base_model_id
    elif adapter_path and _looks_like_full_model_checkpoint(adapter_path):
        tok_source = adapter_path
    else:
        tok_source = base_model_id

    tok = AutoTokenizer.from_pretrained(
        tok_source,
        trust_remote_code=True,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if BF16 else torch.float16

    if adapter_path and _has_adapter_config(adapter_path):
        print(f"Loading PEFT adapter from: {adapter_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        model.config.pad_token_id = tok.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tok.pad_token_id
            model.generation_config.eos_token_id = tok.eos_token_id
            model.generation_config.bos_token_id = tok.bos_token_id

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    elif adapter_path and _looks_like_full_model_checkpoint(adapter_path):
        print(f"Loading full model checkpoint from: {adapter_path}")
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        model.config.pad_token_id = tok.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tok.pad_token_id
            model.generation_config.eos_token_id = tok.eos_token_id
            model.generation_config.bos_token_id = tok.bos_token_id

    elif not adapter_path:
        print(f"Loading base model only: {base_model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        model.config.pad_token_id = tok.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tok.pad_token_id
            model.generation_config.eos_token_id = tok.eos_token_id
            model.generation_config.bos_token_id = tok.bos_token_id

    else:
        raise FileNotFoundError(
            f"Path exists but is neither a PEFT adapter checkpoint nor a full model checkpoint: {adapter_path}"
        )

    model.eval()
    return model, tok


def prepare_mode_data(tok, ds, mode: str):
    if mode == "en":
        answer_col = "answer"
    else:
        answer_col = "da_answer"

    prompts = []
    gold = []

    for row in ds:
        prompts.append(build_chat(tok, mode, row))
        gold.append(extract_gold_number(row[answer_col]))

    return prompts, gold


def uses_danish_language_metrics(mode: str) -> bool:
    return mode in {"da", "da_en"}


@torch.inference_mode()
def evaluate_mode(model, tok, prompts, gold, mode: str) -> Dict[str, float]:
    total = len(prompts)
    correct = 0
    boxed_ok = 0
    num_batches = math.ceil(total / BATCH_SIZE)

    danish_count = 0
    english_count = 0
    mixed_count = 0
    unclear_count = 0
    strict_correct = 0
    correct_but_not_danish = 0
    da_minus_en_sum = 0.0

    for batch_idx in tqdm(range(num_batches), desc=f"{mode.upper()} eval", leave=True):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)

        batch_prompts = prompts[start:end]
        batch_gold = gold[start:end]

        enc = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        ).to(model.device)

        input_len = enc["input_ids"].shape[1]

        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

        gen_token_ids = out[:, input_len:]
        comps = tok.batch_decode(gen_token_ids, skip_special_tokens=True)

        for c, g in zip(comps, batch_gold):
            is_boxed = boxed_compliance(c)
            if is_boxed:
                boxed_ok += 1

            p = extract_boxed_number(c)
            is_correct = p is not None and g is not None and abs(p - g) < 1e-3
            if is_correct:
                correct += 1

            if uses_danish_language_metrics(mode):
                lang_info = danish_english_eval_signal(c)
                label = lang_info["label"]
                score = float(lang_info["score"])
                da_minus_en_sum += score

                if label == "danish":
                    danish_count += 1
                elif label == "english":
                    english_count += 1
                elif label == "mixed":
                    mixed_count += 1
                else:
                    unclear_count += 1

                if is_correct and is_boxed and label == "danish":
                    strict_correct += 1

                if is_correct and label != "danish":
                    correct_but_not_danish += 1

        avg_gen = gen_token_ids.shape[1]
        print(f"[{mode}] avg generated tokens this batch: {avg_gen:.1f}", flush=True)

        if start == 0 and comps:
            print(f"\n--- SAMPLE COMPLETION ({mode}) ---", flush=True)
            print(comps[0][:700], flush=True)
            if uses_danish_language_metrics(mode):
                print("LANG INFO:", danish_english_eval_signal(comps[0]), flush=True)
            print("--- END SAMPLE ---\n", flush=True)

    result = {
        "accuracy": correct / total,
        "boxed_rate": boxed_ok / total,
    }

    if uses_danish_language_metrics(mode):
        result.update({
            "danish_rate": danish_count / total,
            "english_rate": english_count / total,
            "mixed_rate": mixed_count / total,
            "unclear_rate": unclear_count / total,
            "strict_accuracy": strict_correct / total,
            "correct_but_not_danish_rate": correct_but_not_danish / total,
            "avg_da_minus_en": da_minus_en_sum / total,
        })

    return result


def main():
    ds_full = load_dataset("csv", data_files={"data": CSV_PATH})["data"]

    ds = (
        ds_full.select(range(TRAIN_ROWS, len(ds_full)))
        if SPLIT == "test"
        else ds_full.select(range(TRAIN_ROWS))
    )

    if SUBSET_SIZE > 0:
        ds = ds.shuffle(seed=SEED).select(range(SUBSET_SIZE))

    results = []

    warm_tok = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
    )
    if warm_tok.pad_token is None:
        warm_tok.pad_token = warm_tok.eos_token
    warm_tok.padding_side = "left"

    prepared = {
        mode: prepare_mode_data(warm_tok, ds, mode)
        for mode in EVAL_MODES
    }

    for spec in MODELS:
        print(f"\nEvaluating {spec['name']}...")
        model, tok = load_model(BASE_MODEL_ID, spec["adapter"])

        row = {"name": spec["name"]}

        for mode in EVAL_MODES:
            prompts, gold = prepared[mode]
            metrics = evaluate_mode(model, tok, prompts, gold, mode)

            for k, v in metrics.items():
                row[f"{mode}_{k}"] = v

        results.append(row)

        del model
        torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print("\nFINAL RESULTS")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
#Best-of-N and majority vote code. Models needs to be changed according to which models/adapters are used. 

import re
import csv
import json
import math
import os
import random
import gc
import torch
from typing import Optional, Dict, List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from lingua import Language, LanguageDetectorBuilder

import matplotlib.pyplot as plt


CSV_PATH = "gsm8k_golddaen.csv"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MODELS = [
    {
        "name": "Base",
        "adapter": None,
    },
    {
        "name": "English SFT + English GRPO",
        "adapter": "outputs/GRPO/final/grpo_en_Qwen2.5-1.5B-Instruct_EnglishSFT_reason_comp128_g4_rows2000_steps100000_novllm_lr3e-06/checkpoint-26500",
    },
    {
        "name": "Danish SFT only",
        "adapter": "outputs/SFT/danish_only_reasoning_boxed_qwen2p5_1p5b_lora",
    }
]

SPLIT = "test"
TRAIN_ROWS = 7473

SUBSET_SIZE = 0
SEED = 42

BATCH_SIZE = 2
MAX_PROMPT_LEN = 768
MAX_NEW_TOKENS = 512
ALLOW_REASONING = True
BF16 = True

EVAL_MODES = ["en", "da"]

BEST_OF_NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MAX_N = max(BEST_OF_NS)

DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

OUT_CSV = "2neval_results_best_of_and_majority_n.csv"
OUT_JSON = "2neval_results_best_of_and_majority_n.json"
OUT_DIR = "best_of_and_majority_n_outputs_final"
PARTIAL_JSON = "partial_results.json"

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

    s = re.sub(r"\\boxed\s*\{.*?\}", " ", s, flags=re.DOTALL)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^A-Za-zÆØÅæøå\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def danish_english_eval_signal(
    text: str,
    exclude_final_answer_line: bool = True,
    min_chars: int = LANG_DETECT_MIN_CHARS,
    threshold: float = LANG_DETECT_THRESHOLD,
) -> Dict[str, object]:
    if text is None:
        content = ""
    elif isinstance(text, list):
        content = " ".join(str(x) for x in text)
    else:
        content = str(text)

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

    if any(f.startswith("model-") and f.endswith(".safetensors") for f in filenames):
        return True
    if any(f.startswith("pytorch_model-") and f.endswith(".bin") for f in filenames):
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
        print(f"Loading PEFT adapter from: {adapter_path}", flush=True)
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
        print(f"Loading full model checkpoint from: {adapter_path}", flush=True)
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
        print(f"Loading base model only: {base_model_id}", flush=True)
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


def score_completion(text: str, gold: Optional[float], mode: str) -> Dict[str, object]:
    text = "" if text is None else str(text)

    is_boxed = boxed_compliance(text)
    pred = extract_boxed_number(text)
    is_correct = pred is not None and gold is not None and abs(pred - gold) < 1e-3

    out = {
        "text": text,
        "boxed": is_boxed,
        "pred": pred,
        "correct": is_correct,
    }

    if mode == "da":
        lang_info = danish_english_eval_signal(text)
        out["lang_label"] = lang_info["label"]
        out["lang_score"] = float(lang_info["score"])

    return out


def numeric_key(x: Optional[float], decimals: int = 6):
    if x is None:
        return None
    return round(float(x), decimals)


def choose_majority_candidate(prefix: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """
    Select one winning sample from a prefix using majority vote on parsed numeric answers.

    Rules:
    - Ignore samples with no parseable boxed number.
    - Group by parsed numeric answer.
    - Pick the answer with highest vote count.
    - Tie-break 1: lower average index among supporters.
    - Tie-break 2: lower first index.
    - Return the earliest sample supporting the winning answer.
    """
    answer_groups = {}

    for i, x in enumerate(prefix):
        key = numeric_key(x.get("pred"))
        if key is None:
            continue

        if key not in answer_groups:
            answer_groups[key] = {
                "items": [],
                "count": 0,
                "first_idx": i,
                "avg_idx_sum": 0.0,
            }

        answer_groups[key]["items"].append(x)
        answer_groups[key]["count"] += 1
        answer_groups[key]["avg_idx_sum"] += i

    if not answer_groups:
        return None

    ranked = []
    for key, group in answer_groups.items():
        avg_idx = group["avg_idx_sum"] / group["count"]
        ranked.append(
            (
                -group["count"],
                avg_idx,
                group["first_idx"],
                key,
                group,
            )
        )

    ranked.sort()
    winning_group = ranked[0][-1]

    return winning_group["items"][0]


@torch.inference_mode()
def generate_grouped_completions(model, tok, batch_prompts: List[str]) -> List[List[str]]:
    enc = tok(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_LEN,
    ).to(model.device)

    prompt_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_return_sequences=MAX_N,
        use_cache=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    gen_token_ids = out[:, prompt_len:]
    comps = tok.batch_decode(gen_token_ids, skip_special_tokens=True)

    bs = len(batch_prompts)
    grouped = []
    idx = 0
    for _ in range(bs):
        grouped.append(comps[idx: idx + MAX_N])
        idx += MAX_N

    del enc, out, gen_token_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return grouped


@torch.inference_mode()
def evaluate_mode_best_and_majority_of_n(model, tok, prompts, gold, mode: str) -> Dict[str, float]:
    total = len(prompts)
    num_batches = math.ceil(total / BATCH_SIZE)

    best_agg = {
        n: {
            "oracle_accuracy": 0,
            "oracle_boxed_rate": 0,

            # Danish-only, per-problem
            "oracle_strict_accuracy": 0,
            "oracle_correct_but_not_danish_rate": 0,
            "oracle_any_correct_rate": 0,
            "oracle_any_correct_danish_rate": 0,
            "oracle_any_correct_english_rate": 0,

            # Danish-only, per-sample in prefix
            "sample_danish_count": 0,
            "sample_english_count": 0,
            "sample_mixed_count": 0,
            "sample_unclear_count": 0,
            "sample_da_minus_en_sum": 0.0,
        }
        for n in BEST_OF_NS
    }

    maj_agg = {
        n: {
            "majority_accuracy": 0,
            "majority_boxed_rate": 0,
            "majority_parsed_rate": 0,

            # Danish-only
            "majority_strict_accuracy": 0,
            "majority_correct_but_not_danish_rate": 0,
            "majority_selected_danish_rate": 0,
            "majority_selected_english_rate": 0,
            "majority_selected_mixed_rate": 0,
            "majority_selected_unclear_rate": 0,
            "majority_selected_da_minus_en_sum": 0.0,
        }
        for n in BEST_OF_NS
    }

    for batch_idx in tqdm(
        range(num_batches),
        desc=f"{mode.upper()} best-of-n + majority@n eval",
        leave=True,
    ):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)

        batch_prompts = prompts[start:end]
        batch_gold = gold[start:end]

        grouped = generate_grouped_completions(model, tok, batch_prompts)

        for sample_list, g in zip(grouped, batch_gold):
            scored = [score_completion(c, g, mode) for c in sample_list]

            for n in BEST_OF_NS:
                prefix = scored[:n]

                any_correct = any(x["correct"] for x in prefix)
                any_boxed = any(x["boxed"] for x in prefix)

                best_agg[n]["oracle_accuracy"] += int(any_correct)
                best_agg[n]["oracle_boxed_rate"] += int(any_boxed)

                if mode == "da":
                    labels = [x["lang_label"] for x in prefix]

                    best_agg[n]["sample_danish_count"] += sum(lbl == "danish" for lbl in labels)
                    best_agg[n]["sample_english_count"] += sum(lbl == "english" for lbl in labels)
                    best_agg[n]["sample_mixed_count"] += sum(lbl == "mixed" for lbl in labels)
                    best_agg[n]["sample_unclear_count"] += sum(lbl == "unclear" for lbl in labels)
                    best_agg[n]["sample_da_minus_en_sum"] += sum(
                        float(x["lang_score"]) for x in prefix
                    )

                    any_strict = any(
                        x["correct"] and x["boxed"] and x["lang_label"] == "danish"
                        for x in prefix
                    )
                    any_correct_not_danish = any(
                        x["correct"] and x["lang_label"] != "danish"
                        for x in prefix
                    )
                    any_correct_danish = any(
                        x["correct"] and x["lang_label"] == "danish"
                        for x in prefix
                    )
                    any_correct_english = any(
                        x["correct"] and x["lang_label"] == "english"
                        for x in prefix
                    )

                    best_agg[n]["oracle_strict_accuracy"] += int(any_strict)
                    best_agg[n]["oracle_correct_but_not_danish_rate"] += int(
                        any_correct_not_danish
                    )
                    best_agg[n]["oracle_any_correct_rate"] += int(any_correct)
                    best_agg[n]["oracle_any_correct_danish_rate"] += int(any_correct_danish)
                    best_agg[n]["oracle_any_correct_english_rate"] += int(any_correct_english)

                # Majority metrics
                winner = choose_majority_candidate(prefix)

                if winner is not None:
                    maj_agg[n]["majority_parsed_rate"] += 1
                    maj_agg[n]["majority_boxed_rate"] += int(bool(winner["boxed"]))
                    maj_agg[n]["majority_accuracy"] += int(bool(winner["correct"]))

                    if mode == "da":
                        label = winner["lang_label"]

                        maj_agg[n]["majority_selected_danish_rate"] += int(label == "danish")
                        maj_agg[n]["majority_selected_english_rate"] += int(label == "english")
                        maj_agg[n]["majority_selected_mixed_rate"] += int(label == "mixed")
                        maj_agg[n]["majority_selected_unclear_rate"] += int(label == "unclear")
                        maj_agg[n]["majority_selected_da_minus_en_sum"] += float(
                            winner["lang_score"]
                        )

                        is_strict = winner["correct"] and winner["boxed"] and label == "danish"
                        correct_not_danish = winner["correct"] and label != "danish"

                        maj_agg[n]["majority_strict_accuracy"] += int(is_strict)
                        maj_agg[n]["majority_correct_but_not_danish_rate"] += int(
                            correct_not_danish
                        )

        if start == 0 and grouped:
            print(f"\n--- SAMPLE COMPLETIONS ({mode}, shared generations) ---", flush=True)
            for j, s in enumerate(grouped[0][:min(MAX_N, 3)]):
                print(f"[sample {j + 1}]\n{s[:700]}\n", flush=True)

            if mode == "da":
                print(
                    "LANG INFO sample 1:",
                    danish_english_eval_signal(grouped[0][0]),
                    flush=True,
                )

                demo_scored = [score_completion(c, batch_gold[0], mode) for c in grouped[0]]
                for n in BEST_OF_NS[:min(3, len(BEST_OF_NS))]:
                    winner = choose_majority_candidate(demo_scored[:n])
                    if winner is None:
                        print(f"[n={n}] no majority winner", flush=True)
                    else:
                        print(
                            f"[n={n}] winner pred={winner['pred']} "
                            f"correct={winner['correct']} lang={winner.get('lang_label')}",
                            flush=True,
                        )

            print("--- END SAMPLE ---\n", flush=True)

        del grouped
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {}

    for n in BEST_OF_NS:
        # Best-of-n results
        results[f"best_of_{n}_oracle_accuracy"] = best_agg[n]["oracle_accuracy"] / total
        results[f"best_of_{n}_oracle_boxed_rate"] = best_agg[n]["oracle_boxed_rate"] / total

        if mode == "da":
            results[f"best_of_{n}_oracle_strict_accuracy"] = (
                best_agg[n]["oracle_strict_accuracy"] / total
            )
            results[f"best_of_{n}_oracle_correct_but_not_danish_rate"] = (
                best_agg[n]["oracle_correct_but_not_danish_rate"] / total
            )
            results[f"best_of_{n}_oracle_any_correct_rate"] = (
                best_agg[n]["oracle_any_correct_rate"] / total
            )
            results[f"best_of_{n}_oracle_any_correct_danish_rate"] = (
                best_agg[n]["oracle_any_correct_danish_rate"] / total
            )
            results[f"best_of_{n}_oracle_any_correct_english_rate"] = (
                best_agg[n]["oracle_any_correct_english_rate"] / total
            )

            denom = total * n
            results[f"best_of_{n}_sample_danish_rate"] = (
                best_agg[n]["sample_danish_count"] / denom
            )
            results[f"best_of_{n}_sample_english_rate"] = (
                best_agg[n]["sample_english_count"] / denom
            )
            results[f"best_of_{n}_sample_mixed_rate"] = (
                best_agg[n]["sample_mixed_count"] / denom
            )
            results[f"best_of_{n}_sample_unclear_rate"] = (
                best_agg[n]["sample_unclear_count"] / denom
            )
            results[f"best_of_{n}_sample_avg_da_minus_en"] = (
                best_agg[n]["sample_da_minus_en_sum"] / denom
            )

        # Majority results
        results[f"majority_of_{n}_accuracy"] = maj_agg[n]["majority_accuracy"] / total
        results[f"majority_of_{n}_boxed_rate"] = maj_agg[n]["majority_boxed_rate"] / total
        results[f"majority_of_{n}_parsed_rate"] = maj_agg[n]["majority_parsed_rate"] / total

        if mode == "da":
            results[f"majority_of_{n}_strict_accuracy"] = (
                maj_agg[n]["majority_strict_accuracy"] / total
            )
            results[f"majority_of_{n}_correct_but_not_danish_rate"] = (
                maj_agg[n]["majority_correct_but_not_danish_rate"] / total
            )
            results[f"majority_of_{n}_selected_danish_rate"] = (
                maj_agg[n]["majority_selected_danish_rate"] / total
            )
            results[f"majority_of_{n}_selected_english_rate"] = (
                maj_agg[n]["majority_selected_english_rate"] / total
            )
            results[f"majority_of_{n}_selected_mixed_rate"] = (
                maj_agg[n]["majority_selected_mixed_rate"] / total
            )
            results[f"majority_of_{n}_selected_unclear_rate"] = (
                maj_agg[n]["majority_selected_unclear_rate"] / total
            )

            parsed_total = maj_agg[n]["majority_parsed_rate"]
            if parsed_total > 0:
                results[f"majority_of_{n}_selected_avg_da_minus_en"] = (
                    maj_agg[n]["majority_selected_da_minus_en_sum"] / parsed_total
                )
            else:
                results[f"majority_of_{n}_selected_avg_da_minus_en"] = 0.0

    return results


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def pretty_mode_name(mode: str) -> str:
    return {
        "en": "English",
        "da": "Danish",
        "da_en": "Danish + English",
    }.get(mode, mode)


def save_plots(results: List[dict]):
    os.makedirs(OUT_DIR, exist_ok=True)

    for row in results:
        model_name = row["name"]
        safe_model_name = safe_name(model_name)

        # Plot 1: Oracle best-of-n accuracy
        plt.figure(figsize=(9, 5.5))

        for mode in EVAL_MODES:
            xs = []
            ys = []
            for n in BEST_OF_NS:
                key = f"{mode}_best_of_{n}_oracle_accuracy"
                if key in row:
                    xs.append(n)
                    ys.append(100.0 * row[key])

            if xs:
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=pretty_mode_name(mode),
                )

        plt.xlabel("N", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title(f"Best-of-n Accuracy — {model_name}", fontsize=14)
        plt.xticks(BEST_OF_NS, fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Prompt mode", fontsize=11, title_fontsize=11, frameon=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"{safe_model_name}_best_of_accuracy.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 2: Majority@n accuracy
        plt.figure(figsize=(9, 5.5))

        for mode in EVAL_MODES:
            xs = []
            ys = []
            for n in BEST_OF_NS:
                key = f"{mode}_majority_of_{n}_accuracy"
                if key in row:
                    xs.append(n)
                    ys.append(100.0 * row[key])

            if xs:
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=pretty_mode_name(mode),
                )

        plt.xlabel("N", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title(f"Majority vote Accuracy — {model_name}", fontsize=14)
        plt.xticks(BEST_OF_NS, fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Prompt mode", fontsize=11, title_fontsize=11, frameon=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"{safe_model_name}_majority_accuracy.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 3: Best-of-n vs Majority for Danish
        if any(f"da_best_of_{n}_oracle_accuracy" in row for n in BEST_OF_NS):
            plt.figure(figsize=(9, 5.5))

            xs = BEST_OF_NS
            best_vals = [
                100.0 * row.get(f"da_best_of_{n}_oracle_accuracy", float("nan"))
                for n in xs
            ]
            maj_vals = [
                100.0 * row.get(f"da_majority_of_{n}_accuracy", float("nan"))
                for n in xs
            ]
            strict_vals = [
                100.0 * row.get(f"da_majority_of_{n}_strict_accuracy", float("nan"))
                for n in xs
            ]

            plt.plot(xs, best_vals, marker="o", linewidth=2, markersize=6, label="Oracle best-of-n")
            plt.plot(xs, maj_vals, marker="o", linewidth=2, markersize=6, label="Majority@n")
            plt.plot(xs, strict_vals, marker="o", linewidth=2, markersize=6, label="Strict Danish majority@n")

            plt.xlabel("n", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.title(f"Danish Test-Time Scaling — {model_name}", fontsize=14)
            plt.xticks(BEST_OF_NS, fontsize=11)
            plt.yticks(fontsize=11)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11, frameon=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUT_DIR, f"{safe_model_name}_danish_best_vs_majority.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        # Plot 4: Danish detailed best-of-n behavior
        if any(f"da_best_of_{n}_oracle_strict_accuracy" in row for n in BEST_OF_NS):
            plt.figure(figsize=(9, 5.5))

            xs = BEST_OF_NS
            strict_acc = [
                100.0 * row.get(f"da_best_of_{n}_oracle_strict_accuracy", float("nan"))
                for n in xs
            ]
            any_da = [
                100.0 * row.get(f"da_best_of_{n}_oracle_any_correct_danish_rate", float("nan"))
                for n in xs
            ]
            any_en = [
                100.0 * row.get(f"da_best_of_{n}_oracle_any_correct_english_rate", float("nan"))
                for n in xs
            ]

            plt.plot(xs, strict_acc, marker="o", linewidth=2, markersize=6, label="Strict Danish accuracy")
            plt.plot(xs, any_da, marker="o", linewidth=2, markersize=6, label="Any correct Danish")
            plt.plot(xs, any_en, marker="o", linewidth=2, markersize=6, label="Any correct English")

            plt.xlabel("Best-of-n", fontsize=12)
            plt.ylabel("Rate (%)", fontsize=12)
            plt.title(f"Danish Best-of-n Behavior — {model_name}", fontsize=14)
            plt.xticks(BEST_OF_NS, fontsize=11)
            plt.yticks(fontsize=11)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11, frameon=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUT_DIR, f"{safe_model_name}_danish_best_detail.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        # Plot 5: Danish sample language mix under best-of-n
        if any(f"da_best_of_{n}_sample_danish_rate" in row for n in BEST_OF_NS):
            plt.figure(figsize=(9, 5.5))

            xs = BEST_OF_NS
            sample_da = [
                100.0 * row.get(f"da_best_of_{n}_sample_danish_rate", float("nan"))
                for n in xs
            ]
            sample_en = [
                100.0 * row.get(f"da_best_of_{n}_sample_english_rate", float("nan"))
                for n in xs
            ]
            sample_unclear = [
                100.0 * row.get(f"da_best_of_{n}_sample_unclear_rate", float("nan"))
                for n in xs
            ]

            plt.plot(xs, sample_da, marker="o", linewidth=2, markersize=6, label="Sample Danish rate")
            plt.plot(xs, sample_en, marker="o", linewidth=2, markersize=6, label="Sample English rate")
            plt.plot(xs, sample_unclear, marker="o", linewidth=2, markersize=6, label="Sample unclear rate")

            plt.xlabel("Best-of-n", fontsize=12)
            plt.ylabel("Rate (%)", fontsize=12)
            plt.title(f"Danish Sample Language Mix — {model_name}", fontsize=14)
            plt.xticks(BEST_OF_NS, fontsize=11)
            plt.yticks(fontsize=11)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11, frameon=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUT_DIR, f"{safe_model_name}_danish_best_language_mix.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        # Plot 6: Danish selected majority language
        if any(f"da_majority_of_{n}_selected_danish_rate" in row for n in BEST_OF_NS):
            plt.figure(figsize=(9, 5.5))

            xs = BEST_OF_NS
            sel_da = [
                100.0 * row.get(f"da_majority_of_{n}_selected_danish_rate", float("nan"))
                for n in xs
            ]
            sel_en = [
                100.0 * row.get(f"da_majority_of_{n}_selected_english_rate", float("nan"))
                for n in xs
            ]
            sel_unclear = [
                100.0 * row.get(f"da_majority_of_{n}_selected_unclear_rate", float("nan"))
                for n in xs
            ]

            plt.plot(xs, sel_da, marker="o", linewidth=2, markersize=6, label="Selected Danish rate")
            plt.plot(xs, sel_en, marker="o", linewidth=2, markersize=6, label="Selected English rate")
            plt.plot(xs, sel_unclear, marker="o", linewidth=2, markersize=6, label="Selected unclear rate")

            plt.xlabel("Majority", fontsize=12)
            plt.ylabel("Rate (%)", fontsize=12)
            plt.title(f"Danish Majority Selected Language — {model_name}", fontsize=14)
            plt.xticks(BEST_OF_NS, fontsize=11)
            plt.yticks(fontsize=11)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11, frameon=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUT_DIR, f"{safe_model_name}_danish_majority_language.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ds_full = load_dataset("csv", data_files={"data": CSV_PATH})["data"]

    ds = (
        ds_full.select(range(TRAIN_ROWS, len(ds_full)))
        if SPLIT == "test"
        else ds_full.select(range(TRAIN_ROWS))
    )

    if SUBSET_SIZE > 0:
        ds = ds.shuffle(seed=SEED).select(range(SUBSET_SIZE))

    print(f"Using {len(ds)} examples", flush=True)
    print(f"BEST_OF_NS = {BEST_OF_NS}, MAX_N = {MAX_N}", flush=True)
    print(
        f"BATCH_SIZE = {BATCH_SIZE}, effective generation batch = {BATCH_SIZE * MAX_N}",
        flush=True,
    )
    print(
        f"MAX_NEW_TOKENS = {MAX_NEW_TOKENS}, do_sample = {DO_SAMPLE}, "
        f"temperature = {TEMPERATURE}, top_p = {TOP_P}",
        flush=True,
    )

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

    del warm_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for spec in MODELS:
        print(f"\nEvaluating {spec['name']}...", flush=True)
        model, tok = load_model(BASE_MODEL_ID, spec["adapter"])

        row = {"name": spec["name"]}

        for mode in EVAL_MODES:
            print(f"\nStarting mode: {mode}", flush=True)

            prompts, gold = prepared[mode]

            metrics = evaluate_mode_best_and_majority_of_n(
                model=model,
                tok=tok,
                prompts=prompts,
                gold=gold,
                mode=mode,
            )

            for k, v in metrics.items():
                row[f"{mode}_{k}"] = v

            with open(PARTIAL_JSON, "w") as f:
                json.dump(results + [row], f, indent=2)

            print(f"Saved partial results after mode {mode} to: {PARTIAL_JSON}", flush=True)

            del metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(row)

        del model
        del tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    save_plots(results)

    print("\nFINAL RESULTS", flush=True)
    for r in results:
        print(r, flush=True)

    print(f"\nSaved metrics to: {OUT_JSON} and {OUT_CSV}", flush=True)
    print(f"Saved plots to directory: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
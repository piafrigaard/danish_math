import asyncio
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from openai import AsyncOpenAI
from lingua import Language, LanguageDetectorBuilder


# CONFIGURATION
INPUT_PATH  = "xxx"     
OUTPUT_PATH = "xxx" 
BATCH_SIZE  = 10
MODEL       = "xxx"
API_KEY     = "xxxx"
BASE_URL    = "xxx"
MAX_RETRIES = 3


# INITIALIZATION
startTime = datetime.now()
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

detector = LanguageDetectorBuilder.from_languages(
    Language.DANISH, Language.ENGLISH, Language.BOKMAL, Language.NYNORSK
).build()

def is_danish(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        lang = detector.detect_language_of(text)
        return lang == Language.DANISH
    except Exception:
        return False


# TRANSLATION LOGIC
async def async_translate_pair(idx: int, question: str, answer: str, model: str, client) -> dict:
    """
    Translate both question and answer together for term consistency.
    """
    PROMPT = f"""
    You are a world-class English→Danish translator specialized in math word problems.
    Translate both the question and the answer below into Danish in a consistent way:
    - Keep all math notation, numbers, and reasoning structure identical.
    - Use natural, standard Danish phrasing.
    - Do NOT add explanations or comments.
    - Output clearly labeled text sections as:

    Danish Question:
    [your translation of the question]

    Danish Answer:
    [your translation of the answer]

    English Question:
    {question}

    English Answer:
    {answer}
    """

    tries = 0
    while tries < MAX_RETRIES:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
            )
            content = resp.choices[0].message.content.strip()

            # Parse output
            da_q, da_a = parse_danish_sections(content)
            if da_q and da_a and is_danish(da_q) and is_danish(da_a):
                return {
                    "id": idx,
                    "question": question,
                    "answer": answer,
                    "da_question": da_q,
                    "da_answer": da_a
                }
        except Exception:
            pass

        tries += 1
        await asyncio.sleep(1.2 * tries)

    # fallback if it fails
    return {"id": idx, "question": question, "answer": answer, "da_question": "", "da_answer": ""}

def parse_danish_sections(text: str):
    """
    Try to split the model output into Danish question/answer parts.
    Accepts both labeled and unlabeled outputs.
    """
    da_q, da_a = "", ""
    if "Danish Question:" in text and "Danish Answer:" in text:
        parts = text.split("Danish Question:")[1]
        if "Danish Answer:" in parts:
            da_q, rest = parts.split("Danish Answer:", 1)
            da_q = da_q.strip()
            da_a = rest.strip()
    elif "Danish Answer:" in text:
        parts = text.split("Danish Answer:")
        da_q = parts[0].strip()
        da_a = parts[1].strip()
    else:
        lines = text.splitlines()
        mid = len(lines) // 2
        da_q = "\n".join(lines[:mid]).strip()
        da_a = "\n".join(lines[mid:]).strip()
    return da_q, da_a


# ASYNC RUNNER
async def runner(model, client, tasks):
    coroutines = [async_translate_pair(t["id"], t["question"], t["answer"], model, client) for t in tasks]
    return await asyncio.gather(*coroutines)


# MAIN PROCESS
async def main():
    df = pd.read_csv(INPUT_PATH, sep="\t", dtype=str)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    translated = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from saved progress at {OUTPUT_PATH}")
        df_existing = pd.read_csv(OUTPUT_PATH)
        translated = df_existing.to_dict("records")
        start_idx = len(translated)
        print(f"Resumed at row {start_idx}")

    tasks = [{"id": row.id, "question": row.question, "answer": row.answer}
             for row in df.itertuples()]

    for i in tqdm(range(start_idx, len(tasks), BATCH_SIZE), desc="Translating"):
        batch = tasks[i:i+BATCH_SIZE]
        results = await runner(MODEL, client, batch)
        translated.extend(results)

        # periodic checkpoint
        if (i // BATCH_SIZE) % 2 == 0:
            pd.DataFrame(translated).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
            print(f"Progress saved at row {len(translated)}")

    pd.DataFrame(translated).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nDone! Saved to {OUTPUT_PATH}")


# RUN

if __name__ == "__main__":
    asyncio.run(main())
    print("Total runtime:", datetime.now() - startTime)

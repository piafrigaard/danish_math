import asyncio
import os
import re
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from openai import AsyncOpenAI


INPUT_PATH  = "xxx"
OUTPUT_PATH = "xxx"
MODEL       = "xxx"
BASE_URL    = "xxx"
API_KEY     = "xxx"

BATCH_SIZE   = 10
MAX_TOKENS   = 600
MAX_RETRIES  = 3
BACKOFF_BASE = 1.6
TEMP         = 0.0


SYSTEM_INSTRUCTIONS = (
    "You are a strict bilingual math QA reviewer.\n"
    "You will receive English and Danish versions of a math word problem (question + answer for each).\n"
    "Evaluate with these dimensions:\n"
    " 1) en_ok: English reasoning and final numeric result are correct for the EN question.\n"
    " 2) da_ok: Danish reasoning and final numeric result are correct for the DA question.\n"
    " 3) math_ok_en / math_ok_da: math is sound step-by-step in EN/DA answers.\n"
    " 4) crosslang_ok: EN and DA describe the same scenario (names, units, quantities, semantics) and yield the same final numeric result.\n"
    " 5) name_mismatch: True if proper names differ across EN/DA or Q/A (e.g., Natalia vs Julia). If true, do not suggest edits.\n"
    " 6) issues: short structured list of specific problems (target: EN|DA|CROSSLANG|MATH, type, brief, optional fix_suggestion).\n"
    " 7) severity: none|minor|moderate|major|critical (reflects impact on correctness/clarity).\n"
    "Give a concise summary in 'comment' (1–2 sentences). If everything is fine, severity='none' and verdict='OK'.\n"
    "Return STRICT JSON ONLY (no Markdown), exactly this schema:\n"
    "{\n"
    '  \"en_ok\": true/false,\n'
    '  \"da_ok\": true/false,\n'
    '  \"math_ok_en\": true/false,\n'
    '  \"math_ok_da\": true/false,\n'
    '  \"crosslang_ok\": true/false,\n'
    '  \"name_mismatch\": true/false,\n'
    '  \"issues\": [\n'
    '    {\"target\":\"EN|DA|CROSSLANG|MATH\", \"type\":\"short\", \"brief\":\"short\", \"fix_suggestion\":\"optional or empty\"}\n'
    '  ],\n'
    '  \"severity\": \"none|minor|moderate|major|critical\",\n'
    '  \"comment\": \"short\",\n'
    '  \"verdict\": \"OK|ISSUE\"\n'
    "}\n"
)

USER_TEMPLATE = (
    "Evaluate this bilingual math item.\n\n"
    "English Question:\n{q_en}\n\n"
    "English Answer:\n{a_en}\n\n"
    "Danish Question:\n{q_da}\n\n"
    "Danish Answer:\n{a_da}\n"
)

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

def build_prompt(row):
    return USER_TEMPLATE.format(
        q_en=row.get("question",""),
        a_en=row.get("answer",""),
        q_da=row.get("da_question",""),
        a_da=row.get("da_answer",""),
    )

def parse_json_safe(s: str) -> dict:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    try:
        data = json.loads(s)
    except Exception:
        return {}
    # ensure keys
    defaults = {
        "en_ok": None, "da_ok": None, "math_ok_en": None, "math_ok_da": None,
        "crosslang_ok": None, "name_mismatch": False, "issues": [],
        "severity": "", "comment": "", "verdict": "ISSUE"
    }
    for k,v in defaults.items():
        data.setdefault(k, v)
    return data

async def eval_one(row: pd.Series, idx: int) -> dict:
    prompt = build_prompt(row)
    last = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system","content":SYSTEM_INSTRUCTIONS},
                    {"role":"user","content":prompt},
                ],
                temperature=TEMP,
                max_tokens=MAX_TOKENS,
            )
            last = (resp.choices[0].message.content or "").strip()
            data = parse_json_safe(last)
            if data:
                return {"row_idx": idx, "result": data}
        except Exception as e:
            last = f"ERROR: {e}"
        await asyncio.sleep(BACKOFF_BASE ** attempt)
    return {"row_idx": idx, "result": {"verdict":"ISSUE", "comment": f"Model error: {last[:150]}" }}

def is_issue(data: dict) -> bool:
    # Trust the model's explicit verdict first
    v = (data.get("verdict") or "").upper()
    if v == "OK":
        return False
    # Fallback: if any dimension failed, it's an issue
    dims = ["en_ok","da_ok","math_ok_en","math_ok_da","crosslang_ok"]
    for d in dims:
        val = data.get(d)
        if isinstance(val, bool) and val is False:
            return True
    # If severity is not 'none', keep it
    sev = (data.get("severity") or "").lower()
    if sev and sev != "none":
        return True
    # Else default to model verdict
    return True

async def main():
    df = pd.read_csv(INPUT_PATH, dtype=str)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    issues_rows = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="QA reviewing"):
        batch = df.iloc[i:i+BATCH_SIZE]
        coros = [eval_one(row, i+j) for j, row in enumerate(batch.to_dict("records"))]
        results = await asyncio.gather(*coros)

        for res in results:
            idx = res["row_idx"]
            data = res["result"] or {}
            if is_issue(data):
                full_row = df.iloc[idx].to_dict()
                # attach structured fields
                full_row.update({
                    "qa_en_ok": data.get("en_ok"),
                    "qa_da_ok": data.get("da_ok"),
                    "qa_math_ok_en": data.get("math_ok_en"),
                    "qa_math_ok_da": data.get("math_ok_da"),
                    "qa_crosslang_ok": data.get("crosslang_ok"),
                    "qa_name_mismatch": data.get("name_mismatch"),
                    "qa_issues": json.dumps(data.get("issues", []), ensure_ascii=False),
                    "qa_severity": data.get("severity"),
                    "qa_comment": data.get("comment"),
                    "qa_verdict": data.get("verdict"),
                })
                issues_rows.append(full_row)

        # periodic checkpoint
        if (i // BATCH_SIZE) % 3 == 0 and issues_rows:
            pd.DataFrame(issues_rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
            print(f"Checkpoint: {len(issues_rows)} issues saved to {OUTPUT_PATH}")

    pd.DataFrame(issues_rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"{len(issues_rows)} issue rows saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
    print("Finished at", datetime.now())

"""
Fill in null LLM signals in multi_signal_nasdaq_news.csv using a free OpenRouter model.
Reads the existing file, scores only null rows in batches, writes back in place.
"""

import os, sys, time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SIGNAL_COLS = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]

SYSTEM_PROMPT = (
    "You are a quantitative financial analyst. "
    "For each news snippet about a stock output exactly four integer scores on a 1-5 scale, "
    "separated by a pipe '|', in this exact order:\n"
    "  sentiment | risk | confidence | volatility_forecast\n\n"
    "Definitions:\n"
    "  sentiment:            1=very negative, 3=neutral, 5=very positive\n"
    "  risk:                 1=very low risk, 5=very high risk\n"
    "  confidence:           1=very uncertain, 5=very confident\n"
    "  volatility_forecast:  1=very calm, 5=highly volatile\n\n"
    "When multiple news items are given, output one line per item. "
    "Never add explanation — only scores."
)
FEW_SHOT_USER = (
    "Stock: AAPL | News: Apple beats earnings by 20%, raises guidance\n"
    "Stock: AAPL | News: Apple recalls 500k iPhones due to safety issue\n"
    "Stock: MSFT | News: Microsoft acquires gaming studio for $1B"
)
FEW_SHOT_ASSISTANT = "5|1|5|2\n1|5|4|4\n4|2|4|2"


def score_batch(client, model, symbol, texts, max_retries=4, base_delay=3.0):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    n = len(texts)
    if n == 0:
        return []

    prompt = "\n".join(f"Stock: {symbol} | News: {t}" for t in texts)
    conversation = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user",      "content": prompt},
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=conversation,
                temperature=0,
                max_tokens=n * 20,
            )
            content = resp.choices[0].message.content.strip()
            print(f"  [{symbol}] {content[:60]!r}")
        except Exception as exc:
            msg = str(exc)
            if "402" in msg or "credit" in msg.lower():
                print(f"  ERROR 402 credits exhausted — aborting.", file=sys.stderr)
                sys.exit(1)
            if "429" in msg or "rate" in msg.lower():
                wait = base_delay * (2 ** attempt)
                print(f"  Rate-limited, waiting {wait:.0f}s…", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  API error [{symbol}] attempt {attempt}: {msg[:100]}", file=sys.stderr)
            time.sleep(base_delay)
            continue

        results = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            row = []
            for p in parts[:4]:
                try:
                    row.append(int(p.strip()))
                except ValueError:
                    row.append(np.nan)
            while len(row) < 4:
                row.append(np.nan)
            results.append(tuple(row[:4]))

        while len(results) < n:
            results.append((np.nan, np.nan, np.nan, np.nan))
        return results[:n]

    return [(np.nan, np.nan, np.nan, np.nan)] * n


def main():
    csv_path = "multi_signal_nasdaq_news.csv"
    model    = os.environ.get("OPENROUTER_MODEL", "google/gemma-3-12b-it:free")
    batch_sz = 20
    save_every = 400   # rows saved per checkpoint

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title":      os.environ.get("OPENROUTER_APP_NAME", "FinRL-DeepSeek"),
        },
    )

    print(f"Model : {model}")
    print(f"File  : {csv_path}")

    df = pd.read_csv(csv_path)
    null_mask = df["llm_sentiment"].isna()
    null_idx  = df.index[null_mask].tolist()
    print(f"Null rows: {len(null_idx)} / {len(df)}")

    start = time.time()
    scored_since_save = 0

    for start_pos in range(0, len(null_idx), batch_sz):
        batch_idx = null_idx[start_pos : start_pos + batch_sz]
        batch     = df.loc[batch_idx]
        symbol    = batch.iloc[0]["Stock_symbol"]
        texts     = batch["Lsa_summary"].tolist()

        results = score_batch(client, model, symbol, texts)

        for j, (sent, risk, conf, vol) in enumerate(results):
            idx = batch_idx[j]
            df.loc[idx, "llm_sentiment"]           = sent
            df.loc[idx, "llm_risk"]                = risk
            df.loc[idx, "llm_confidence"]          = conf
            df.loc[idx, "llm_volatility_forecast"] = vol

        scored_since_save += len(batch_idx)
        done = start_pos + len(batch_idx)
        elapsed = time.time() - start
        pct = done / len(null_idx) * 100
        eta = (elapsed / done) * (len(null_idx) - done) if done > 0 else 0
        print(f"  Progress: {done}/{len(null_idx)} ({pct:.1f}%) | ETA {eta/60:.1f}min")

        if scored_since_save >= save_every:
            df.to_csv(csv_path, index=False)
            print(f"  Checkpoint saved ({done} rows done)")
            scored_since_save = 0

        time.sleep(1.0)  # free-tier rate limit buffer

    df.to_csv(csv_path, index=False)
    remaining = df["llm_sentiment"].isna().sum()
    print(f"\nDone in {(time.time()-start)/60:.1f}min")
    print(f"Still null: {remaining}")


if __name__ == "__main__":
    main()

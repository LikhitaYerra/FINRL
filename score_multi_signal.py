"""
Multi-signal LLM scorer — powered by OpenRouter.

For each news article this script calls an LLM once and extracts four scores:
  - sentiment           (1=very negative … 5=very positive)
  - risk                (1=very low … 5=very high)
  - confidence          (1=very uncertain … 5=very confident)
  - volatility_forecast (1=very calm … 5=highly volatile)

Output CSV has the same rows as the input plus four new columns:
  llm_sentiment, llm_risk, llm_confidence, llm_volatility_forecast

Usage:
    python score_multi_signal.py \
        --input  nasdaq_news_full.csv \
        --output multi_signal_nasdaq_news.csv

Set OPENROUTER_API_KEY in your .env or environment.
Optionally override --model with any OpenRouter model ID.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env from the project root

# --------------------------------------------------------------------------- #
# Prompt engineering
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a quantitative financial analyst. "
    "For each news snippet about a stock you will output exactly four integer "
    "scores on a 1-5 scale, separated by a pipe '|', in this exact order:\n"
    "  sentiment | risk | confidence | volatility_forecast\n\n"
    "Definitions:\n"
    "  sentiment:            1=very negative, 3=neutral, 5=very positive\n"
    "  risk:                 1=very low company/market risk, 5=very high risk\n"
    "  confidence:           1=very uncertain outlook, 5=very high certainty\n"
    "  volatility_forecast:  1=very calm/stable price expected, 5=highly volatile\n\n"
    "When multiple news items are given for one batch, output one line per item. "
    "Never add explanation — only scores."
)

FEW_SHOT_USER = (
    "Stock: AAPL | News: Apple beats earnings by 20%, raises guidance\n"
    "Stock: AAPL | News: Apple recalls 500k iPhones due to safety issue\n"
    "Stock: MSFT | News: Microsoft acquires gaming studio for $1B"
)
FEW_SHOT_ASSISTANT = (
    "5|1|5|2\n"
    "1|5|4|4\n"
    "4|2|4|2"
)


def _build_batch_text(symbol: str, texts: list[str]) -> str:
    lines = [f"Stock: {symbol} | News: {t}" for t in texts]
    return "\n".join(lines)


def get_multi_signals(
    client: OpenAI,
    model: str,
    symbol: str,
    *texts,
) -> list[tuple]:
    """Return a list of (sentiment, risk, confidence, volatility_forecast) tuples."""
    texts = [t for t in texts if t and t != 0]
    num_text = len(texts)
    if num_text == 0:
        return []

    batch_text = _build_batch_text(symbol, texts)

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": batch_text},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=0,
            max_tokens=num_text * 15,  # e.g. "3|4|3|2\n" ≈ 10 tokens per item
        )
        content = response.choices[0].message.content.strip()
        print(f"[{symbol}] raw response: {content!r}")
    except Exception as exc:
        print(f"API error for {symbol}: {exc}", file=sys.stderr)
        return [(np.nan, np.nan, np.nan, np.nan)] * num_text

    results = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        row = []
        for part in parts[:4]:
            try:
                row.append(int(part.strip()))
            except ValueError:
                row.append(np.nan)
        # Pad to 4 values if model returned fewer
        while len(row) < 4:
            row.append(np.nan)
        results.append(tuple(row[:4]))

    # If model returned wrong number of lines, pad / truncate
    while len(results) < num_text:
        results.append((np.nan, np.nan, np.nan, np.nan))
    return results[:num_text]


# --------------------------------------------------------------------------- #
# CSV processing
# --------------------------------------------------------------------------- #

def process_csv(
    input_csv_path: str,
    output_csv_path: str,
    client: OpenAI,
    model: str,
    batch_size: int = 5,
    chunk_size: int = 100_000,
    retry_delay: float = 2.0,
    max_retries: int = 3,
):
    SIGNAL_COLS = [
        "llm_sentiment",
        "llm_risk",
        "llm_confidence",
        "llm_volatility_forecast",
    ]

    start_time = time.time()

    # Resume support: skip already-written rows
    last_processed_row = 0
    if os.path.exists(output_csv_path):
        existing = pd.read_csv(output_csv_path, on_bad_lines="warn", engine="python")
        last_processed_row = len(existing)
        print(f"Resuming from row {last_processed_row}")

    chunks = pd.read_csv(
        input_csv_path,
        encoding="utf-8",
        chunksize=chunk_size,
        on_bad_lines="warn",
        engine="python",
    )

    for chunk_number, chunk in enumerate(chunks):
        if chunk_number * chunk_size < last_processed_row:
            continue

        chunk.columns = chunk.columns.str.capitalize()

        for col in SIGNAL_COLS:
            if col not in chunk.columns:
                chunk[col] = np.nan

        for i in range(0, len(chunk), batch_size):
            batch = chunk.iloc[i : i + batch_size]
            texts = batch["Lsa_summary"].tolist()
            symbol = batch.iloc[0]["Stock_symbol"]

            for attempt in range(max_retries):
                signals = get_multi_signals(client, model, symbol, *texts)
                if signals:
                    break
                time.sleep(retry_delay * (attempt + 1))

            for j, (sent, risk, conf, vol) in enumerate(signals):
                idx = chunk.index[i + j]
                chunk.loc[idx, "llm_sentiment"] = sent
                chunk.loc[idx, "llm_risk"] = risk
                chunk.loc[idx, "llm_confidence"] = conf
                chunk.loc[idx, "llm_volatility_forecast"] = vol

        write_header = not os.path.exists(output_csv_path)
        chunk.to_csv(output_csv_path, mode="a", header=write_header, index=False)
        print(
            f"Chunk {chunk_number} written "
            f"({time.time() - start_time:.1f}s elapsed)"
        )

    print(f"Done in {time.time() - start_time:.2f}s → {output_csv_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Multi-signal LLM scorer via OpenRouter")
    parser.add_argument("--input",  default="nasdaq_news_full.csv")
    parser.add_argument("--output", default="multi_signal_nasdaq_news.csv")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
    )
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title":      os.environ.get("OPENROUTER_APP_NAME", "FinRL-DeepSeek"),
        },
    )

    print(f"Model : {args.model}")
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")

    process_csv(
        input_csv_path=args.input,
        output_csv_path=args.output,
        client=client,
        model=args.model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

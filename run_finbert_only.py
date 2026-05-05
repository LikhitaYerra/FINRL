#!/usr/bin/env python3
"""
Standalone FinBERT scorer — run with the compatible venv:
  /tmp/finbert_env/bin/python3 run_finbert_only.py

Reads existing dense_text_panel_features.csv (which already has vader + tfidf),
fills in the finbert_sent column using ProsusAI/finbert, and saves the result.
"""

from __future__ import annotations

import argparse
import os
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

NEUTRAL = 3.0  # same sentinel used by LLM pipeline


def load_docs(
    news_path: str, train_csv: str, trade_csv: str
) -> pd.DataFrame:
    news = pd.read_csv(news_path)
    news = news.rename(columns={"Stock_symbol": "tic", "Lsa_summary": "text"})
    news["tic"] = news["tic"].astype(str).str.strip().str.upper()
    news["dt"] = pd.to_datetime(news["Date"]).dt.normalize()
    news["text"] = news["text"].fillna("").astype(str).str.slice(0, 2000)

    tr = pd.read_csv(train_csv)[["date", "tic"]]
    te = pd.read_csv(trade_csv)[["date", "tic"]]
    panel = pd.concat([tr, te], ignore_index=True).drop_duplicates()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["tic"] = panel["tic"].astype(str).str.upper()
    panel = panel.sort_values(["tic", "date"]).reset_index(drop=True)

    lookup: dict[str, tuple[list, list]] = {}
    for tic, g in news.groupby("tic"):
        g = g.sort_values("dt")
        lookup[tic] = (g["dt"].tolist(), g["text"].tolist())

    doc_texts = []
    for _, row in panel.iterrows():
        tic = row["tic"]
        end = pd.Timestamp(row["date"]).normalize()
        start = end - pd.Timedelta(days=3)
        if tic not in lookup:
            doc_texts.append("")
            continue
        dates, texts = lookup[tic]
        lo = bisect_left(dates, start)
        hi = bisect_right(dates, end)
        chunk = texts[lo:hi]
        doc_texts.append(" ".join(chunk) if chunk else "")

    panel["doc_text"] = doc_texts
    return panel


def score_finbert(
    panel: pd.DataFrame, batch_size: int, model_name: str = "ProsusAI/finbert"
) -> np.ndarray:
    print(f"Loading {model_name} (torch {torch.__version__})...")
    tok = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Verify label order: 0=positive, 1=negative, 2=neutral
    id2label = model.config.id2label
    pos_idx = [k for k, v in id2label.items() if "pos" in v.lower()][0]
    neg_idx = [k for k, v in id2label.items() if "neg" in v.lower()][0]
    print(f"Label map: {id2label}  pos={pos_idx} neg={neg_idx}")

    texts = panel["doc_text"].tolist()
    scores = np.zeros(len(texts))
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_idx = i // batch_size
        if batch_idx % 100 == 0:
            pct = 100 * batch_idx / n_batches
            print(f"  FinBERT {pct:.0f}%  ({batch_idx}/{n_batches})", flush=True)

        batch = [str(t) if t and str(t).strip() else "neutral" for t in texts[i : i + batch_size]]
        inputs = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).numpy()
        # score = P(positive) - P(negative) ∈ [-1, +1]
        scores[i : i + len(batch)] = probs[:, pos_idx] - probs[:, neg_idx]

    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="nasdaq_news_full.csv")
    ap.add_argument("--train", default="train_data_multi_signal_2013_2018.csv")
    ap.add_argument("--trade", default="trade_data_multi_signal_2019_2023.csv")
    ap.add_argument("--features-csv", default="backtest_results/dense_text_panel_features.csv")
    ap.add_argument("--out", default="backtest_results/dense_text_panel_features.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    print("Building panel + doc texts...")
    panel = load_docs(args.news, args.train, args.trade)
    print(f"Panel rows: {len(panel):,}")

    scores = score_finbert(panel, args.batch_size)
    panel["finbert_sent"] = scores

    if os.path.exists(args.features_csv):
        existing = pd.read_csv(args.features_csv, parse_dates=["date"])
        existing["tic"] = existing["tic"].astype(str).str.upper()
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        # Update finbert_sent column in existing file
        merge_key = ["date", "tic"]
        existing = existing.drop(columns=["finbert_sent"], errors="ignore")
        merged = existing.merge(
            panel[merge_key + ["finbert_sent"]], on=merge_key, how="left"
        )
        merged.to_csv(args.out, index=False)
        print(f"Updated existing CSV → {args.out}  ({len(merged):,} rows)")
    else:
        cols = ["date", "tic", "finbert_sent"]
        panel[cols].to_csv(args.out, index=False)
        print(f"Saved → {args.out}")

    # Quick summary
    fb = panel["finbert_sent"]
    print(f"\nFinBERT scores: mean={fb.mean():.4f}  std={fb.std():.4f}")
    print(f"  non-zero rows: {(fb != 0).sum():,} / {len(fb):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

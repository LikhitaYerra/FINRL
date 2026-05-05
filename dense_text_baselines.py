#!/usr/bin/env python3
"""
Build dense / lexical text signals aligned with FinRL panel dates.

Produces ``backtest_results/dense_text_panel_features.csv`` with columns (date, tic):
  - finbert_sent (optional --finbert): ProsusAI/finbert tone proxy
  - vader_compound: VADER lexicon sentiment [-1, 1]
  - tfidf_svd_0 .. tfidf_svd_{n-1}: train-fit TF-IDF + truncated SVD on aggregated headlines

Same 3-calendar-day headline aggregation window as the LLM protocol.

  python dense_text_baselines.py
  python dense_text_baselines.py --finbert --batch-size 16

pip install scikit-learn vaderSentiment
optional: transformers sentencepiece
"""

from __future__ import annotations

import argparse
import os
import warnings
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


def load_news(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Stock_symbol": "tic", "Lsa_summary": "text"})
    df["tic"] = df["tic"].astype(str).str.strip().str.upper()
    df["dt"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["text"] = df["text"].fillna("").astype(str).str.slice(0, 2000)
    return df[["tic", "dt", "text"]]


def load_panel(train_csv: str, trade_csv: str) -> pd.DataFrame:
    tr = pd.read_csv(train_csv)[["date", "tic"]]
    te = pd.read_csv(trade_csv)[["date", "tic"]]
    panel = pd.concat([tr, te], ignore_index=True).drop_duplicates()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["tic"] = panel["tic"].astype(str).str.upper()
    return panel.sort_values(["tic", "date"]).reset_index(drop=True)


def concat_docs_for_panel(panel: pd.DataFrame, news: pd.DataFrame) -> pd.Series:
    lookup: dict[str, tuple[list[pd.Timestamp], list[str]]] = {}
    for tic, g in news.groupby("tic"):
        g = g.sort_values("dt")
        lookup[tic] = (g["dt"].tolist(), g["text"].tolist())

    out = []
    for _, row in panel.iterrows():
        tic = row["tic"]
        end = pd.Timestamp(row["date"]).normalize()
        start = end - pd.Timedelta(days=3)
        if tic not in lookup:
            out.append("")
            continue
        dates, texts = lookup[tic]
        lo = bisect_left(dates, start)
        hi = bisect_right(dates, end)
        chunk = texts[lo:hi]
        out.append(" ".join(chunk) if chunk else "")
    return pd.Series(out, index=panel.index)


def vader_series(doc_texts: pd.Series) -> pd.Series:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vad = SentimentIntensityAnalyzer()
    vals = []
    for s in doc_texts:
        if not s or not str(s).strip():
            vals.append(0.0)
        else:
            vals.append(float(vad.polarity_scores(str(s))["compound"]))
    return pd.Series(vals, index=doc_texts.index)


def tfidf_svd_matrix(doc_texts: np.ndarray, train_mask: np.ndarray, n_components: int, max_features: int) -> np.ndarray:
    train_docs = doc_texts[train_mask]
    nonempty_train = np.array([bool(str(d).strip()) for d in train_docs])
    if nonempty_train.sum() < max(50, n_components * 5):
        return np.zeros((len(doc_texts), n_components))

    vec = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=5)
    X_train = vec.fit_transform(train_docs[nonempty_train])
    raw_dim = X_train.shape[1]
    if raw_dim < 2:
        return np.zeros((len(doc_texts), n_components))

    n_comp = min(n_components, raw_dim - 1, max(X_train.shape[0] - 2, 2))
    n_comp = max(n_comp, 2)
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    svd.fit(X_train)

    X_all = vec.transform(doc_texts)
    red = svd.transform(X_all)
    full = np.zeros((len(doc_texts), n_components))
    full[:, : red.shape[1]] = red
    return full


def maybe_finbert(doc_texts: list[str], batch_size: int):
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("Install transformers for --finbert") from exc

    pipe = pipeline(
        task="sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        device=-1,
    )
    out = np.zeros(len(doc_texts))
    for i in range(0, len(doc_texts), batch_size):
        batch = [str(x) if x.strip() else "neutral" for x in doc_texts[i : i + batch_size]]
        preds = pipe(batch)
        for j, p in enumerate(preds):
            idx = i + j
            lab = str(p["label"]).lower()
            sc = float(p["score"])
            if "pos" in lab:
                out[idx] = sc
            elif "neg" in lab:
                out[idx] = -sc
            else:
                out[idx] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="nasdaq_news_full.csv")
    ap.add_argument("--train", default="train_data_multi_signal_2013_2018.csv")
    ap.add_argument("--trade", default="trade_data_multi_signal_2019_2023.csv")
    ap.add_argument("--out", default="backtest_results/dense_text_panel_features.csv")
    ap.add_argument("--tfidf-dim", type=int, default=16)
    ap.add_argument("--tfidf-max-features", type=int, default=4096)
    ap.add_argument("--finbert", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    news = load_news(args.news)
    panel = load_panel(args.train, args.trade)
    print(f"Panel rows: {len(panel):,} | News rows: {len(news):,}")

    panel["doc_text"] = concat_docs_for_panel(panel, news)
    panel["vader_compound"] = vader_series(panel["doc_text"])

    docs = panel["doc_text"].to_numpy()
    train_mask = (panel["date"] < pd.Timestamp("2019-01-01")).to_numpy()
    svd_mat = tfidf_svd_matrix(docs, train_mask, args.tfidf_dim, args.tfidf_max_features)
    for j in range(args.tfidf_dim):
        panel[f"tfidf_svd_{j}"] = svd_mat[:, j]

    if args.finbert:
        print("Running FinBERT (CPU; first run downloads weights)...")
        fb = maybe_finbert(panel["doc_text"].tolist(), args.batch_size)
        panel["finbert_sent"] = fb
    else:
        panel["finbert_sent"] = np.nan

    keep = ["date", "tic", "vader_compound", "finbert_sent"] + [f"tfidf_svd_{j}" for j in range(args.tfidf_dim)]
    panel[keep].to_csv(args.out, index=False)
    print(f"Saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

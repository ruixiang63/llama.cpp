#!/usr/bin/env python3
"""Run MT-Bench prompts against a running llama-server and report
per-category and total tokens/sec, draft acceptance rate, and average
accepted draft tokens per target step.

The server is expected to be already running (llama-server). For
speculative-decoding metrics it must have been started with --spec-type.

Acceptance rate = draft_n_accepted / draft_n  (fraction of drafted
tokens that were verified by the target).

Avg accept length = draft_n_accepted / (predicted_n - draft_n_accepted),
i.e. accepted draft tokens per target forward pass during generation.
"""

import argparse
import json
import logging
import os
import sys
import urllib.request
from collections import defaultdict
from time import time

import requests

MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
DEFAULT_QUESTIONS_PATH = "/tmp/mtbench_questions.jsonl"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("mtbench")


def ensure_questions(path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log.info(f"Downloading MT-Bench questions to {path}")
    urllib.request.urlretrieve(MT_BENCH_URL, path)
    return path


def load_questions(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def select_questions(
    questions: list[dict],
    categories: list[str] | None,
    per_category: int | None,
    max_total: int | None,
) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        if categories and q["category"] not in categories:
            continue
        by_cat[q["category"]].append(q)
    # questions in the file are already grouped/ordered by id within category
    selected: list[dict] = []
    for cat in sorted(by_cat):
        qs = by_cat[cat]
        if per_category is not None:
            qs = qs[:per_category]
        selected.extend(qs)
    if max_total is not None:
        selected = selected[:max_total]
    return selected


def send_chat(url: str, messages: list[dict], max_tokens: int, temperature: float, timeout: float) -> dict:
    r = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def extract_metrics(resp: dict) -> dict:
    """Pull the fields we need out of the server response."""
    timings = resp.get("timings", {}) or {}
    content = resp["choices"][0]["message"]["content"]
    return {
        "predicted_n": int(timings.get("predicted_n", 0)),
        "predicted_ms": float(timings.get("predicted_ms", 0.0)),
        "prompt_n": int(timings.get("prompt_n", 0)),
        "prompt_ms": float(timings.get("prompt_ms", 0.0)),
        "draft_n": int(timings.get("draft_n", 0)),
        "draft_n_accepted": int(timings.get("draft_n_accepted", 0)),
        "content": content,
    }


def aggregate(rows: list[dict]) -> dict:
    pred_n = sum(r["predicted_n"] for r in rows)
    pred_ms = sum(r["predicted_ms"] for r in rows)
    draft_n = sum(r["draft_n"] for r in rows)
    draft_acc = sum(r["draft_n_accepted"] for r in rows)
    target_calls = pred_n - draft_acc  # tokens emitted directly by the target

    return {
        "n_requests": len(rows),
        "predicted_n": pred_n,
        "predicted_ms": pred_ms,
        "tok_per_s": (pred_n / pred_ms * 1000.0) if pred_ms > 0 else 0.0,
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "accept_rate": (draft_acc / draft_n) if draft_n > 0 else 0.0,
        "avg_accept_len": (draft_acc / target_calls) if target_calls > 0 else 0.0,
    }


def fmt_row(label: str, a: dict) -> str:
    return (
        f"{label:<14} | {a['n_requests']:>3d} | "
        f"{a['predicted_n']:>6d} | {a['tok_per_s']:>7.2f} | "
        f"{a['accept_rate']:>6.3f} | {a['avg_accept_len']:>5.2f}"
    )


HEADER = f"{'category':<14} | {'n':>3} | {'toks':>6} | {'t/s':>7} | {'accept':>6} | {'avg_l':>5}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8080", help="llama-server base URL")
    p.add_argument(
        "--questions", default=DEFAULT_QUESTIONS_PATH, help="MT-Bench question.jsonl (auto-downloaded if missing)"
    )
    p.add_argument("--per-category", type=int, default=None, help="Use first N questions from each category")
    p.add_argument(
        "--max-questions", type=int, default=None, help="Cap total number of questions (after per-category filter)"
    )
    p.add_argument("--categories", default=None, help="Comma-separated subset of categories (e.g. coding,math)")
    p.add_argument("--multi-turn", action="store_true", help="Also run turn 2 of each question (chained on turn-1 response)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request HTTP timeout in seconds")
    p.add_argument("--out", default=None, help="Write per-request JSONL to this path")
    p.add_argument("--quiet", action="store_true", help="Don't print per-request progress")
    args = p.parse_args()

    qpath = ensure_questions(args.questions)
    all_q = load_questions(qpath)
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    questions = select_questions(all_q, cats, args.per_category, args.max_questions)
    if not questions:
        log.error("No questions selected.")
        return 2

    log.info(
        f"Server: {args.url}  |  questions: {len(questions)} "
        f"({'all categories' if not cats else ','.join(cats)})  |  "
        f"turns: {'1+2' if args.multi_turn else '1'}  |  "
        f"max_tokens={args.max_tokens}  temp={args.temperature}"
    )

    out_fp = open(args.out, "w") if args.out else None
    rows: list[dict] = []
    t_wall_start = time()

    try:
        for i, q in enumerate(questions, 1):
            qid = q["question_id"]
            cat = q["category"]
            turns = q["turns"]

            messages: list[dict] = []
            for turn_idx in range(2 if args.multi_turn else 1):
                if turn_idx >= len(turns):
                    break
                messages.append({"role": "user", "content": turns[turn_idx]})
                try:
                    resp = send_chat(args.url, messages, args.max_tokens, args.temperature, args.timeout)
                except Exception as e:
                    log.error(f"[{i}/{len(questions)}] qid={qid} turn={turn_idx + 1} request failed: {e}")
                    # don't append next turn
                    break
                m = extract_metrics(resp)
                row = {
                    "qid": qid,
                    "category": cat,
                    "turn": turn_idx + 1,
                    **{k: v for k, v in m.items() if k != "content"},
                }
                rows.append(row)
                if out_fp:
                    out_fp.write(json.dumps({**row, "content": m["content"]}) + "\n")
                    out_fp.flush()
                if not args.quiet:
                    tps = (m["predicted_n"] / m["predicted_ms"] * 1000.0) if m["predicted_ms"] > 0 else 0.0
                    acc = (m["draft_n_accepted"] / m["draft_n"]) if m["draft_n"] > 0 else 0.0
                    log.info(
                        f"[{i}/{len(questions)}] qid={qid} cat={cat} turn={turn_idx + 1}  "
                        f"toks={m['predicted_n']}  t/s={tps:6.2f}  accept={acc:.3f}"
                    )
                # extend conversation for next turn
                messages.append({"role": "assistant", "content": m["content"]})
    finally:
        if out_fp:
            out_fp.close()

    if not rows:
        log.error("No successful requests.")
        return 1

    # group per category
    per_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_cat[r["category"]].append(r)

    t_wall = time() - t_wall_start
    print()
    print(HEADER)
    print("-" * len(HEADER))
    for cat in sorted(per_cat):
        print(fmt_row(cat, aggregate(per_cat[cat])))
    print("-" * len(HEADER))
    total = aggregate(rows)
    print(fmt_row("TOTAL", total))
    print()
    print(
        f"wall: {t_wall:.1f}s  |  total drafted: {total['draft_n']}  "
        f"|  total accepted: {total['draft_n_accepted']}  "
        f"|  target steps during gen: {total['predicted_n'] - total['draft_n_accepted']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# evaluate_retrieval.py

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS

import config
from indexer import load_index


EVAL_FILE = config.DATA_DIR / "eval_questions.json"
RESULT_CSV = config.DATA_DIR / "retrieval_eval_results.csv"


def normalize_lecture_label(label: Optional[str]) -> Optional[str]:
    """
    Normalize various lecture label formats into a unified 'lec-{number}' form
    to make comparison easier.

    Examples:
    - 'Lec 3' / 'lec03' / 'Lecture3' / 'Lec 03' -> 'lec-3'
    - None or empty string -> None
    - If no digits are found, return the lowercase original text.
    """
    if label is None:
        return None

    text = label.strip()
    if not text:
        return None

    import re

    # Extract the first sequence of digits
    m = re.search(r"(\d+)", text)
    if not m:
        # No digits found, fall back to lowercased original
        return text.lower()

    num = int(m.group(1))
    return f"lec-{num}"


def load_eval_questions(path: Path) -> List[Dict[str, Any]]:
    """
    Load evaluation questions from a JSON file.

    Expected format: a list, where each element looks like:
    {
        "id": 2,
        "question": "...",
        "gold_lecture": "Lec 03",
        "type": "formula",
        "note": "..."
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Eval file must contain a JSON list of questions.")

    print(f"[EVAL] Loaded {len(data)} evaluation questions from {path}")
    return data


def evaluate_retrieval(vectorstore: FAISS, eval_data: List[Dict[str, Any]], ks: List[int]):

    #Evaluate retrieval performance on a given question set, computing Recall K.
    max_k = max(ks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": max_k})

    total = 0
    hits_at_k = {k: 0 for k in ks}

    # Collect detailed results for error analysis (written to CSV)
    detailed_rows: List[Dict[str, Any]] = []

    for item in eval_data:
        qid = item.get("id")
        question = item.get("question", "")
        gold_lecture_raw = item.get("gold_lecture")

        gold_norm = normalize_lecture_label(gold_lecture_raw)
        if gold_norm is None:
            # Skip samples without a gold lecture label
            print(f"[EVAL] Skip question id={qid}, missing gold_lecture.")
            continue

        total += 1

        # Run retrieval
        docs = retriever.invoke(question)

        # Collect lecture labels for the top-max_k documents
        predicted_lectures_raw: List[str] = []
        predicted_lectures_norm: List[str] = []

        for d in docs:
            lec_raw = d.metadata.get("lecture", "Unknown")
            lec_raw_str = str(lec_raw)
            predicted_lectures_raw.append(lec_raw_str)
            norm = normalize_lecture_label(lec_raw_str) or "unknown"
            predicted_lectures_norm.append(norm)

        # Compute hit flags for each K
        hits_flags: Dict[int, bool] = {}
        for k in ks:
            top_k_norm = predicted_lectures_norm[:k]
            hit = gold_norm in top_k_norm
            hits_flags[k] = hit
            if hit:
                hits_at_k[k] += 1

        # Record one detailed result row
        row: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "gold_lecture": gold_lecture_raw,
            "gold_lecture_norm": gold_norm,
            "top_lectures_raw": "; ".join(predicted_lectures_raw),
            "top_lectures_norm": "; ".join(predicted_lectures_norm),
        }
        for k in ks:
            row[f"hit@{k}"] = int(hits_flags[k])
        detailed_rows.append(row)

    # Print overall Recall@K
    print("\n[EVAL] Retrieval Results")
    print(f"Total evaluated questions: {total}")
    print("K\tRecall@K")
    for k in ks:
        recall_k = hits_at_k[k] / total if total > 0 else 0.0
        print(f"{k}\t{recall_k:.3f}")

    # Write detailed CSV for further analysis
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "gold_lecture",
        "gold_lecture_norm",
        "top_lectures_raw",
        "top_lectures_norm",
    ] + [f"hit@{k}" for k in ks]

    with open(RESULT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)

    print(f"\n[EVAL] Detailed results saved to: {RESULT_CSV}")


def main():
    # 1. Load the same FAISS index used in your main app
    print(f"[EVAL] Loading FAISS index from {config.INDEX_DIR} / {config.INDEX_NAME} ...")
    vectorstore = load_index()
    print("[EVAL] Index loaded.")

    # 2. Load evaluation question set
    eval_data = load_eval_questions(EVAL_FILE)

    # 3. Define K values for Recall@$K$
    ks = [1, 3, 5, 10]

    # 4. Run evaluation
    evaluate_retrieval(vectorstore, eval_data, ks)


if __name__ == "__main__":
    main()
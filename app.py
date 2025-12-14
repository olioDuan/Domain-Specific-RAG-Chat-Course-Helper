# app.py

import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any
from urllib.parse import quote  
import re
import gradio as gr


import config
from indexer import load_index
from rag_pipeline import rag_answer

import visualize_embeddings
import evaluate_retrieval


# Load index once at startup
vectorstore = load_index()

# Output artifacts (kept consistent with previous scripts)
PCA_IMG_1 = config.DATA_DIR / "lecture_pca.png"
PCA_IMG_2 = config.DATA_DIR / "lecture_centers_pca.png"

EVAL_CSV = config.DATA_DIR / "retrieval_eval_results.csv"
EVAL_PLOT = config.DATA_DIR / "recall_at_k.png"


def respond(message: str, history):
    """
    Callback for ChatInterface, keeping current logic and debug prints.
    """
    if not message or not message.strip():
        return "Please enter a question about the ECE-6143 lectures."

    # Call RAG and get docs (documents) as well as the answer
    answer, docs = rag_answer(message, vectorstore, chat_history=[])

    # ================= Debug code start =================
    print("\n" + "=" * 50)
    print(f"User question: {message}")
    print(f"Retrieved docs: {len(docs)}")

    if len(docs) == 0:
        print("Warning: no documents retrieved. Index may be empty or retrieval is failing.")

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        slide = doc.metadata.get("page", "unknown")
        content_preview = doc.page_content[:100].replace("\n", " ")
        print(f"--- Doc {i+1} [{source} | Slide {slide}] ---")
        print(f"Preview: {content_preview}...")

    print(f"Answer: {answer}")
    print("=" * 50 + "\n")
    # ================= Debug code end =================

    return answer


def _ensure_data_dir():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_pca(force: bool = False):
    """
    Default: load existing PCA plots if they exist.
    force=True: recompute PCA plots.
    """
    _ensure_data_dir()

    if force or (not PCA_IMG_1.exists()) or (not PCA_IMG_2.exists()):
        visualize_embeddings.main()

    status = []
    status.append(f"PCA slide-level: {'OK' if PCA_IMG_1.exists() else 'MISSING'} ({PCA_IMG_1.name})")
    status.append(f"PCA lecture-centers: {'OK' if PCA_IMG_2.exists() else 'MISSING'} ({PCA_IMG_2.name})")
    status_text = "\n".join(status)

    return (
        status_text,
        str(PCA_IMG_1) if PCA_IMG_1.exists() else None,
        str(PCA_IMG_2) if PCA_IMG_2.exists() else None,
    )


def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _compute_recall(headers: List[str], rows: List[List[str]]) -> Dict[int, float]:
    # Find hit@k columns
    hit_cols: Dict[int, int] = {}
    for i, h in enumerate(headers):
        if h.startswith("hit@"):
            try:
                k = int(h.split("@", 1)[1])
                hit_cols[k] = i
            except Exception:
                pass

    recalls: Dict[int, float] = {}
    n = len(rows)
    for k in sorted(hit_cols.keys()):
        col = hit_cols[k]
        if n == 0:
            recalls[k] = 0.0
            continue
        s = 0
        for r in rows:
            try:
                s += int(r[col])
            except Exception:
                pass
        recalls[k] = s / n
    return recalls


def _plot_recall(recalls: Dict[int, float], out_path: Path):
    import matplotlib.pyplot as plt

    ks = sorted(recalls.keys())
    vals = [recalls[k] for k in ks]

    plt.figure(figsize=(5.5, 3.5))
    plt.bar([str(k) for k in ks], vals)
    plt.ylim(0, 1.0)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Retrieval Recall@K")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_eval(force: bool = False):
    """
    Default: load existing CSV if present.
    force=True: re-run evaluate_retrieval.main() to regenerate CSV.

    Note: Recall@K plot is generated only from the CSV and does not trigger retrieval
    (unless force=True and we re-run the evaluation).
    """
    _ensure_data_dir()

    if force or (not EVAL_CSV.exists()):
        evaluate_retrieval.main()

    if not EVAL_CSV.exists():
        return "Evaluation CSV missing.", [], None

    headers, rows = _read_csv(EVAL_CSV)
    recalls = _compute_recall(headers, rows)

    # Only redraw the plot if missing or force=True
    if force or (not EVAL_PLOT.exists()):
        _plot_recall(recalls, EVAL_PLOT)

    summary_lines = [f"Total questions: {len(rows)}"]
    for k in sorted(recalls.keys()):
        summary_lines.append(f"Recall@{k}: {recalls[k]:.3f}")
    summary = "\n".join(summary_lines)

    # For Gradio Dataframe we pass a list-of-lists (first row is header)
    table = [headers] + rows

    return summary, table, str(EVAL_PLOT) if EVAL_PLOT.exists() else None


def build_pca_tab() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("## PCA Visualization")

        status = gr.Markdown("Loading PCA artifacts...")
        img1 = gr.Image(label="Slide-level PCA", type="filepath")
        img2 = gr.Image(label="Lecture centers PCA", type="filepath")

        refresh_btn = gr.Button("Refresh PCA (recompute)")
        refresh_btn.click(fn=lambda: load_pca(force=True), inputs=None, outputs=[status, img1, img2])

        # On load, show existing results without recomputing
        demo.load(fn=lambda: load_pca(force=False), inputs=None, outputs=[status, img1, img2])

    return demo


def build_eval_tab() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("## Retrieval Evaluation")

        summary = gr.Markdown("Loading evaluation artifacts...")
        plot = gr.Image(label="Recall@K", type="filepath")
        table = gr.Dataframe(label="Detailed Results (CSV)", interactive=False, wrap=True)

        refresh_btn = gr.Button("Refresh Evaluation (re-run)")
        refresh_btn.click(fn=lambda: load_eval(force=True), inputs=None, outputs=[summary, table, plot])

        # On load, show existing results without re-running evaluation
        demo.load(fn=lambda: load_eval(force=False), inputs=None, outputs=[summary, table, plot])

    return demo


def main():
    # 1. Build the three UI Blocks
    chat_ui = gr.ChatInterface(
        fn=respond,
        title="6143 RAG Chat Helper",
        description=
            "A domain-specific Retrieval-Augmented Generation (RAG) system built on the official "
            "6143 Machine Learning lecture slides.\n\n"
            "How to use:\n"
            "- Ask concrete questions about the course content, for example:\n"
            "  (1) \"What is backpropagation and where is it used?\"\n"
            "  (2) \"How is PCA applied in this course?\"\n"
            "  (3) \"Explain the difference between training error and test error.\"\n"
            "- Avoid vague inputs like \"test\" or off-topic questions such as \"how to make a pizza\".\n\n"
            "For each valid question, the system retrieves the most relevant slides, generates an "
            "answer with slide-level citations (e.g., (Lec 05, Slide 11)), and produces a short "
            "self-quiz to help you check your understanding.",
        
    )
    
    # PCA and evaluation UIs are also Blocks
    pca_ui = build_pca_tab()
    eval_ui = build_eval_tab()

    # 2. Define CSS (enlarge tab buttons)
    # Override styles for Gradio tab buttons
    custom_css = """
    button.selected, button.tab_button {
        font-size: 20px !important; 
        font-weight: bold !important;
        padding: 10px 20px !important;
    }
    /* Highlight selected state (optional) */
    button.selected {
        color: #2563eb !important;
        border-color: #2563eb !important;
    }
    """
    my_theme = gr.themes.Soft(
        font=["ui-sans-serif", "system-ui", "sans-serif"], # 优先使用系统无衬线字体
    )
    # 3. Use gr.Blocks to manually assemble Tabs
    with gr.Blocks(title="6143 RAG Chat Helper",css=custom_css, theme=my_theme) as demo:
        # Page title
        gr.Markdown("# 🎓 Domain-Specific RAG Chat Course Helper for 6143-ML", elem_classes=["center-text"])
        with gr.Tabs():
            
            with gr.Tab("💬 Chat"):
                # Render the previously defined ChatInterface here
                chat_ui.render()
            
            with gr.Tab("🗺️ PCA Maps"):
                pca_ui.render()
            
            with gr.Tab("📊 Evaluation"):
                eval_ui.render()

    # 4. Launch app
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        favicon_path="assets/ece_rag_icon.png"
    )


if __name__ == "__main__":
    main()
# rag_pipeline.py

from typing import List, Tuple
import re
import unicodedata
import ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config


ChatHistory = List[Tuple[str, str]]  # (user, assistant)


def get_retriever(vectorstore: FAISS):
    """
    Build a retriever from the vector store.
    """
    return vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})


def format_context_with_sources(docs: List[Document]) -> str:
    """
    Format retrieved documents into context blocks with [Source: Lec X, Slide Y].
    """
    blocks: List[str] = []
    for doc in docs:
        lecture = doc.metadata.get("lecture", "Unknown")
        page = doc.metadata.get("page", "?")
        header = f"[Source: {lecture}, Slide {page}]"
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def promote_single_dollar_to_double(text: str) -> str:
    return re.sub(r'(?<!\$)\$(?!\$)', '$$', text)


def call_llm(question: str, context: str, chat_history: ChatHistory) -> str:
    """
    Call the LLM to generate an answer strictly based on the context with citations.
    """
    system_prompt = (


        "You are a teaching assistant for the Machine Learning course at NYU. "
        "You must answer strictly based on the provided lecture context. "
        "If the answer is not in the context, say you cannot answer. "
        "At the end of each sentence, you MUST cite the source in the format (Lec X, Slide Y)."
        "Formatting Rules:\n"
        "1. Use LaTeX formatting for ALL mathematical expressions, variables, and formulas.\n"
        "2. Enclose inline math in double dollar signs (e.g., $$x^2$$).\n"
        "3. Enclose independent formulas in double dollar signs (e.g., $$y = mx + b$$).\n"
        "4. Do NOT use Unicode mathematical characters (like 𝒘, 𝑧); use standard LaTeX syntax instead."
    )
    # For now, only keep the most recent few turns (not yet added to the prompt, but kept for future extension).
    recent_history = chat_history[-config.MAX_HISTORY :]

    user_prompt = (
        "Context information is below. Each context block has a source ID.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in English. Be concise and precise. "
        "For every sentence you write, append a citation in the form (Lec X, Slide Y)."
    )

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            # We currently do not include history in messages; can extend later.
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response["message"]["content"]
    # Strip deepseek-style <think> blocks.
    final_answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    final_answer = promote_single_dollar_to_double(final_answer)
    return final_answer


def generate_self_quiz(question: str, context: str) -> str:
    """
    Generate one multiple-choice question from the same context (non-interactive), including question, options, answer, and explanation with citations.
    """
    if not context.strip():
        return "Self-Quiz:\nQ: Not available (no context retrieved).\nCorrect Answer: N/A\nExplanation: The system could not retrieve relevant slides."

    system_prompt = (
        "You are a strict exam question writer for 6143-Machine Learning. "
        "You must write ONE simple multiple-choice question strictly based on the provided context. "
        "Do NOT use any knowledge outside the context. "
        "Your explanation MUST cite sources like (Lec X, Slide Y). "
        "If the context does not support a good question, say 'Not available'."
        "Formatting Rules:\n"
        "1. Use LaTeX formatting for ALL mathematical expressions, variables, and formulas.\n"
        "2. Enclose inline math in double dollar signs (e.g., $$x^2$$).\n"
        "3. Enclose independent formulas in double dollar signs (e.g., $$y = mx + b$$).\n"
        "4. Do NOT use Unicode mathematical characters (like 𝒘, 𝑧); use standard LaTeX syntax instead."
    )

    user_prompt = (
        "Context information is below. Each context block has a source ID.\n\n"
        f"Context:\n{context}\n\n"
        f"User question (for topic): {question}\n\n"
        "Generate EXACTLY in this format:\n"
        "Self-Quiz:\n"
        "Q: <one question>\n"
        "A) <option A>\n"
        "B) <option B>\n"
        "C) <option C>\n"
        "D) <option D>\n"
        "Correct Answer: <A/B/C/D>\n"
        "Explanation: <2-4 sentences, each sentence ends with a citation (Lec X, Slide Y)>\n"
    )

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response["message"]["content"]
    quiz_text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Fallback: ensure a 'Self-Quiz' header so the UI can easily detect it.
    if "Self-Quiz" not in quiz_text:
        quiz_text = "Self-Quiz:\n" + quiz_text
    return quiz_text

def is_valid_question(question: str) -> bool:
    """
    Heuristic to decide whether a string is a reasonable question.
    Too short / too vague / clearly not about the course returns False.
    Just meant to filter out things like 'test' or 'hello'.
    """
    q = question.strip().lower()
    if len(q) < 4:
        return False

    domain_keywords = {
        "cnn", "rnn", "lstm", "gru", "gan", "vae", "bert", "gpt",
        "svm", "pca", "lda", "knn", "mlp", "ann", "sgd", "adam",
        "backpropagation", "backprop", "softmax", "sigmoid", "relu",
        "convolution", "pooling", "stride", "padding", "kernel",
        "dropout", "normalization", "batchnorm", "regularization",
        "transformer", "attention", "encoder", "decoder",
        
        "gradient", "derivative", "derivation", "formula", "equation",
        "theorem", "proof", "lemma", "definition", "notation",
        "matrix", "vector", "eigenvalue", "eigenvector", "hessian",
        "convex", "concave", "optimization", "objective", "loss",
        "cost", "error", "bias", "variance", "weight", "parameter",
        
        "train", "validation", "overfitting", "underfitting",
        "hyperparameter", "learning", "rate", "epoch", "batch",
        "classifier", "regressor", "clustering", "dimension"
    }
    # Extract English word tokens.
    tokens = re.findall(r"[a-zA-Z]+", q)
    long_tokens = [t for t in tokens if len(t) >= 3]
    if any(t in domain_keywords for t in tokens):
        return True
    # Common question words.
    question_words = {
        "what", "how", "why", "when", "where", "which",
        "describe", "explain", "define", "compare", "contrast",
        "prove", "derive", "show", "compute", "calculate",
    }

    has_q_word = any(t in question_words for t in tokens)
    has_domain_word = any(t in domain_keywords for t in tokens)

    if has_q_word and has_domain_word:
        return True

    # If it has at least two meaningful tokens, also treat as a question (e.g., 'gradient descent step').
    if has_q_word and not has_domain_word:
        if len(long_tokens) >= 5: 
            return True
        else:
            return False # "How to make pizza" (3 tokens) -> False
        
    # Allow questions that contain digits (e.g., formulas / exercise numbers).
    if any(ch.isdigit() for ch in q):
        return True
    if len(long_tokens) >= 3: 
        return True
    return False


def rag_answer(question: str, vectorstore: FAISS, chat_history: ChatHistory) -> Tuple[str, List[Document]]:
    """
    Main RAG pipeline:
    1. First check whether the question looks like a valid course-related question.
    2. Only valid questions trigger retrieval + LLM.
    3. The quiz is generated only when retrieval and answer both look normal.
    """

    # 1) If the question is too short / too vague, reject it and skip retrieval/quiz.
    if not is_valid_question(question):
        base_msg = (
            f'I cannot answer the question "{question}" because it is not a meaningful or specific '
            "question about the ECE-6143 lectures."
        )
        suggest = (
            "Please ask a concrete question about the lecture content, "
            'for example: "What is backpropagation?" or "How is PCA used in this course?".'
        )
        answer = base_msg + " " + suggest

        if getattr(config, "QUIZ_ENABLED", True):
            answer += (
                "\n\nSelf-Quiz:\n"
                "(Not available: the question is not a valid lecture-related question.)"
            )

        # No retrieval performed; return empty docs.
        return answer, []

    # 2) Valid question: perform retrieval and generation.
    retriever = get_retriever(vectorstore)
    retrieved_docs: List[Document] = retriever.invoke(question)
    context = format_context_with_sources(retrieved_docs)

    answer = call_llm(question, context, chat_history)

    # 3) Control quiz generation.
    if getattr(config, "QUIZ_ENABLED", True):
        lower_answer = answer.lower()
        no_retrieval = (not retrieved_docs) or (not context.strip())
        cannot_answer = "cannot answer" in lower_answer

        if no_retrieval or cannot_answer:
            answer += (
                "\n\nSelf-Quiz:\n"
                "(Not available: no relevant slides were retrieved or the question is not covered by the slides.)"
            )
        else:
            quiz = generate_self_quiz(question, context)
            answer = answer + "\n\n" + quiz

    return answer, retrieved_docs
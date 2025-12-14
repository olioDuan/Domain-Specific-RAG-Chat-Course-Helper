# indexer.py

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from embedder import get_embedding_model


def build_index(docs: List[Document]) -> FAISS:
    """
    Build a FAISS index from a list of LangChain Documents.
    """
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def save_index(
    vectorstore: FAISS,
    index_dir: Optional[Path] = None,
    index_name: Optional[str] = None,
) -> None:
    """
    Persist the FAISS index to disk.
    """
    if index_dir is None:
        index_dir = config.INDEX_DIR
    if index_name is None:
        index_name = config.INDEX_NAME

    index_path = Path(index_dir) / index_name
    index_path.parent.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_path))


def load_index(
    index_dir: Optional[Path] = None,
    index_name: Optional[str] = None,
) -> FAISS:
    """
    Load a FAISS index from disk.
    """
    if index_dir is None:
        index_dir = config.INDEX_DIR
    if index_name is None:
        index_name = config.INDEX_NAME

    index_path = Path(index_dir) / index_name

    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
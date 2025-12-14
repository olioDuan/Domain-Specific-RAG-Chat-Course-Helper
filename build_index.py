# build_index.py

from pathlib import Path
from collections import Counter 
from langchain_core.documents import Document

import config
from loader import load_all_slides
from indexer import build_index, save_index


def main():
    print(f"Loading slides from: {config.SLIDES_DIR}")
    docs = load_all_slides(config.SLIDES_DIR)
    
    # 1. Print total number of documents
    total_docs = len(docs)
    print(f"Loaded {total_docs} cleaned slide documents.")

    # ================= Sanity Check =================
    if total_docs == 0:
        print("❌ CRITICAL: Please Check File path or PDF file name.")
        return

    print("\n📊 --- Sanity Check: Slides per Lecture ---")
    
    # Collect lecture IDs from metadata
    lecture_ids = [d.metadata.get("lecture", "Unknown") for d in docs]
    # Count how many slides we have per lecture
    counts = Counter(lecture_ids)

    # Print in sorted order for easier inspection
    for lec_id in sorted(counts.keys()):
        count = counts[lec_id]
        
        if count < 5:
            status = "⚠️  Warning (Too few slides!)" 
        else:
            status = "✅ OK"
            
        print(f"{lec_id:<15}: {count} slides  {status}")
        
    print("--------------------------------------------\n")
    # ===========================================================

    print("Building FAISS index...")
    vectorstore = build_index(docs)

    print(f"Saving index to: {config.INDEX_DIR} / {config.INDEX_NAME}")
    save_index(vectorstore, config.INDEX_DIR, config.INDEX_NAME)

    print("Done.")


if __name__ == "__main__":
    main()
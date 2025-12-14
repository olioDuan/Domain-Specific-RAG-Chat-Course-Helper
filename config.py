# config.py

from pathlib import Path
from langchain_core.documents import Document

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Lecture slides PDF directory
SLIDES_DIR = PROJECT_ROOT / "Slides"

# Index and intermediate data paths
INDEX_DIR = PROJECT_ROOT / "indexes"
INDEX_NAME = "ece6143_faiss"

DATA_DIR = PROJECT_ROOT / "data"

# Model configuration
LLM_MODEL = "deepseek-r1:8b"      
EMBED_MODEL = "all-MiniLM-L6-v2"  

# RAG retrieval parameters
TOP_K = 8

# Number of chat history turns to keep
MAX_HISTORY = 3

QUIZ_ENABLED = True          
QUIZ_NUM_CHOICES = 4         
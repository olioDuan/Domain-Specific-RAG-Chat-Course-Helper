
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

def get_embedding_model():

    print(f"Loading HuggingFace model: {config.EMBED_MODEL}...")
    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
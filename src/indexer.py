from __future__ import annotations
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma as LCChroma

def embed_and_index(
    chunks: List[Dict[str, str]],
    collection_name: str = "ag_miniLM",
    persist_path: str = "data/chroma",
):
    """
    Embed chunk texts with MiniLM, write to a Chroma collection,
    and return a LangChain-wrapped retriever.
    """
    model_name = "all-MiniLM-L6-v2"
    sbert_model = SentenceTransformer(model_name, device="cpu")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    #low level Chroma client (for data write)
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{"parent_id": c["parent_id"]} for c in chunks]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)

    # LangChain wrapper (for retrieval API)
    lc_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    lc_vectorstore = LCChroma(
        client=client,
        collection_name=collection_name,
        embedding_function=lc_embeddings,
    )
    return lc_vectorstore

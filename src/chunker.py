from __future__ import annotations
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(
    docs: List[Dict[str, str]],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[Dict[str, str]]:
    """
    Turn a list of {'doc_id', 'content'} contents into 
    a list of {'id', 'text', 'parent_id'} chunks. 
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks: List[Dict[str, str]] = []
    for d in docs:
        pieces = splitter.split_text(d["content"])
        for i, text in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{d['doc_id']}_c{i:02d}",
                    "text": text,
                    "parent_id": d["doc_id"],
                }
            )
            
    return chunks
# Mini RAG – Generative Benchmark (Work in Progress)

This repo walks through a slimmed-down version of Chroma’s generative-benchmarking loop.
Completed so far:

| Stage | Status | Key artefacts |
|-------|--------|---------------|
| **1 Load data** | done | 500-article AG-News slice → `data/processed/docs.parquet` |
| **2 Chunk / Embed / Index** | done | MiniLM embeddings in a local Chroma DB, wrapped with a LangChain retriever (`data/chroma/ag_miniLM/`)|

Everything lives in `notebooks/01_end_to_end.ipynb`.

---

## Quick start

```bash
git clone https://github.com/<your-user>/generative-benchmark.git
cd generative-benchmark

conda env create -f environment.yml
conda activate genbench

jupyter lab        # open 01_end_to_end.ipynb and run Sections 1-3

The AG-News Parquet snapshot is committed, so data loading runs offline.
The vector store is created locally in data/chroma/ (ignored by Git).
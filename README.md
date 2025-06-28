# Mini RAG – Generative Benchmark (Work in Progress)

This repo walks through a slimmed-down version of Chroma’s generative-benchmarking loop.  
Completed so far:

| Stage                         | Status | Key artefacts                                              |
|-------------------------------|:------:|:-----------------------------------------------------------|
| **1 Load data**               | done   | 500-article AG-News slice → `data/processed/docs.parquet`   |
| **2 Chunk / Embed / Index**   | done   | MiniLM embeddings in a local Chroma DB (`data/chroma/…`)   |
| **3 Synthetic query generation** | done   | 1 question per chunk saved to `data/queries.jsonl` (≈549)  |

All pipeline steps live in `notebooks/01_end_to_end.ipynb`.

---

## Quick start

```bash
git clone https://github.com/<your-user>/generative-benchmark.git
cd generative-benchmark

conda env create -f environment.yml
conda activate genbench

jupyter lab

Offline Data: The AG-News Parquet snapshot (data/processed/docs.parquet) is committed, so Section 1 runs without external downloads.
Local Vector Store: Section 2 builds and persists a Chroma store under data/chroma/ (this directory is git-ignored).
Cached Queries: Section 3 generates and caches synthetic queries in data/queries.jsonl, so you won’t need to regenerate unless you delete that file.
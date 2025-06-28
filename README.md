# Mini RAG – Generative Benchmark (Work in Progress)

This repo walks through a slimmed-down version of Chroma’s generative-benchmarking loop.  
Completed so far:

| Stage                              | Status   | Key artefacts                                                     |
|------------------------------------|:--------:|:------------------------------------------------------------------|
| **1 Load data**                    | done     | 500-article AG-News slice → `data/processed/docs.parquet`          |
| **2 Chunk / Embed / Index**        | done     | MiniLM embeddings in a local Chroma DB (`data/chroma/ag_miniLM/`) |
| **3 Synthetic query generation**   | done     | 1 question per chunk → `data/queries.jsonl` (≈549 lines)          |
| **4 Retrieval evaluation**         | done     | Recall@1,3,5 computed & bar charts → `figures/recall_baseline.png` and `figures/recall_baseline_zoom.png` |

All pipeline steps live in `01_end_to_end.ipynb` (or `notebooks/01_end_to_end.ipynb`).

---

## Quick start

```bash
git clone https://github.com/<your-user>/generative-benchmark.git
cd generative-benchmark

conda env create -f environment.yml
conda activate genbench

jupyter lab           # open the end-to-end notebook and run up through Section 4

Offline Data: data/processed/docs.parquet is committed, so data ingestion runs without downloads.
Local Index: Section 2 builds a Chroma store under data/chroma/ (git-ignored).
Cached Queries: Section 3 writes data/queries.jsonl, so you can skip regeneration if it already exists.
Baseline Results: Section 4 saves two charts in figures/:
recall_baseline.png (standard Recall@{1,3,5})
recall_baseline_zoom.png (zoomed into 0.88–1.00 range)
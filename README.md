````markdown
# Mini RAG – Generative Benchmark

This repo walks through a slimmed-down version of Chroma’s generative-benchmarking loop. We apply the core workflow—ingest data, generate synthetic queries, evaluate retrieval, tweak one component, and measure the impact—on our own documents.

---

## Why AG News & Mini-Benchmark?

- **Dataset choice:** AG News is a clean, well-structured news-article corpus with distinct, self-contained snippets. Its clarity and size (we sample 500 articles) make it ideal for a quick, repeatable proof-of-concept benchmark.
- **Free tooling:** We use open-source models (Sentence-Transformers and Hugging Face) and a local Chroma store—no API costs or gated dependencies.
- **Tweak rationale:** We begin with a general-purpose similarity model (`all-MiniLM-L6-v2`) for baseline retrieval, then swap in a QA-tuned model (`multi-qa-MiniLM-L6-dot-v1`) to see how a targeted embedding objective affects recall.

---

## Completed so far

| Stage                              | Status | Key artefacts                                                       |
|:----------------------------------:|:------:|:--------------------------------------------------------------------|
| **1 Load data**                    | done   | 500-article AG-News slice → `data/processed/docs.parquet`            |
| **2 Chunk / Embed / Index**        | done   | MiniLM embeddings in a local Chroma DB (`data/chroma/ag_miniLM/`)   |
| **3 Synthetic query generation**   | done   | 1 question per chunk → `data/queries.jsonl` (≈ 549 lines)           |
| **4 Retrieval evaluation**         | done   | Recall@1,3,5 & bar charts → `figures/recall_baseline*.png`          |
| **5 Embedding-Model Swap**         | done   | Retrained with `multi-qa-MiniLM-L6-dot-v1` → `figures/recall_comparison_embedding.png` |
| **6 Matrix benchmark**             | done   | Recall@1,3,5 over 6 embed/chunk combos → `figures/recall_matrix_*_*.png` |

All pipeline steps live in `01_end_to_end.ipynb`.

---

## Quick start

```bash
git clone https://github.com/<your-user>/generative-benchmark.git
cd generative-benchmark

conda env create -f environment.yml
conda activate genbench

jupyter lab           # open 01_end_to_end.ipynb and run through Section 6
````

* **Offline Data:** `data/processed/docs.parquet` is committed—no downloads in Section 1.
* **Local Index:** Section 2 writes to `data/chroma/` (git-ignored).
* **Cached Queries:** Section 3 writes `data/queries.jsonl`—skip regeneration if present.
* **Baseline Results:** Section 4 saves two charts in `figures/`.
* **Comparison Results:** Section 5 saves `figures/recall_comparison_embedding.png`.
* **Matrix Results:** Section 6 saves six heatmaps & bar charts in `figures/recall_matrix_k{1,3,5}_{heatmap,barchart}.png`.

---

## Results

### Baseline Recall\@k (zoomed)

![Baseline Recall zoomed](figures/recall_baseline_zoom.png)

|  k  | Recall |
| :-: | :----: |
|  1  |  0.920 |
|  3  |  0.985 |
|  5  |  0.993 |

### After swap to multi-qa-MiniLM-L6-dot-v1

![Recall Comparison](figures/recall_comparison_embedding.png)

|  k  | all-MiniLM-L6-v2 | multi-qa-MiniLM |
| :-: | :--------------: | :-------------: |
|  1  |       0.920      |      0.699      |
|  3  |       0.985      |      0.801      |
|  5  |       0.993      |      0.825      |

### Matrix Benchmark: Embedding × Chunking

|             Model             | chunk/overlap | Recall\@1 | Recall\@3 | Recall\@5 |
| :---------------------------: | :-----------: | :-------: | :-------: | :-------: |
|      **all-MiniLM-L6-v2**     |     400/50    |   0.920   |   0.985   |   0.993   |
|      **all-MiniLM-L6-v2**     |    200/100    |   0.922   |   0.976   |   0.991   |
|      **all-MiniLM-L6-v2**     |     100/25    |   0.916   |   0.974   |   0.982   |
| **multi-qa-MiniLM-L6-dot-v1** |     400/50    |   0.701   |   0.805   |   0.829   |
| **multi-qa-MiniLM-L6-dot-v1** |    200/100    |   0.820   |   0.882   |   0.894   |
| **multi-qa-MiniLM-L6-dot-v1** |     100/25    |   0.869   |   0.929   |   0.938   |

#### Recall\@3 Heatmap

![Heatmap Recall@3](figures/recall_matrix_k3_heatmap.png)

#### Recall\@3 Grouped Bar Chart

![Recall@3 by Embedding & Chunking](figures/recall_matrix_k3_barchart.png)

---

## Analysis & Takeaways

* **Embedding–chunk trade-off (baseline):** For `all-MiniLM-L6-v2`, shrinking chunks (and adding overlap) fragments context, slightly reducing recall across k.
* **Embedding–chunk synergy (QA-tuned):** `multi-qa-MiniLM-L6-dot-v1` improves with finer chunks—its question-focused embeddings shine on shorter, more targeted passages.
* **Practical impact:** More, smaller chunks increase storage/index cost; your best chunk size depends on the embedding objective.
* **Next steps:** Introduce a lightweight cross-encoder reranker atop the top-k hits to recover recall for general models, or explore hybrid chunking strategies.

---

```
```

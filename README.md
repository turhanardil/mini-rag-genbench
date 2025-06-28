````markdown
# Mini RAG – Generative Benchmark (Work in Progress)

This repo walks through a slimmed-down version of Chroma’s generative-benchmarking loop. We apply the core workflow—ingest data, generate synthetic queries, evaluate retrieval, tweak one component, and measure the impact—on our own documents.

---

## Why AG News & Mini-Benchmark?

- **Dataset choice:** AG News is a clean, well-structured news-article corpus with distinct, self-contained snippets. Its clarity and size (we sample 500 articles) make it ideal for a quick, repeatable proof-of-concept benchmark. 
- **Free tooling:** We use open-source models (Sentence-Transformers and HuggingFace) and a local Chroma store, so there are no API costs or gated dependencies.
- **Tweak rationale:** We begin with a general-purpose similarity model (`all-MiniLM-L6-v2`) for baseline retrieval, then swap in a QA-tuned model (`multi-qa-MiniLM-L6-dot-v1`) to see how a targeted embedding objective affects recall.

---

## Completed so far

| Stage                              | Status   | Key artefacts                                                     |
|------------------------------------|:--------:|:------------------------------------------------------------------|
| **1 Load data**                    |    done  | 500-article AG-News slice → `data/processed/docs.parquet`          |
| **2 Chunk / Embed / Index**        |    done  | MiniLM embeddings in a local Chroma DB (`data/chroma/ag_miniLM/`) |
| **3 Synthetic query generation**   |    done  | 1 question per chunk → `data/queries.jsonl` (≈ 549 lines)         |
| **4 Retrieval evaluation**         |    done  | Recall@1,3,5 & bar charts → `figures/recall_baseline*.png`        |
| **5 Embedding-Model Swap**         |    done  | Retrained with `multi-qa-MiniLM-L6-dot-v1` → `figures/recall_comparison.png` |

All pipeline steps live in `notebooks/01_end_to_end.ipynb`.

---

## Quick start

```bash
git clone https://github.com/<your-user>/generative-benchmark.git
cd generative-benchmark

conda env create -f environment.yml
conda activate genbench

jupyter lab           # open 01_end_to_end.ipynb and run through Section 5
````

* **Offline Data:** `data/processed/docs.parquet` is committed—no downloads in Section 1.
* **Local Index:** Section 2 writes to `data/chroma/` (git-ignored).
* **Cached Queries:** Section 3 writes `data/queries.jsonl`—skip regeneration if present.
* **Baseline Results:** Section 4 saves two charts in `figures/`.
* **Comparison Results:** Section 5 saves `figures/recall_comparison_embedding.png`.

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

---

## Analysis & Takeaways

* **Embedding-model effect:** Swapping from a general similarity model to a QA-tuned model dropped Recall\@1 by \~22 points (0.920 → 0.699). Even at k=5 the QA model peaks at \~0.825 vs. \~0.993 baseline.
* **Expected vs. observed:** We anticipated some degradation—`multi-qa-MiniLM` is optimized for ranking passages given real user questions, not synthetic “quiz” prompts derived directly from passages. The sharper drop confirms that embedding objectives must align with your retrieval task.
* **Next steps:** To regain recall we could tweak chunk size/overlap or add a lightweight reranker (e.g., a cross-encoder on top-k candidates). This would form a small matrix benchmark of embedding + chunking + reranking choices.

---


```
```

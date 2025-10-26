# FinGenie RAG Evaluation Framework

This repository provides a **lightweight, reproducible framework** to evaluate and visualize the performance of a **Retrieval-Augmented Generation (RAG)** system using quantitative and semantic metrics.

The evaluation scripts analyze RAG outputs (stored in Excel or CSV files) and compute standard NLP metrics to assess **semantic quality**, **lexical fidelity**, and **model reliability** across different retrieval depths (`k=1,3,5`) and query cohorts (Known, Inferred, Out-of-KB).

---

## Repository Structure

````
rag_eval/
├── data/                 # Input files (e.g., RAGfaq.xlsx)
├── output_results/       # Generated metrics & summaries
│   ├── metrics_per_query.csv
│   └── metrics_by_cohort.csv
├── plots/                # Generated figures (PNG only)
├── evaluate.py           # Core evaluation pipeline
├── visualize.py          # Plot generation and visualization
├── utils.py              # Helper functions (cleaning, scoring)
├── requirements.txt      # Dependencies list
├── eval_README.md        # (This file)
````



---
## Installation


```bash
# Clone and enter the folder
git clone https://github.com/rmit-ds-investor-assistant/WIL_RAG.git
cd WIL_RAG/rag_eval

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt 
```

## Run Evaluation

python evaluate.py --input data/RAGfaq.xlsx --outdir output_results

````
This generates:
	•	metrics_per_query.csv — Per-question detailed results
	•	metrics_by_cohort.csv — Aggregated scores by cohort and k
````

## Metrics Used

| Category | Metric | Description |
|-----------|---------|-------------|
| **Lexical overlap** | **ROUGE-L** | Measures word-sequence overlap (Longest Common Subsequence **F1**) between generated and reference answers, capturing fluency and coverage. |
| **Semantic similarity** | **SBERT cosine** | Uses Sentence-BERT embeddings to measure semantic alignment — robust to paraphrasing and word reordering. |
| **Reliability** | **Good Refusal Rate / Hallucination Rate** | Evaluates the model’s ability to abstain on unknown queries and avoid fabricating information. |

## Visualizations

python visualize.py --input output_results/metrics_by_cohort.csv

```aiignore
Plots will be saved in /plots/, including:
	•	trend_small_multiples_rougeL.png — ROUGE-L by cohort and retrieval depth
	•	trend_small_multiples_semantic_cosine.png — SBERT cosine similarity trends
```

## Notes
 	•	Embeddings computed via SentenceTransformers (all-MiniLM-L6-v2).
	•	ROUGE-L calculated using Google’s rouge_score library.
	•	Evaluation optimized for readability and reproducibility on local hardware (Mac M3, 16 GB RAM).


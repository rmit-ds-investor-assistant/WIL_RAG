This repository provides a lightweight, reproducible framework to evaluate and visualize the performance of a Retrieval-Augmented Generation (RAG) system using standard NLP metrics.

Running Evaluation

The evaluation script reads Excel or CSV files containing reference (gold) and model-generated answers.

Each sheet or dataset typically represents a query cohort:
Known: In-domain (ground-truth available)
Inferred: Reasoning/generalization queries
Out_of_KB: Out-of-knowledge-base queries

Run evaluation:
python evaluate.py –input data/RAGfaq.xlsx

This generates:
output_results/metrics_summary.csv
output_results/metrics_by_cohort.csv
Detailed per-query metrics (ROUGE-1, BERTScore, NDCG, % Answered, etc.)

Generating Visualizations

Once metrics are computed, visualize results with:
python visualize.py –input output_results/metrics_by_cohort.csv –out output_results/plots.png

This will create publication-ready charts for:
ROUGE-1 (Lexical overlap between generated and reference answers)
BERTScore (Semantic similarity using embeddings)
NDCG (Ranking quality of retrieved documents)
% Answered (Fraction of queries answered within the knowledge base)

The plots use a soft watercolor theme for readability and presentation quality—ideal for reports, academic submissions, or video explanations.


Metrics Used

ROUGE-1: Measures unigram overlap between generated and reference answers.
BERTScore: Uses BERT embeddings to compute semantic similarity beyond surface text.
NDCG: Normalized Discounted Cumulative Gain—evaluates ranked retrieval quality.
% Answered: Indicates system coverage; high values suggest robust knowledge retrieval.


Output
Grouped plots by Cohort (Known / Inferred / Out-of-KB)
Bar charts by Retrieval Depth (k)



Attribution

This evaluation setup draws conceptual inspiration from the Walert (RMIT IR Lab) project, adapted for educational and practical reproducibility in Case Studies in Data Science coursework (COSC2669).
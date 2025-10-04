import pandas as pd
from rouge_score import rouge_scorer
import numpy as np

# Load your FAQ evaluation Excel/CSV
df = pd.read_excel("RAGfaq.xlsx")   # change filename

# Expected columns in your file:
# Query | Expected Company | Gold Answer | Generated answer k=3 | Generated answer k=5

# Initialize scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Add new columns for ROUGE scores
for col in ["Generated answer k=3", "Generated answer k=5"]:
    rouge1_list, rouge2_list, rougel_list = [], [], []
    
    for i, row in df.iterrows():
        gold = str(row["Gold Answer"])
        gen = str(row[col])

        scores = scorer.score(gold, gen)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougel_list.append(scores["rougeL"].fmeasure)
    
    df[f"{col}_ROUGE1"] = rouge1_list
    df[f"{col}_ROUGE2"] = rouge2_list
    df[f"{col}_ROUGEL"] = rougel_list

# Compute averages
results = {}
for col in ["Generated answer k=3", "Generated answer k=5"]:
    results[col] = {
        "ROUGE-1": df[f"{col}_ROUGE1"].mean(),
        "ROUGE-2": df[f"{col}_ROUGE2"].mean(),
        "ROUGE-L": df[f"{col}_ROUGEL"].mean()
    }

print("📊 Average ROUGE Scores:")
for model, scores in results.items():
    print(f"\n{model}:")
    for k,v in scores.items():
        print(f"  {k}: {v:.3f}")

# Save back to Excel with scores
df.to_excel("faq_eval_with_rouge.xlsx", index=False)
print("\n✅ Results saved to faq_eval_with_rouge.xlsx")

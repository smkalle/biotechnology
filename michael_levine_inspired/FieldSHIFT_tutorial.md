# FieldSHIFT Exploration Tutorial

Machine Learning & Bioinformatics Self‑Study Guide  
*Bridging Developmental Bioelectricity with Cross‑Disciplinary Insights*

---

## 1 ️⃣ Overview

FieldSHIFT is a machine‑generated hypothesis dataset published on Hugging Face that proposes mechanistic links between developmental bioelectricity and diverse biological processes.  
This tutorial walks you through:

1. **Accessing** the FieldSHIFT dataset programmatically.  
2. **Exploring** and filtering the hypotheses it contains.  
3. **Extending** the dataset by prompting a large‑language model (LLM) to draft *new* cross‑field research questions.  
4. **Evaluating novelty** using simple semantic‑similarity techniques.  

> **Audience:** graduate‑level Bioinformatics or Computational Biology researchers looking to integrate NLP & ML workflows into literature exploration.

---

## 2 ️⃣ Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python ≥ 3.9 | runtime | – |
| [`datasets`](https://github.com/huggingface/datasets) | load FieldSHIFT | `pip install datasets` |
| [`transformers`](https://github.com/huggingface/transformers) | LLM access & embeddings | `pip install transformers` |
| [`openai`](https://github.com/openai/openai-python) <br>or <br> [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) | API wrapper (choose one) | `pip install openai` *or* `pip install huggingface_hub` |
| `scikit‑learn` | cosine similarity | `pip install scikit-learn` |

Create a virtualenv and install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install datasets transformers scikit-learn openai
```

Add your OpenAI key (or Hugging Face token) to your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 3 ️⃣ Access the FieldSHIFT Dataset

```python
from datasets import load_dataset

# The dataset lives at `synthesizebio/FieldSHIFT`
ds = load_dataset("synthesizebio/FieldSHIFT", split="train")
print(ds)
ds[:3]          # peek at first three entries
```

Typical schema ➜ `hypothesis`, `source_pubmed_id`, `confidence`, `bridge_field`.

---

## 4 ️⃣ Explore Generated Hypotheses

### 4.1 Basic Summary

```python
import pandas as pd

df = ds.to_pandas()
print(df["bridge_field"].value_counts()[:10])      # top linked fields
```

### 4.2 Filter by Confidence

```python
high_conf = df[df["confidence"] > 0.8]
```

### 4.3 Visualize Term Frequencies

```python
from collections import Counter
import matplotlib.pyplot as plt
words = Counter(" ".join(high_conf.hypothesis).lower().split())
plt.bar(*zip(*words.most_common(20))); plt.title("Top Terms in High‑Confidence Hypotheses")
```

---

## 5 ️⃣ Exercise: Generate New Cross‑Field Questions

Use an LLM of your choice (GPT‑4o shown).

```python
import openai, textwrap

TEMPLATE = textwrap.dedent("""You are a systems-biology researcher. Suggest **5** novel research questions that connect
developmental bioelectricity with {bridge_field}. Each question should be specific, testable, and <200 characters.
Return as a numbered list.
""")

def propose_questions(bridge_field: str) -> list[str]:
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":TEMPLATE.format(bridge_field=bridge_field)}],
        temperature=0.9
    )
    return [q.strip(" .") for q in resp.choices[0].message.content.splitlines() if q.strip()]

new_qs = propose_questions("immunology")
print(new_qs)
```

---

## 6 ️⃣ Evaluate Novelty

### 6.1 Embed Text

```python
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    toks = tok(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        embs = model(**toks).last_hidden_state.mean(1)
    return embs / embs.norm(dim=1, keepdim=True)

base_vecs = embed(df.hypothesis.tolist())
new_vecs  = embed(new_qs)
```

### 6.2 Cosine Similarity & Threshold

```python
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(new_vecs, base_vecs).max(axis=1)
for q, s in zip(new_qs, sims):
    novelty = "✅ NEW" if s < 0.7 else "⚠️ Similar"
    print(f"{novelty}  ({s:.2f})  {q}")
```

Interpretation: lower similarity → higher novelty.

---

## 7 ️⃣ Suggested Projects

| Idea | Description |
|------|-------------|
| **Active‑Learning Loop** | Re‑prompt the LLM only on topics with sparse FieldSHIFT coverage. |
| **Hypothesis Prioritization** | Combine novelty score with citation counts of referenced genes/proteins. |
| **Bench‑to‑Dataset** | Validate top‑ranked questions via literature mining and manual expert review. |

---

## 8 ️⃣ Further Reading

1. Levin M. *Bioelectric signaling in development and regeneration.* **Cell** (2021)  
2. FieldSHIFT Dataset Card – <https://huggingface.co/datasets/synthesizebio/FieldSHIFT>  
3. Baglaenko et al. *LLM‑assisted scientific discovery.* **arXiv** (2023)  
4. Abid et al. *Concrete autoencoders uncover novel associations in genomics.* **Nature ML** (2022)

---

## 9 ️⃣ License

This tutorial is released under the MIT License.  
FieldSHIFT dataset may have its own license terms—verify before redistribution.

---

> **Happy hypothesis hunting!** 🧠⚡️

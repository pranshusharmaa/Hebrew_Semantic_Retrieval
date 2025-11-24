# Hebrew Semantic Retrieval (Zero-Shot, No Fine-Tuning)

This repository contains my solution attempt for a Hebrew semantic retrieval challenge, built under a strict constraint:

- I did not have access to the competition dataset.
- I therefore could not fine-tune any model or evaluate against the official labels.

Instead, I designed a zero-shot retrieval pipeline using:

- A pre-trained multilingual dense encoder: `intfloat/multilingual-e5-base`
- Classical IR baselines: TF–IDF and BM25
- A hybrid scoring scheme that combines lexical and dense similarity

The focus of this project is on architecture and retrieval strategy rather than leaderboard optimization.

---

## Project Overview

The core goal is to retrieve semantically relevant Hebrew texts for a given query.

Because the official dataset was not accessible, this project focuses on:

- Designing a retrieval system that works in a data-scarce / zero-shot setting.
- Leveraging a pre-trained multilingual Transformer-based encoder (`intfloat/multilingual-e5-base`) to generate dense representations.
- Implementing TF–IDF and BM25 as strong lexical baselines.
- Building a modular pipeline that can later be extended to:
  - Supervised training / fine-tuning when labeled data is available.
  - More advanced hybrid retrieval and reranking strategies.

In practice, this repository demonstrates how to reason about and implement semantic retrieval when you only have the task definition, not the labeled data.

---

## Constraints and Design Choices

### Key constraint

- No access to the Hebrew dataset  
  → No supervised training, no fine-tuning, and no metric-based model selection on the official data.

### Design choices under this constraint

- Use `intfloat/multilingual-e5-base`:
  - A pre-trained multilingual sentence encoder that supports Hebrew.
  - Used strictly in zero-shot mode (no training, no fine-tuning).
- Implement lexical baselines:
  - TF–IDF vectors with cosine similarity.
  - BM25 scoring for bag-of-words relevance.
- Build a hybrid retrieval pipeline:
  - Lexical retrieval (TF–IDF / BM25) to obtain a candidate set.
  - Dense retrieval using multilingual E5 to re-score or combine with lexical scores.
- Keep the system:
  - Model-agnostic – easy to swap in another encoder.
  - Data-ready – the same structure can be used for fine-tuning and proper evaluation once Hebrew labels are available.
  - Simple enough to be reliable in a competition setting with limited submissions.

---

## Methodology

### 1. Task Definition

- Input: A Hebrew query (short text).
- Output: A ranked list of Hebrew candidate texts that are semantically related to the query.

This mirrors a standard semantic search / retrieval setup.

---

### 2. Approach

#### 2.1 Hebrew Text Preprocessing

Preprocessing is intentionally simple and generic, since the real dataset was not available:

- Basic normalization (e.g., trimming whitespace, normalizing punctuation).
- Light cleaning to remove obvious noise.
- No dataset-specific rules or heavy token-level heuristics.

---

#### 2.2 Lexical Retrieval: TF–IDF

As a first baseline, I use TF–IDF:

- Build a TF–IDF vectorizer over the candidate corpus.
- Represent:
  - Queries as TF–IDF vectors.
  - Candidate texts as TF–IDF vectors.
- Compute cosine similarity between the query vector and each candidate vector.
- Rank candidates by similarity.

This provides a simple but solid lexical matching baseline that is easy to interpret and debug.

---

#### 2.3 Lexical Retrieval: BM25

To better capture term frequency and saturation effects, I also implement BM25:

- Represent documents using a bag-of-words model.
- Use a BM25 scorer to compute relevance between the query and each candidate.
- Rank candidates by BM25 score.

BM25 typically outperforms plain TF–IDF on many retrieval tasks and serves as a stronger lexical baseline.

---

#### 2.4 Dense Retrieval: `intfloat/multilingual-e5-base`

For semantic retrieval, I use the E5 multilingual model:

- Model: `intfloat/multilingual-e5-base`
- Behavior:
  - Encode queries into dense embeddings.
  - Encode candidate texts / documents into dense embeddings.
- Use cosine similarity between dense embeddings as a semantic similarity signal.

The E5 model is used as-is:

- No fine-tuning.
- No training loop.
- No supervised optimization on the Hebrew task.

This is a purely zero-shot dense retrieval setup.

---

#### 2.5 Hybrid Scoring (Lexical + Dense)

To combine lexical and semantic signals:

1. Use TF–IDF or BM25 to obtain an initial candidate set.
2. Compute dense similarity scores using multilingual E5.
3. Combine scores, for example via a weighted sum:

\[
\text{score}(d) = \alpha \cdot \text{lexical\_score}(d) + (1 - \alpha) \cdot \text{dense\_score}(d)
\]

where:

- `lexical_score(d)` is from TF–IDF or BM25,
- `dense_score(d)` is cosine similarity in E5 embedding space,
- `α` is a heuristic weight (since no labeled validation data was available).

This produces a hybrid ranking that:

- Preserves exact term matching (lexical).
- Benefits from semantic similarity and multilingual understanding (dense).

---

#### 2.6 Cross-Encoder Reranking: BGE Reranker

On top of the hybrid retrieval stage, I optionally apply a cross-encoder reranking step using a BGE reranker model (`[BGE_RERANKER_MODEL]`).

The setup is:

1. Use the hybrid pipeline (TF–IDF / BM25 + E5) to retrieve a small set of top-N candidates for each query.
2. For each (query, candidate) pair in this top-N set, run the BGE reranker as a cross-encoder:
   - Concatenate query and candidate.
   - Obtain a relevance score from the BGE model.
3. Re-rank the top-N candidates by the BGE score to produce the final ordering.

Important points:

- The BGE reranker is also used in zero-shot mode: no fine-tuning and no task-specific training.
- Because it is applied only to a small top-N set (rather than the entire corpus), the additional latency is controlled and remains within the 2-second time budget.
- This step is designed to improve fine-grained ranking (especially when multiple candidates are semantically close) without changing the overall architecture of the retrieval stack.


#### 2.7 Indexing and Efficiency (Why FAISS Was Not Used)

I intentionally did not use FAISS or approximate nearest neighbor (ANN) indices in this project.

Rationale:

- The competition effectively allowed a single meaningful submission:
  - Reliability and simplicity were more important than squeezing out marginal latency gains.
  - Over-engineering the retrieval stack increased the risk of subtle bugs without a strong upside.
- With the current implementation:
  - Exact cosine similarity over dense embeddings ran in roughly 80 ms per query.
  - The competition’s time budget was around 2 seconds.
- In this context:
  - FAISS or ANN indexing would realistically save on the order of tens of milliseconds (around 50 ms at best) within a 2-second constraint.
  - That gain is not significant enough to justify the additional complexity, dependencies, and potential edge cases.

For this reason, I kept the retrieval stack straightforward:

- Exact cosine similarity.
- No FAISS / ANN layer.
- Focus on correctness, readability, and robustness for a one-shot submission scenario.

FAISS remains an obvious extension point for future iterations if the system needs to scale to much larger corpora or stricter latency requirements.

---

#### 2.8 Evaluation Strategy (Without Official Labels)

Because I did not have access to the official labels, I could not compute NDCG, MRR, or Recall@k on the true data.

Instead, I relied on sanity checks and qualitative evaluation:

- Constructed small synthetic examples (toy queries and candidate sets) to verify:
  - BM25 and TF–IDF behave sensibly.
  - E5 embeddings retrieve semantically related Hebrew sentences.
  - Hybrid scoring improves over purely lexical or purely dense retrieval in intuitive scenarios.
- Performed manual inspection of ranked results for Hebrew queries:
  - Checked whether top-ranked candidates were reasonable for the query.
  - Compared the behavior of TF–IDF, BM25, E5-only, and the hybrid approach.

The emphasis is on getting the architecture and logic correct, rather than optimizing a specific leaderboard metric.

---

## Tech Stack

- Language: Python
- Dense embeddings: `intfloat/multilingual-e5-base`
- Lexical retrieval:
  - TF–IDF (e.g., via `scikit-learn`)
  - BM25 (via a BM25 implementation library)
- Core libraries (typical):
  - `transformers` / `sentence-transformers`
  - `numpy`, `scipy`
  - `scikit-learn`
  - `tqdm`, `argparse` for CLI support

---


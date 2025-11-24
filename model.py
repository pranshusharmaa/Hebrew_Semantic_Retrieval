import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import bm25s


def reciprocal_rank_fusion(results_list, k=60, weights=None):
    """
    Weighted Reciprocal Rank Fusion (RRF).

    results_list: list of ranked lists, each list is [(doc_id, score), ...] sorted by score desc.
    weights: optional list of floats, same length as results_list.
    """
    if weights is None:
        weights = [1.0] * len(results_list)

    fused_scores = {}
    for list_idx, results in enumerate(results_list):
        w = weights[list_idx]
        for rank, (doc_id, _score) in enumerate(results):
            # Classic RRF term: 1 / (k + rank + 1)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + w * (1.0 / (k + rank + 1))

    # Sort by fused score descending
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return reranked


class E5Retriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the E5 retriever using the multilingual E5 base model.
        """
        # Use local model
        if model_name is None:
            local_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models",
                "multilingual-e5-base",
            )
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local E5 model from: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading E5 multilingual model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.corpus_ids = []
        self.corpus_embeddings = None

    def embed_texts(self, texts, is_query=False, batch_size=32):
        """
        Generates embeddings for texts using E5 model with proper prefixes.
        E5 requires specific prefixes for queries vs passages.
        """
        # E5 model requires specific prefixes
        if is_query:
            # Add query prefix for E5
            prefixed_texts = [f"query: {text.strip()}" for text in texts]
        else:
            # Add passage prefix for E5
            prefixed_texts = [f"passage: {text.strip()}" for text in texts]

        all_embeddings = []
        total_batches = (len(prefixed_texts) + batch_size - 1) // batch_size

        for i in range(0, len(prefixed_texts), batch_size):
            batch_num = i // batch_size + 1
            if not is_query and batch_num % 50 == 0:
                print(
                    f"Processing batch {batch_num}/{total_batches}"
                    f" ({(batch_num / total_batches) * 100:.1f}%)"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch_texts = prefixed_texts[i : i + batch_size]

            try:
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded)

                    # E5 uses mean pooling with attention mask
                    attention_mask = encoded["attention_mask"]
                    embeddings = model_output.last_hidden_state

                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                    # L2 normalize embeddings (important for E5)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Move to CPU immediately
                all_embeddings.append(embeddings.cpu())

                # Clear GPU memory
                del encoded, model_output, embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM at batch {batch_num}, reducing batch size... ({e})")
                # Process one item at a time
                for single_text in batch_texts:
                    try:
                        encoded = self.tokenizer(
                            [single_text],
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        ).to(self.device)

                        with torch.no_grad():
                            model_output = self.model(**encoded)
                            attention_mask = encoded["attention_mask"]
                            embeddings = model_output.last_hidden_state

                            # Mean pooling
                            mask_expanded = (
                                attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                            )
                            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                            embeddings = torch.nn.functional.normalize(
                                embeddings,
                                p=2,
                                dim=1,
                            )

                        all_embeddings.append(embeddings.cpu())

                        del encoded, model_output, embeddings
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e2:
                        print(f"Failed to process single text: {e2}")
                        # E5-base has 768 dimensions
                        zero_embedding = torch.zeros(1, 768).float()
                        all_embeddings.append(zero_embedding)

        return torch.cat(all_embeddings, dim=0).numpy()


class BGEReranker:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the BGE reranker for fine-grained relevance scoring.
        """
        # Use local model
        if model_name is None:
            local_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models",
                "bge-reranker-v2-m3",
            )
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local BGE model from: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading BGE reranker on device: {self.device}")

        # BGE reranker is actually a special model type
        from transformers import AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rerank(self, query_text, passages, passage_ids, top_k=20):
        """
        Rerank the passages using BGE reranker.
        """
        if not passages:
            return []

        scores = []
        batch_size = 4  # Conservative batch size

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]

            try:
                # BGE reranker expects SEPARATE query and passage inputs
                batch_queries = [query_text] * len(batch_passages)

                # Tokenize query-passage pairs properly
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch_queries,
                        batch_passages,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)

                    # Get relevance scores from sequence classification model
                    outputs = self.model(**inputs)

                    # BGE reranker outputs logits for relevance classification
                    logits = outputs.logits

                    # Handle different output shapes
                    if len(logits.shape) == 1:
                        # Single score per pair
                        batch_scores = logits.cpu().numpy()
                    elif logits.shape[1] == 1:
                        # Single column output
                        batch_scores = logits.squeeze(-1).cpu().numpy()
                    else:
                        # Binary classification - take positive class (index 1)
                        batch_scores = logits[:, 1].cpu().numpy()

                scores.extend(batch_scores.tolist())

                # Cleanup
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in reranking batch {i // batch_size + 1}: {e}")
                # Fallback: Use neutral scores for this batch
                fallback_scores = [0.5] * len(batch_passages)
                scores.extend(fallback_scores)

        # Combine results and sort by reranking score
        results = list(zip(passage_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


# Global instances
retriever = None          # E5Retriever (dense retriever)
reranker = None           # BGEReranker (cross-encoder)
corpus_texts = {}         # doc_id -> passage text

# TF-IDF globals
tfidf_vectorizer = None
tfidf_matrix = None

# BM25s globals
bm25_retriever = None     # bm25s.BM25 instance


def preprocess(corpus_dict):
    """
    Preprocessing function using:
      - E5 multilingual model (dense retriever)
      - BGE reranker (cross-encoder)
      - TF-IDF lexical retriever (1–2 grams)
      - BM25s lexical retriever

    Input:
        corpus_dict: dict mapping document IDs -> document objects
                     with 'passage' or 'text' fields.

    Output:
        dict containing initialized models, embeddings, and corpus data.
    """
    global retriever, reranker, corpus_texts
    global tfidf_vectorizer, tfidf_matrix, bm25_retriever

    print("=" * 60)
    print("PREPROCESSING: Initializing E5 + BGE + TF-IDF + BM25s pipeline...")
    print("=" * 60)

    # GPU memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ----- E5 dense retriever -----
    print("Loading E5 retriever...")
    retriever = E5Retriever()

    # ----- BGE cross-encoder reranker -----
    print("Loading BGE reranker...")
    reranker = BGEReranker()

    num_docs = len(corpus_dict)
    print(f"Preparing corpus with {num_docs} documents...")

    # Preserve a fixed order for corpus IDs
    retriever.corpus_ids = list(corpus_dict.keys())

    # Extract passage text in the same order as corpus_ids
    passages = [
        corpus_dict[doc_id].get("passage", corpus_dict[doc_id].get("text", ""))
        for doc_id in retriever.corpus_ids
    ]

    # Map doc_id -> passage text
    corpus_texts = {
        doc_id: passages[i]
        for i, doc_id in enumerate(retriever.corpus_ids)
    }

    # ----- Dense E5 embeddings -----
    print("Computing E5 embeddings for corpus...")
    retriever.corpus_embeddings = retriever.embed_texts(
        passages,
        is_query=False,
        batch_size=32,
    )

    print("✓ Dense E5 corpus embeddings complete")
    print(f"  - Documents: {len(retriever.corpus_ids)}")
    print(f"  - Embedding matrix shape: {retriever.corpus_embeddings.shape}")

    # ----- TF-IDF lexical index -----
    print("Building TF-IDF index (1–2 grams, max_features=100k)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        lowercase=False,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(passages)
    print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

    # ----- BM25s lexical index -----
    print("Building BM25s index...")
    # For Hebrew, do not force English stopwords; use defaults.
    corpus_tokens = bm25s.tokenize(passages)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens)
    print("✓ BM25s index built")

    print("✓ Corpus preprocessing complete!")

    return {
        "retriever": retriever,
        "reranker": reranker,
        "corpus_ids": retriever.corpus_ids,
        "corpus_embeddings": retriever.corpus_embeddings,
        "corpus_texts": corpus_texts,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "bm25_retriever": bm25_retriever,
        "num_documents": num_docs,
    }


def predict(query, preprocessed_data):
    """
    Hybrid prediction:
      E5 dense retrieval + TF-IDF lexical + BM25s lexical
      -> weighted 3-way RRF fusion
      -> BGE reranking on top-100 candidates.

    Input:
        query: dict with 'query' field containing query text
        preprocessed_data: dict from preprocess() with models and corpus data

    Output:
        list of dicts:
            {
                "paragraph_uuid": <doc_id>,
                "score": <float BGE score>
            }
        sorted by relevance desc.
    """
    global retriever, reranker, corpus_texts
    global tfidf_vectorizer, tfidf_matrix, bm25_retriever

    # Extract query text
    query_text = query.get("query", "")
    if not query_text:
        return []

    # Ensure globals are initialized (Codabench reload safety)
    if (
        retriever is None
        or reranker is None
        or tfidf_vectorizer is None
        or tfidf_matrix is None
        or bm25_retriever is None
    ):
        retriever = preprocessed_data.get("retriever")
        reranker = preprocessed_data.get("reranker")
        corpus_texts = preprocessed_data.get("corpus_texts", {})
        tfidf_vectorizer = preprocessed_data.get("tfidf_vectorizer")
        tfidf_matrix = preprocessed_data.get("tfidf_matrix")
        bm25_retriever = preprocessed_data.get("bm25_retriever")

        if (
            retriever is None
            or reranker is None
            or tfidf_vectorizer is None
            or tfidf_matrix is None
            or bm25_retriever is None
        ):
            print("Error: Missing components in preprocessed data")
            return []

    # Number of candidates for each retriever and for reranking
    CANDIDATE_COUNT = 100

    try:
        # ==========================
        # STAGE 1a: E5 dense retrieval
        # ==========================
        print("Stage 1a: E5 retrieval...")
        query_embedding = retriever.embed_texts(
            [query_text],
            is_query=True,
            batch_size=1,
        )

        e5_scores = cosine_similarity(
            query_embedding,
            retriever.corpus_embeddings,
        )[0]

        e5_top_indices = np.argsort(e5_scores)[::-1][:CANDIDATE_COUNT]
        e5_results = [
            (retriever.corpus_ids[idx], float(e5_scores[idx]))
            for idx in e5_top_indices
        ]

        # ==========================
        # STAGE 1b: TF-IDF lexical retrieval
        # ==========================
        print("Stage 1b: TF-IDF retrieval...")
        tfidf_results = []
        try:
            query_vec = tfidf_vectorizer.transform([query_text])
            tfidf_scores = (query_vec @ tfidf_matrix.T).toarray()[0]
            tfidf_top_indices = np.argsort(tfidf_scores)[::-1][:CANDIDATE_COUNT]
            tfidf_results = [
                (retriever.corpus_ids[idx], float(tfidf_scores[idx]))
                for idx in tfidf_top_indices
            ]
        except Exception as e_lex:
            print(f"TF-IDF retrieval failed, proceeding without it: {e_lex}")

        # ==========================
        # STAGE 1c: BM25s lexical retrieval
        # ==========================
        print("Stage 1c: BM25s retrieval...")
        bm25_results = []
        try:
            # Tokenize single query; returns token IDs suitable for BM25.retrieve
            query_tokens = bm25s.tokenize([query_text])
            bm25_ids, bm25_scores = bm25_retriever.retrieve(
                query_tokens,
                k=CANDIDATE_COUNT,
            )
            # bm25_ids and bm25_scores have shape (n_queries, k); here n_queries = 1
            doc_indices = bm25_ids[0]
            doc_scores = bm25_scores[0]

            bm25_results = [
                (retriever.corpus_ids[int(doc_idx)], float(doc_scores[pos]))
                for pos, doc_idx in enumerate(doc_indices)
            ]
        except Exception as e_bm25:
            print(f"BM25s retrieval failed, proceeding without it: {e_bm25}")

        # ==========================
        # STAGE 1d: 3-way RRF fusion
        # ==========================
        print("Stage 1d: RRF fusion (E5 + BM25s + TF-IDF)...")
        results_lists = []
        weights = []

        # Always include E5
        results_lists.append(e5_results)
        weights.append(1.5)  # E5 is primary signal

        # Include BM25s if available
        if bm25_results:
            results_lists.append(bm25_results)
            weights.append(1.0)

        # Include TF-IDF if available
        if tfidf_results:
            results_lists.append(tfidf_results)
            weights.append(0.7)

        fused_results = reciprocal_rank_fusion(
            results_lists,
            k=60,
            weights=weights,
        )

        # Take top-100 fused docs for reranking
        candidate_ids = [
            doc_id for doc_id, _score in fused_results[:CANDIDATE_COUNT]
        ]
        candidate_passages = [
            corpus_texts.get(doc_id, "") for doc_id in candidate_ids
        ]

        # ==========================
        # STAGE 2: BGE reranking
        # ==========================
        print(f"Stage 2: BGE reranking on {len(candidate_ids)} fused candidates...")
        reranked_results = reranker.rerank(
            query_text,
            candidate_passages,
            candidate_ids,
            top_k=20,
        )

        # Final results: use actual BGE scores
        results = []
        for passage_id, rerank_score in reranked_results:
            results.append(
                {
                    "paragraph_uuid": passage_id,
                    "score": float(rerank_score),
                }
            )

        print(f"✓ Returned {len(results)} results with reranker scores")
        return results

    except Exception as e:
        print(f"Error in prediction, falling back to dense-only E5: {e}")
        # Fallback to E5-only retrieval with E5 scores
        try:
            query_embedding = retriever.embed_texts(
                [query_text],
                is_query=True,
                batch_size=1,
            )
            e5_scores = cosine_similarity(
                query_embedding,
                retriever.corpus_embeddings,
            )[0]
            top_indices = np.argsort(e5_scores)[::-1][:20]

            results = []
            for idx in top_indices:
                results.append(
                    {
                        "paragraph_uuid": retriever.corpus_ids[idx],
                        "score": float(e5_scores[idx]),
                    }
                )

            return results
        except Exception as e2:
            print(f"Secondary fallback also failed: {e2}")
            return []

"""
retrieval.py — PubMed E-utilities fetcher + Hybrid Retrieval (Dense + BM25 + RRF) + Cross-Encoder Reranking.

Features:
1. Live fetch from NCBI E-utilities (esearch + efetch) with local disk caching.
2. Passage chunking (2-3 sentences per chunk).
3. Hybrid Retrieval:
   - Dense retrieval with sentence-transformers 'all-MiniLM-L6-v2'
   - Sparse retrieval with BM25 (rank_bm25)
   - Reciprocal Rank Fusion (RRF, k=60)
4. Reranking with 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
"""

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from src.data_utils import BASE_DIR, load_disease_config

# ── Paths ──────────────────────────────────────────────────────────────────
CACHE_DIR = BASE_DIR / 'data' / 'pubmed_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Models (Lazy Loaded) ───────────────────────────────────────────────────
_DENSE_MODEL: Optional[SentenceTransformer] = None
_RERANK_MODEL: Optional[CrossEncoder] = None


def get_dense_model() -> SentenceTransformer:
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        print("  Loading dense retriever ('all-MiniLM-L6-v2')...")
        _DENSE_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _DENSE_MODEL


def get_rerank_model() -> CrossEncoder:
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        print("  Loading reranker ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
        _RERANK_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _RERANK_MODEL


# ── PubMed E-utilities API ─────────────────────────────────────────────────
def fetch_pubmed_abstracts(query: str, max_results: int = 20) -> List[Dict]:
    """
    Search PubMed via NCBI E-utilities (esearch + efetch) and parse XML response.
    Returns list of dicts: {'pmid': str, 'title': str, 'abstract': str}
    Caches results locally to data/pubmed_cache/<safe_query>.json
    """
    safe_query_filename = re.sub(r'[^a-zA-Z0-9_]', '_', query)[:60] + ".json"
    cache_path = CACHE_DIR / safe_query_filename

    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            if articles:
                print(f"  Loaded {len(articles)} PubMed articles from cache ({cache_path.name})")
                return articles
        except Exception as e:
            print(f"  [WARNING] Cache read failed: {e}. Re-fetching...")

    print(f"  Fetching PubMed abstracts for query: '{query}'...")

    # Step 1: ESearch
    base_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params_search = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'json',
        'sort': 'relevance',
    }
    url_search = f"{base_esearch}?{urllib.parse.urlencode(params_search)}"

    req = urllib.request.Request(url_search, headers={'User-Agent': 'DiseaseRAGPipeline/1.0'})
    with urllib.request.urlopen(req) as resp:
        search_data = json.loads(resp.read().decode('utf-8'))

    id_list = search_data.get('esearchresult', {}).get('idlist', [])
    if not id_list:
        print(f"  [WARNING] No PubMed IDs found for query '{query}'")
        return []

    # Step 2: EFetch
    base_efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params_fetch = {
        'db': 'pubmed',
        'id': ','.join(id_list),
        'retmode': 'xml',
    }
    url_fetch = f"{base_efetch}?{urllib.parse.urlencode(params_fetch)}"

    req_fetch = urllib.request.Request(url_fetch, headers={'User-Agent': 'DiseaseRAGPipeline/1.0'})
    with urllib.request.urlopen(req_fetch) as resp:
        xml_data = resp.read()

    # Step 3: Parse XML
    root = ET.fromstring(xml_data)
    articles = []

    for article_elem in root.findall('.//PubmedArticle'):
        pmid_elem = article_elem.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else 'Unknown'

        title_elem = article_elem.find('.//ArticleTitle')
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else 'No title'

        abstract_texts = []
        for abs_text in article_elem.findall('.//AbstractText'):
            label = abs_text.attrib.get('Label', '')
            txt = "".join(abs_text.itertext()).strip()
            if label:
                abstract_texts.append(f"{label}: {txt}")
            else:
                abstract_texts.append(txt)

        abstract = " ".join(abstract_texts).strip()

        if abstract:
            articles.append({
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
            })

    # Save to cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"  Cached {len(articles)} articles to {cache_path}")

    return articles


# ── Text Chunking ──────────────────────────────────────────────────────────
def split_into_sentences(text: str) -> List[str]:
    """Simple regex rule-based sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def chunk_articles(articles: List[Dict], sentences_per_chunk: int = 3) -> List[Dict]:
    """
    Chunk PubMed abstracts into 2-3 sentence passages.
    Returns list of dicts: {'chunk_id': str, 'pmid': str, 'title': str, 'text': str}
    """
    chunks = []
    for art in articles:
        sentences = split_into_sentences(art['abstract'])
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_text = " ".join(sentences[i:i + sentences_per_chunk])
            if len(chunk_text.split()) >= 10:  # Ignore tiny fragments
                chunks.append({
                    'chunk_id': f"{art['pmid']}_chunk_{i//sentences_per_chunk}",
                    'pmid': art['pmid'],
                    'title': art['title'],
                    'text': f"Title: {art['title']}. {chunk_text}",
                    'passage': chunk_text,
                })
    return chunks


# ── Hybrid Retrieval & Reranking ──────────────────────────────────────────
def reciprocal_rank_fusion(
    dense_rankings: List[Dict],
    sparse_rankings: List[Dict],
    rrf_k: int = 60,
    top_k: int = 15,
) -> List[Dict]:
    """Reciprocal Rank Fusion (RRF) combining dense and BM25 rankings."""
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for rank, item in enumerate(dense_rankings):
        cid = item['chunk_id']
        chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

    for rank, item in enumerate(sparse_rankings):
        cid = item['chunk_id']
        chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
    fused_chunks = []
    for cid in sorted_cids[:top_k]:
        chunk = chunk_map[cid].copy()
        chunk['rrf_score'] = rrf_scores[cid]
        fused_chunks.append(chunk)

    return fused_chunks


class PubMedRetriever:
    """Class encapsulating search, chunking, hybrid retrieval, and reranking."""

    def __init__(self, disease_name: str, query: Optional[str] = None):
        self.disease_name = disease_name
        config = load_disease_config(disease_name)
        self.query = query or config.get('pubmed_query', f"{disease_name} diagnosis classification")

        # Fetch & chunk
        self.articles = fetch_pubmed_abstracts(self.query, max_results=25)
        self.chunks = chunk_articles(self.articles, sentences_per_chunk=3)

        if not self.chunks:
            print(f"  [WARNING] No valid passages extracted for query '{self.query}'")
            return

        # Prepare BM25
        tokenized_corpus = [c['text'].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Prepare Dense Embeddings
        dense_model = get_dense_model()
        texts = [c['text'] for c in self.chunks]
        self.corpus_embeddings = dense_model.encode(texts, convert_to_tensor=True)

    def retrieve(self, query: str, top_k_final: int = 5) -> List[Dict]:
        """
        Run hybrid retrieval (dense + BM25 + RRF) followed by Cross-Encoder reranking.
        Returns top_k_final candidate passages with PMID, title, rerank score.
        """
        if not self.chunks:
            return []

        # 1. BM25 Sparse Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:20]
        sparse_results = [self.chunks[i] for i in top_bm25_idx]

        # 2. Dense Embedding Search
        dense_model = get_dense_model()
        query_emb = dense_model.encode(query, convert_to_tensor=True)
        cos_sims = torch.nn.functional.cosine_similarity(query_emb, self.corpus_embeddings).cpu().numpy()
        top_dense_idx = np.argsort(cos_sims)[::-1][:20]
        dense_results = [self.chunks[i] for i in top_dense_idx]

        # 3. Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(dense_results, sparse_results, top_k=15)

        # 4. Cross-Encoder Reranking
        rerank_model = get_rerank_model()
        pairs = [[query, c['text']] for c in fused_candidates]
        scores = rerank_model.predict(pairs)

        for chunk, score in zip(fused_candidates, scores):
            chunk['rerank_score'] = float(score)

        reranked = sorted(fused_candidates, key=lambda c: c['rerank_score'], reverse=True)
        return reranked[:top_k_final]


if __name__ == '__main__':
    # Test retrieval
    retriever = PubMedRetriever('breast_cancer')
    results = retriever.retrieve("malignant tumor features on ultrasound", top_k_final=3)
    print(f"\nTop 3 Retrieved Evidence:")
    for r in results:
        print(f"PMID: {r['pmid']} | Rerank Score: {r['rerank_score']:.4f}")
        print(f"Passage: {r['passage']}\n")

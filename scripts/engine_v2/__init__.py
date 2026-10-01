"""Engine v2: two-stage retrieval + LLM cross-reranker for the Fang Yuan
strategic decision engine.

Stage 1 (cheap recall): FTS5 bm25 search over decisions + classics
(contentless indexes joined back by rowid), with a LIKE-scan fallback.
Stage 2 (LLM cross-rerank): GLM scores each candidate 0-10 for
strategic-intent relevance; strict top-3 injection to avoid context
dilution ("lost in the middle" fix).
"""
from .rerank import (  # noqa: F401
    parse_score_response,
    recall_candidates,
    rerank,
    score_candidates,
)

__all__ = ["rerank", "recall_candidates", "score_candidates",
           "parse_score_response"]

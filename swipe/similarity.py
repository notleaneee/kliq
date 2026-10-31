"""
similarity.py — "Looks like" search with FAISS, and a similarity-based baseline.

FAISS (Facebook AI Similarity Search) finds the vectors closest to a query
vector, fast, even among millions of them.

Which index? IndexFlatIP: "flat" means it compares the query with every
vector (exact results, no approximation), "IP" means inner product. Our
CLIP embeddings are L2-normalised (length 1), and for unit vectors the inner
product IS the cosine similarity:
    1.0  = identical direction (looks the same)
    0.0  = unrelated
With ~500 sets an exact flat search takes well under a millisecond. For
millions of sets you'd switch to an approximate index (IndexIVFFlat or
IndexHNSWFlat), trading a little accuracy for a lot of speed.

The index is rebuilt from the cached embeddings each time the app starts,
which is instant, so it isn't saved to disk.

The baseline recommender ("similar to your likes"):
    1. average the embeddings of every set you liked -> your "taste vector"
    2. normalise it to length 1
    3. ask FAISS for the unrated sets closest to it
It ignores dislikes entirely and learns nothing; that's the point. A
baseline is the simple method the logistic regression model has to beat to
be worth having (Stage 6 measures that).
"""

from typing import Optional

import faiss
import numpy as np
import pandas as pd

MIN_LIKES = 1


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _search(index: faiss.IndexFlatIP, query: np.ndarray, k: int):
    """Top-k (similarities, rows) for one query vector."""
    query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
    k = min(k, index.ntotal)
    similarities, rows = index.search(query, k)
    keep = rows[0] >= 0                     # FAISS pads with -1 if it runs out
    return similarities[0][keep], rows[0][keep]


def _rows_to_sets(rows, similarities, index_df: pd.DataFrame) -> pd.DataFrame:
    by_row = index_df.set_index("embedding_row")["set_num"]
    return pd.DataFrame({"set_num": by_row.reindex(rows).to_numpy(), "similarity": similarities})


def similar_sets(set_num: str, index: faiss.IndexFlatIP, embeddings: np.ndarray,
                 index_df: pd.DataFrame, catalog: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """The k catalog sets that look most like `set_num` (itself excluded)."""
    match = index_df.loc[index_df["set_num"] == set_num, "embedding_row"]
    if match.empty:
        return catalog.head(0).assign(similarity=[])
    # Ask for extra results: the set itself comes back first, and some
    # embedded sets may not be in the catalog.
    sims, rows = _search(index, embeddings[int(match.iloc[0])], k + 10)
    found = _rows_to_sets(rows, sims, index_df)
    found = found[found["set_num"] != set_num]
    return catalog.merge(found, on="set_num").sort_values("similarity", ascending=False).head(k)


def taste_vector(ratings: pd.DataFrame, embeddings: np.ndarray,
                 index_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Normalised average embedding of your liked sets, or None if you've liked nothing."""
    liked = ratings[ratings["liked"] == 1].merge(index_df, on="set_num")
    if len(liked) < MIN_LIKES:
        return None
    centre = embeddings[liked["embedding_row"].to_numpy()].mean(axis=0)
    norm = np.linalg.norm(centre)
    return centre / norm if norm > 0 else None


def baseline_recommend(catalog: pd.DataFrame, ratings: pd.DataFrame, index: faiss.IndexFlatIP,
                       embeddings: np.ndarray, index_df: pd.DataFrame,
                       k: int = 5) -> Optional[pd.DataFrame]:
    """
    Top-k unrated catalog sets closest to your taste vector, with a
    `similarity` column (cosine, higher = closer). None if nothing liked yet.
    """
    taste = taste_vector(ratings, embeddings, index_df)
    if taste is None:
        return None
    # Rated sets will come back too (they're close to your likes!), so ask
    # FAISS for enough extra results to still have k after removing them.
    sims, rows = _search(index, taste, k + len(ratings) + 10)
    found = _rows_to_sets(rows, sims, index_df)
    found = found[~found["set_num"].isin(set(ratings["set_num"]))]
    out = catalog.merge(found, on="set_num")
    return out.sort_values(["similarity", "set_num"], ascending=[False, True]).head(k).reset_index(drop=True)

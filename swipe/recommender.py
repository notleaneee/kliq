"""
recommender.py — Learn what you like from your ratings, then rank unrated sets.

This is the machine learning part.

The data:
    X  one row per set you've rated: its 512-number CLIP embedding
       (what the set's picture looks like, as numbers)
    y  your label for that set: 1 = like, 0 = dislike

The model: logistic regression.
    It learns one weight per embedding number, plus a bias. For a new set it
    computes   z = w · x + b   and squashes z through the sigmoid
    1 / (1 + e^-z)   into a probability between 0 and 1:
    "how likely you are to like this set". Sets that look like your likes
    get high scores, sets that look like your dislikes get low ones.

Why logistic regression? It's simple, fast, and hard to overfit badly with
few examples, which matters because you'll start with only ~10 ratings.
Two settings help with that:
  * C (regularisation): keeps the weights small so the model doesn't
    memorise your handful of ratings. Smaller C = stronger smoothing,
    and scores closer to 50%.
  * class_weight="balanced": if you like 9 sets and dislike 1, the model
    would otherwise learn "always predict like". Balancing weights each
    class equally.

Training takes milliseconds, so the model is simply retrained whenever your
ratings change; nothing needs saving to disk.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MIN_RATINGS = 10       # total ratings before we train
MIN_PER_CLASS = 3      # at least this many likes AND dislikes

# Tried on simulated ratings (6 likes / 6 dislikes, real CLIP embeddings):
# C barely changes the ORDER of recommendations, only how spread out the
# scores are. C=1 squeezed every score to ~0.4-0.6; C=100 gave ~0.97 from
# 12 ratings (overconfident). C=10 sits in between. Stage 6 can tune it on
# held-out ratings.
DEFAULT_C = 10.0


def blas_quiet():
    """
    numpy 2.0 on macOS (Apple Accelerate) raises bogus "divide by zero /
    overflow in matmul" warnings on ordinary, finite matrix products. We
    silence them around training and prediction only, then check the results
    are finite ourselves, so a REAL numeric problem still fails loudly.
    """
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


@dataclass
class Readiness:
    ready: bool
    likes: int
    dislikes: int
    message: str


def check_readiness(ratings: pd.DataFrame) -> Readiness:
    """Do we have enough ratings, of both kinds, to train a model worth trusting?"""
    likes = int((ratings["liked"] == 1).sum()) if not ratings.empty else 0
    dislikes = int((ratings["liked"] == 0).sum()) if not ratings.empty else 0
    needed = []
    if likes + dislikes < MIN_RATINGS:
        needed.append(f"{MIN_RATINGS - likes - dislikes} more ratings")
    if likes < MIN_PER_CLASS:
        needed.append(f"{MIN_PER_CLASS - likes} more likes")
    if dislikes < MIN_PER_CLASS:
        needed.append(f"{MIN_PER_CLASS - dislikes} more dislikes")
    if needed:
        return Readiness(False, likes, dislikes, "Need " + ", ".join(needed) + ".")
    return Readiness(True, likes, dislikes, f"Learned from {likes + dislikes} ratings.")


def training_data(ratings: pd.DataFrame, embeddings: np.ndarray,
                  index_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn ratings into (X, y). Ratings for sets without an embedding are
    skipped (that shouldn't happen: the app only shows embedded sets).
    """
    rows = ratings.merge(index_df, on="set_num", how="inner")
    X = embeddings[rows["embedding_row"].to_numpy()]
    y = rows["liked"].to_numpy().astype(int)
    return X, y


def train(X: np.ndarray, y: np.ndarray, C: float = DEFAULT_C) -> LogisticRegression:
    if len(set(y)) < 2:
        raise ValueError("Need at least one like and one dislike to train")
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    with blas_quiet():
        model.fit(X, y)
    if not np.isfinite(model.coef_).all():
        raise FloatingPointError("Model weights are not finite")
    return model


def predict_like(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """P(like) for each row of X."""
    # predict_proba returns [P(dislike), P(like)] per row; pick the "like" column.
    like_column = list(model.classes_).index(1)
    with blas_quiet():
        p = model.predict_proba(X)[:, like_column]
    if not np.isfinite(p).all():
        raise FloatingPointError("Predictions are not finite")
    return p


def score_unrated(model: LogisticRegression, catalog: pd.DataFrame, ratings: pd.DataFrame,
                  embeddings: np.ndarray, index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicted like-probability for every catalog set you HAVEN'T rated.
    Returns the catalog rows plus a `score` column (0-1), best first.
    """
    rated = set(ratings["set_num"])
    candidates = catalog[~catalog["set_num"].isin(rated)].merge(index_df, on="set_num", how="inner")
    if candidates.empty:
        return candidates.assign(score=[])
    X = embeddings[candidates["embedding_row"].to_numpy()]
    candidates["score"] = predict_like(model, X)
    return (candidates.drop(columns="embedding_row")
            .sort_values(["score", "set_num"], ascending=[False, True])
            .reset_index(drop=True))


def recommend(catalog: pd.DataFrame, ratings: pd.DataFrame, embeddings: np.ndarray,
              index_df: pd.DataFrame, k: int = 5) -> Optional[pd.DataFrame]:
    """Top-k unrated sets by predicted score, or None if there aren't enough ratings."""
    if not check_readiness(ratings).ready:
        return None
    X, y = training_data(ratings, embeddings, index_df)
    if len(set(y)) < 2:
        return None
    model = train(X, y)
    return score_unrated(model, catalog, ratings, embeddings, index_df).head(k)

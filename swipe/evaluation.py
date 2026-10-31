"""
evaluation.py — Does the model actually beat the simple baseline? Precision@5.

The question: if we recommend 5 sets, how many of them would you like?
    Precision@5 = (liked sets among the top 5) / 5
    e.g. 3 liked out of 5 -> 0.6

The catch: we only know your opinion of sets you've RATED. So we test on
those, like a teacher hiding some exam answers:

    1. Split your ratings into TRAIN (70%) and TEST (30%).
    2. Each method sees ONLY the train ratings.
         model     trains logistic regression on train likes + dislikes
         baseline  averages the train likes into a taste vector (FAISS)
    3. Each method ranks the TEST sets (it never saw their labels).
    4. Look up your real labels for its top 5 -> Precision@5.

No data leakage: a test rating must never influence its own prediction.
That's why the split happens first and everything learned (model weights,
taste vector) comes from train only. `evaluate` checks the two halves don't
overlap. CLIP embeddings are fine to use for every set: CLIP was trained by
OpenAI and never saw your ratings.

One split is luck-heavy (with ~37 ratings the test half has ~12 sets), so we
repeat with 20 different random splits and average. Splits are "stratified":
each test half keeps roughly your like/dislike ratio.

Two reference points make the numbers meaningful:
    random  picking 5 test sets blindly; its expected score is simply the
            share of likes in the test half
    best    the score if a method ranked every test like first; with only
            2 likes in a test half, even a perfect method scores 2/5 = 0.4

About C: `compare_C` scores several regularisation strengths. Picking the
best C from that table and then quoting its score would be leakage of a
subtler kind (we'd have used the test halves to choose), so the headline
number always uses the fixed DEFAULT_C.
"""

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from swipe import recommender as rec
from swipe import similarity as sim

K = 5
N_SPLITS = 20
TEST_FRACTION = 0.3
MIN_EVAL_RATINGS = 20     # the test half needs at least K sets
MIN_EVAL_PER_CLASS = 5    # so both halves get some likes and some dislikes

MODEL, BASELINE, RANDOM, BEST = "model", "baseline", "random", "best"


@dataclass
class EvalResult:
    k: int
    n_ratings: int
    n_splits: int
    test_size: int
    scores: dict = field(default_factory=dict)   # method -> list of Precision@k per split

    def mean(self, method: str) -> float:
        return float(np.mean(self.scores[method]))

    def spread(self, method: str) -> float:
        # Spread between splits, not a true error bar: splits share ratings.
        return float(np.std(self.scores[method]))

    def head_to_head(self):
        """(wins, ties, losses) of the model against the baseline, split by split."""
        diff = np.array(self.scores[MODEL]) - np.array(self.scores[BASELINE])
        return int((diff > 1e-9).sum()), int((np.abs(diff) <= 1e-9).sum()), int((diff < -1e-9).sum())


def check_eval_readiness(ratings: pd.DataFrame) -> str:
    """Empty string if we can evaluate, otherwise what's missing."""
    likes = int((ratings["liked"] == 1).sum())
    dislikes = int((ratings["liked"] == 0).sum())
    needed = []
    if likes + dislikes < MIN_EVAL_RATINGS:
        needed.append(f"{MIN_EVAL_RATINGS - likes - dislikes} more ratings")
    if likes < MIN_EVAL_PER_CLASS:
        needed.append(f"{MIN_EVAL_PER_CLASS - likes} more likes")
    if dislikes < MIN_EVAL_PER_CLASS:
        needed.append(f"{MIN_EVAL_PER_CLASS - dislikes} more dislikes")
    return ("Need " + ", ".join(needed) + ".") if needed else ""


def precision_at_k(scores: np.ndarray, labels: np.ndarray, set_nums: np.ndarray, k: int) -> float:
    """Share of likes among the k highest-scored items (ties broken by set number)."""
    order = np.lexsort((set_nums, -scores))   # last key sorts first: score desc, then set_num
    return float(labels[order[:k]].mean())


def model_scores(train: pd.DataFrame, X_train: np.ndarray, X_test: np.ndarray, C: float) -> np.ndarray:
    model = rec.train(X_train, train["liked"].to_numpy(), C=C)
    return rec.predict_like(model, X_test)


def baseline_scores(train: pd.DataFrame, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    liked = X_train[train["liked"].to_numpy() == 1]
    taste = liked.mean(axis=0)
    taste /= np.linalg.norm(taste)
    # FAISS over the test sets only: it returns every test set with its cosine similarity.
    index = sim.build_index(X_test)
    sims, rows = index.search(taste.reshape(1, -1).astype(np.float32), len(X_test))
    scores = np.empty(len(X_test), dtype=np.float32)
    scores[rows[0]] = sims[0]
    return scores


def _splits(ratings: pd.DataFrame, n_splits: int, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=TEST_FRACTION, random_state=seed)
    return splitter.split(np.zeros(len(ratings)), ratings["liked"].to_numpy())


def _prepare(ratings: pd.DataFrame, embeddings: np.ndarray, index_df: pd.DataFrame):
    rows = ratings.merge(index_df, on="set_num", how="inner").reset_index(drop=True)
    rows["liked"] = rows["liked"].astype(int)
    if check_eval_readiness(rows):
        raise ValueError(check_eval_readiness(rows))
    return rows, embeddings[rows["embedding_row"].to_numpy()]


def evaluate(ratings: pd.DataFrame, embeddings: np.ndarray, index_df: pd.DataFrame,
             k: int = K, n_splits: int = N_SPLITS, C: float = rec.DEFAULT_C,
             seed: int = 0) -> EvalResult:
    """Precision@k of the model, the FAISS baseline, random and best, over repeated splits."""
    rows, X = _prepare(ratings, embeddings, index_df)
    result = EvalResult(k=k, n_ratings=len(rows), n_splits=n_splits, test_size=0,
                        scores={MODEL: [], BASELINE: [], RANDOM: [], BEST: []})
    for train_idx, test_idx in _splits(rows, n_splits, seed):
        train, test = rows.iloc[train_idx], rows.iloc[test_idx]
        assert not set(train["set_num"]) & set(test["set_num"]), "train/test overlap"
        labels, set_nums = test["liked"].to_numpy(), test["set_num"].to_numpy()

        m = model_scores(train, X[train_idx], X[test_idx], C)
        b = baseline_scores(train, X[train_idx], X[test_idx])
        result.scores[MODEL].append(precision_at_k(m, labels, set_nums, k))
        result.scores[BASELINE].append(precision_at_k(b, labels, set_nums, k))
        result.scores[RANDOM].append(float(labels.mean()))
        result.scores[BEST].append(min(int(labels.sum()), k) / k)
        result.test_size = len(test)
    return result


def compare_C(ratings: pd.DataFrame, embeddings: np.ndarray, index_df: pd.DataFrame,
              Cs: Iterable[float] = (0.1, 1.0, 10.0, 100.0), k: int = K,
              n_splits: int = N_SPLITS, seed: int = 0) -> pd.DataFrame:
    """Model Precision@k for several C values (same splits for each). For exploration only."""
    rows, X = _prepare(ratings, embeddings, index_df)
    splits = list(_splits(rows, n_splits, seed))
    out: List[dict] = []
    for C in Cs:
        scores = []
        for train_idx, test_idx in splits:
            test = rows.iloc[test_idx]
            m = model_scores(rows.iloc[train_idx], X[train_idx], X[test_idx], C)
            scores.append(precision_at_k(m, test["liked"].to_numpy(), test["set_num"].to_numpy(), k))
        out.append({"C": C, "precision": float(np.mean(scores))})
    return pd.DataFrame(out)

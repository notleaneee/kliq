"""
embeddings.py — CLIP image embedding extraction and caching.

What is CLIP?
  CLIP (Contrastive Language-Image Pretraining) is a neural network trained by
  OpenAI on 400 million image-text pairs. Given an image, it outputs a 512-number
  vector (embedding) that captures the visual "meaning" of the image. Sets that
  look similar (e.g. two space-themed sets) will have embeddings that are close
  together in this 512-dimensional space. That closeness is what powers our
  recommendations.

What we store on disk:
  data/embeddings.npy       — float32 array of shape (N, 512), one row per set
  data/embeddings_index.csv — maps each row index -> set_num, so we can look up
                              "which row belongs to set 75192-1?"

Why cache?
  Running CLIP on 10,000 images takes ~10 minutes on CPU. We never want to
  redo that work every time the app starts, so we save the results once.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# PyTorch and transformers are imported inside generate_embeddings(): they
# take seconds to load, and the app only needs to READ the cached embeddings.


def _data_dir() -> Path:
    # Overridable so tests can use small fake embeddings.
    return Path(os.environ.get("LEGO_EMBEDDINGS_DIR", "data"))


def _embeddings_path() -> Path:
    return _data_dir() / "embeddings.npy"


def _index_path() -> Path:
    return _data_dir() / "embeddings_index.csv"

# The CLIP variant we use. clip-vit-base-patch32 gives 512-dim embeddings and
# is fast enough on CPU. Larger variants (patch16, large) give slightly better
# quality but take much longer to run.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def _get_device():
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_embeddings(
    df: pd.DataFrame,
    batch_size: int = 32,
    force: bool = False,
) -> tuple:
    """
    Run CLIP on set images in df and save the results to disk.

    Incremental: sets that already have a cached embedding are skipped, and
    new embeddings are appended to the cache. So you can grow the catalog
    later without re-processing everything.

    Parameters
    ----------
    df         : DataFrame from data_loader.load_sets(), must have 'local_image_path'
                 and 'set_num' columns. Rows where local_image_path is None are skipped.
    batch_size : How many images to send to CLIP at once. Larger = faster but more RAM.
                 32 is a safe default for most machines.
    force      : If True, throw away the cache and re-embed everything in df.

    Returns
    -------
    (embeddings, index_df) covering every cached set (old + new)
      embeddings : np.ndarray of shape (N, 512), float32
      index_df   : pd.DataFrame with columns [set_num, embedding_row]
                   embedding_row is the row index in the embeddings array
    """
    valid = df[df["local_image_path"].notna()].copy()
    if valid.empty:
        raise ValueError(
            "No local images found. Run download_images_batch() before generate_embeddings()."
        )

    if not force and _embeddings_path().exists() and _index_path().exists():
        old_embeddings, old_index = load_embeddings()
        valid = valid[~valid["set_num"].isin(old_index["set_num"])]
        if valid.empty:
            print(f"All requested sets already embedded ({len(old_index):,} cached).")
            return old_embeddings, old_index
        print(f"{len(old_index):,} embeddings cached; {len(valid):,} new sets to embed.")
    else:
        old_embeddings = np.empty((0, 512), dtype=np.float32)
        old_index = pd.DataFrame({"set_num": [], "embedding_row": []})

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = _get_device()
    print(f"Loading CLIP model ({CLIP_MODEL_NAME}) on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()   # disable dropout — we're doing inference, not training

    all_embeddings = []
    set_nums = []

    print(f"Generating embeddings for {len(valid):,} images (batch_size={batch_size})...")
    rows = valid.to_dict("records")

    for i in tqdm(range(0, len(rows), batch_size), unit="batch"):
        batch = rows[i : i + batch_size]
        images = []
        valid_batch_nums = []

        for row in batch:
            try:
                img = Image.open(row["local_image_path"]).convert("RGB")
                images.append(img)
                valid_batch_nums.append(row["set_num"])
            except Exception as e:
                print(f"  [warn] Skipping {row['set_num']}: {e}")

        if not images:
            continue

        # CLIPProcessor resizes + normalises images into the tensor format CLIP expects.
        # return_tensors="pt" means "return PyTorch tensors".
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # get_image_features() runs the image encoder half of CLIP.
            # The output shape is (batch_size, 512).
            features = model.get_image_features(**inputs)

            # L2-normalise so that dot-product == cosine similarity.
            # This makes FAISS distance comparisons more meaningful.
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())
        set_nums.extend(valid_batch_nums)

    if not all_embeddings:
        print("No new images could be embedded.")
        return old_embeddings, old_index

    embeddings = np.vstack([old_embeddings] + all_embeddings).astype(np.float32)

    # Build the index: row position in embeddings array -> set_num
    all_set_nums = list(old_index["set_num"]) + set_nums
    index_df = pd.DataFrame({
        "set_num": all_set_nums,
        "embedding_row": range(len(all_set_nums)),
    })

    # Save to disk
    _data_dir().mkdir(parents=True, exist_ok=True)
    np.save(_embeddings_path(), embeddings)
    index_df.to_csv(_index_path(), index=False)

    print(f"Saved {embeddings.shape[0]:,} embeddings → {_embeddings_path()}")
    print(f"Saved index → {_index_path()}")
    return embeddings, index_df


def load_embeddings() -> tuple:
    """
    Load cached embeddings and index from disk.

    Returns
    -------
    (embeddings, index_df)
      embeddings : np.ndarray shape (N, 512)
      index_df   : pd.DataFrame with columns [set_num, embedding_row]

    Raises FileNotFoundError if the cache files don't exist yet.
    """
    if not _embeddings_path().exists() or not _index_path().exists():
        raise FileNotFoundError(
            "Embeddings not found. Build them with:  python build_swipe_catalog.py"
        )

    rebuild = " Rebuild with:  python build_swipe_catalog.py"
    try:
        embeddings = np.load(_embeddings_path())
        index_df = pd.read_csv(_index_path(), dtype={"set_num": str})
    except (ValueError, OSError, pd.errors.ParserError) as e:
        raise ValueError(f"The embeddings cache can't be read ({e})." + rebuild) from e

    # Catch a half-written or mismatched cache here, with a clear message,
    # instead of as a confusing IndexError deep inside the model.
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("The embeddings cache is corrupted." + rebuild)
    if not {"set_num", "embedding_row"} <= set(index_df.columns):
        raise ValueError(f"{_index_path()} is missing columns." + rebuild)
    rows = index_df["embedding_row"]
    if index_df.empty or rows.min() < 0 or rows.max() >= len(embeddings):
        raise ValueError("The embeddings index doesn't match the embeddings file." + rebuild)
    return embeddings, index_df


def get_embedding_for_set(set_num: str, embeddings: np.ndarray, index_df: pd.DataFrame) -> np.ndarray:
    """
    Look up the embedding vector for a single set by its set_num.

    Returns a 1D array of shape (512,), or raises KeyError if not found.
    """
    row = index_df[index_df["set_num"] == set_num]
    if row.empty:
        raise KeyError(f"No embedding found for set_num='{set_num}'")
    idx = int(row.iloc[0]["embedding_row"])
    return embeddings[idx]

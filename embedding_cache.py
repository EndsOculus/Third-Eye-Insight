import hashlib
import logging
import os
import json
import tempfile
import re

import numpy as np

logger = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_CACHE_DIR = os.path.join("output", "cache", "embeddings")


def text_cache_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_embedding_cache_path(model_name):
    safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return os.path.join(EMBEDDING_CACHE_DIR, f"{safe_model_name}.json")


def load_embedding_cache(cache_path, model_name):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            cache = json.load(handle)
        if isinstance(cache, dict):
            if cache.get("__version__") != model_name:
                logger.warning(f"embedding 缓存版本不匹配，忽略旧缓存：{cache_path}")
                return {}
            logger.info(f"已加载 embedding 缓存：{cache_path}（{len(cache) - 1} 条）")
            return {k: (v if k == "__version__" else np.asarray(v, dtype=np.float32)) for k, v in cache.items()}
    except Exception as exc:
        logger.warning(f"读取 embedding 缓存失败，将忽略旧缓存：{exc}")
    return {}


def save_embedding_cache(cache_path, cache, model_name):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache["__version__"] = model_name
    data = {k: (v if k == "__version__" else np.asarray(v).tolist()) for k, v in cache.items()}
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(cache_path), delete=False) as handle:
            temp_path = handle.name
            json.dump(data, handle)
        os.replace(temp_path, cache_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def purge_stale_caches(model_name):
    # Old pickle caches are ignored, never deserialized or deleted.
    return None


def load_sentence_transformer(model_name, device=None):
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if device is not None:
        kwargs["device"] = str(device)
    return SentenceTransformer(model_name, local_files_only=True, token=False, trust_remote_code=False, **kwargs)


def encode_batch(texts, model, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)


def encode_texts_parallel(texts, model_name, batch_size=32, num_workers=4):
    # One model, batched inference: avoids loading N CPU models into memory.
    if not texts:
        return np.empty((0,))
    model = load_sentence_transformer(model_name, device="cpu")
    return encode_batch(texts, model, batch_size=batch_size)


def encode_missing_texts(texts, model_name, batch_size=32, num_workers=4, device="cpu"):
    if not texts:
        return np.empty((0,))
    if str(device) != "cpu":
        model = load_sentence_transformer(model_name, device=device)
        return model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return encode_texts_parallel(texts, model_name, batch_size=batch_size, num_workers=num_workers)


def attach_text_embeddings(chat_df, model_name=DEFAULT_MODEL_NAME, batch_size=32, num_workers=4, device="cpu"):
    purge_stale_caches(model_name)

    chat_df = chat_df.copy()
    chat_df["content"] = chat_df["content"].fillna("").astype(str)
    cache_path = get_embedding_cache_path(model_name)
    cache = load_embedding_cache(cache_path, model_name)

    unique_texts = list(dict.fromkeys(chat_df["content"].tolist()))
    missing_texts = [text for text in unique_texts if text_cache_key(text) not in cache]
    if missing_texts:
        logger.info(f"embedding 缓存未命中 {len(missing_texts)} 条文本，开始增量编码...")
        new_embeddings = encode_missing_texts(
            missing_texts,
            model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        for text, embedding in zip(missing_texts, new_embeddings):
            cache[text_cache_key(text)] = np.asarray(embedding, dtype=np.float32)
        save_embedding_cache(cache_path, cache, model_name)
    else:
        logger.info("embedding 全部命中缓存，跳过模型编码。")

    chat_df["text_embedding"] = [cache[text_cache_key(text)] for text in chat_df["content"]]
    return chat_df

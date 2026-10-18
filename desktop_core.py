"""Offline desktop analysis boundary. Never calls reporting/API code."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import logging
import os
import time

class InputError(ValueError):
    """Safe user-facing validation error, contains no source values."""

MAX_ROWS = 50000
MAX_USERS = 300

@dataclass(frozen=True)
class Request:
    db: str = ""
    identifier: str = ""
    mode: str = "group"
    start: str = ""
    end: str = ""
    focus: str = ""
    exclude: str = ""
    encrypted: bool = False
    gpu: bool = False
    auto_weight: bool = False
    demo: bool = False

    def validate(self):
        if self.mode not in ("group", "c2c"):
            raise InputError("请选择群聊或私聊模式。")
        if not self.demo:
            if not Path(self.db).is_file():
                raise InputError("请选择现有数据库文件。原始文件只读打开。")
            if not self.identifier.isascii() or not self.identifier.isdigit():
                raise InputError("群号 / 我的 QQ 必须是数字。")
        for value in [self.focus] + self.exclude.split(","):
            if value.strip() and not (value.strip().isascii() and value.strip().isdigit()):
                raise InputError("聚焦与排除账号请使用数字，排除账号以英文逗号分隔。")
        try:
            for value in (self.start, self.end):
                if value:
                    date.fromisoformat(value)
        except ValueError:
            raise InputError("日期格式应为 YYYY-MM-DD。") from None
        if self.start and self.end and self.start > self.end:
            raise InputError("开始日期不能晚于结束日期。")

def synthetic_frame(rows=1200, users=12):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "sender_id": [str(20000+i % users) for i in range(rows)],
        "peer_id": [str(20001 + (i // 2) % max(1, users-1)) for i in range(rows)],
        "sender_nickname": [f"演示成员 {i % users + 1:02d}" for i in range(rows)],
        "content": ["synthetic hello question answer " + str(i % 5) for i in range(rows)],
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="30s"),
        "text_embedding": list(rng.normal(size=(rows, 16)).astype("float32")),
    })

def analyze(request, progress=lambda value, text: None):
    request.validate()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    local_hub = Path(__file__).resolve().parent / ".cache" / "huggingface"
    if local_hub.is_dir():
        os.environ.setdefault("HF_HOME", str(local_hub))
    started = time.perf_counter()
    progress(5, "加载本地分析组件…")
    import pandas as pd
    from types import SimpleNamespace
    from analysis_types import AnalysisParams
    from metrics import compute_metrics, compute_final_intimacy
    identifier = int(request.identifier or 20000)
    progress(15, "只读提取数据…")
    if request.demo:
        frame = synthetic_frame()
    else:
        from extract_chat_data import extract_chat_data
        config = {}
        if request.encrypted:
            import json
            with open("config.json", encoding="utf-8") as handle:
                config = json.load(handle)
            config["encrypted"] = True
        frame = extract_chat_data(request.db, identifier, mode=request.mode,
                                  cipher_config=config, max_rows=MAX_ROWS)
    if frame.attrs.get("truncated", False) or len(frame) > MAX_ROWS:
        raise InputError("数据超过桌面分析上限 50,000 条；请使用较小的只读数据副本或 CLI。")
    if frame.empty:
        raise InputError("没有可分析的记录。请检查数据库格式、模式、账号及加密设置。")
    if request.start:
        frame = frame[frame.timestamp >= pd.Timestamp(request.start)]
    if request.end:
        frame = frame[frame.timestamp < pd.Timestamp(request.end) + pd.Timedelta(days=1)]
    excluded = {v.strip() for v in request.exclude.split(",") if v.strip()}
    frame = frame[~frame.sender_id.astype(str).isin(excluded)].dropna(subset=["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    users = list(frame.sender_id.astype(str).unique())
    if len(users) < 2:
        raise InputError("筛选后不足两名成员，请放宽筛选条件。")
    if len(users) > MAX_USERS:
        raise InputError("桌面分析最多支持 300 名成员，请缩小数据范围。")
    if request.focus and request.focus not in users:
        raise InputError("聚焦账号不在筛选后的数据中。")
    progress(30, "准备本地语义向量…")
    # Both modes use embeddings: private replies combine cosine similarity and lexical overlap.
    if not request.demo:
        from embedding_cache import attach_text_embeddings
        frame = attach_text_embeddings(frame, device="cuda" if request.gpu else "cpu", num_workers=1)
    progress(55, "计算互动、语义与关系指标…")
    args = SimpleNamespace(mode=request.mode, auto_weight=request.auto_weight)
    indexes = {u: i for i, u in enumerate(users)}
    metrics = compute_metrics(frame, args, identifier, users, indexes, AnalysisParams())
    scores, weights = compute_final_intimacy(args, metrics)
    progress(90, "整理关系视图…")
    # No account identifiers, message content, or actual names cross into the UI.
    pairs = []
    for i, u in enumerate(users):
        for j in range(i+1, len(users)):
            if request.focus and request.focus not in (u, users[j]):
                continue
            if request.mode == "c2c" and str(identifier) not in (u, users[j]):
                continue
            pairs.append((f"成员 {i+1:03d}", f"成员 {j+1:03d}", float(scores[i,j]),
                          float(metrics.semantic_matrix_mapped[i,j]), float(metrics.behavior_norm[i,j]),
                          float(metrics.third_matrix[i,j])))
    pairs.sort(key=lambda row: row[2], reverse=True)
    return {"rows": len(frame), "users": len(users), "pairs": pairs,
            "seconds": round(time.perf_counter()-started, 3), "demo": request.demo}

def worker(request, events):
    # Existing CLI libraries can log identifiers; suppress them in this process.
    logging.disable(logging.CRITICAL)
    import sys
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    try:
        result = analyze(request, lambda n, text: events.put(("progress", (n, text))))
        events.put(("result", result))
    except InputError as exc:
        events.put(("error", str(exc)))
    except Exception:
        events.put(("error", "分析未完成，请检查数据库格式、加密配置及本地模型是否可用。"))

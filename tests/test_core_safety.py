import hashlib
import sqlite3
import numpy as np
import pytest
from embedding_cache import load_embedding_cache, save_embedding_cache, encode_texts_parallel
from extract_chat_data import extract_chat_data

def test_cache_roundtrip_no_pickle(tmp_path):
    path = tmp_path / "cache.json"
    save_embedding_cache(str(path), {"key":np.array([1., 2.])}, "model")
    assert np.array_equal(load_embedding_cache(str(path), "model")["key"], [1,2])
    path.write_bytes(b"not json")
    assert load_embedding_cache(str(path), "model") == {}

def test_cpu_model_loaded_once(monkeypatch):
    import embedding_cache
    calls=[]
    class Model:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3))
    monkeypatch.setattr(embedding_cache, "load_sentence_transformer", lambda *a, **k: calls.append(1) or Model())
    assert encode_texts_parallel(["a", "b"], "mock", num_workers=4).shape == (2,3)
    assert len(calls) == 1

def test_database_readonly_and_limit(tmp_path, monkeypatch):
    import extract_chat_data as extraction
    path=tmp_path / "中文 space.db"
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE group_msg_table ("40033" TEXT, "40090" TEXT, "40093" TEXT, "40800" TEXT, "40050" INTEGER, "40027" INTEGER, "40011" INTEGER, "40012" INTEGER)')
        conn.executemany('INSERT INTO group_msg_table VALUES (?,?,?,?,?,?,?,?)', [(str(20000+i%3),"Synthetic","Synthetic","hello",1767225600+i,12345,2,1) for i in range(100)])
    conn.close()
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    original=sqlite3.connect
    def checked(*args, **kwargs):
        conn=original(*args, **kwargs)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute('CREATE TABLE forbidden (x)')
        return conn
    monkeypatch.setattr(extraction.sqlite3, "connect", checked)
    frame=extract_chat_data(str(path),12345,cipher_config={},max_rows=10)
    assert len(frame)==11
    assert frame.attrs["truncated"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    missing=tmp_path / "missing.db"
    assert extract_chat_data(str(missing),12345,cipher_config={}).empty
    assert not missing.exists()

def test_system_rows_cannot_hide_truncation(tmp_path):
    path=tmp_path / "synthetic.db"
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE group_msg_table ("40033" TEXT, "40090" TEXT, "40093" TEXT, "40800" TEXT, "40050" INTEGER, "40027" INTEGER, "40011" INTEGER, "40012" INTEGER)')
        conn.executemany('INSERT INTO group_msg_table VALUES (?,?,?,?,?,?,?,?)', [("10000","Synthetic","Synthetic","hello",1767225600+i,12345,2,1) for i in range(20)])
    conn.close()
    result=extract_chat_data(str(path),12345,cipher_config={},max_rows=10)
    assert result.empty
    assert result.attrs["truncated"]

def test_c2c_sqlite_desktop_encodes_extracted_messages(tmp_path, monkeypatch):
    from desktop_core import Request, analyze
    import embedding_cache
    path = tmp_path / "私聊 synthetic space.db"
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE c2c_msg_table ("40033" TEXT, "40030" TEXT, "40093" TEXT, "40800" TEXT, "40050" INTEGER, "40011" INTEGER, "40012" INTEGER)')
        conn.executemany('INSERT INTO c2c_msg_table VALUES (?,?,?,?,?,?,?)', [
            (str(20000+i%2), "20001", "Synthetic", "synthetic reply " + str(i%3), 1767225600+i*30, 2, 1)
            for i in range(12)
        ])
    conn.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    extracted = extract_chat_data(str(path), 20000, mode="c2c", cipher_config={})
    assert len(extracted) == 12
    assert set(extracted.sender_id) == {"20000", "20001"}
    assert "text_embedding" not in extracted.columns
    calls = []
    class Model:
        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return np.tile(np.array([1., 0., 1.], dtype=np.float32), (len(texts), 1))
    monkeypatch.setattr(embedding_cache, "load_sentence_transformer", lambda *a, **k: Model())
    monkeypatch.setattr(embedding_cache, "EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
    result = analyze(Request(db=str(path), identifier="20000", mode="c2c"))
    assert result["rows"] == 12 and len(result["pairs"]) == 1
    assert calls  # Actual attach_text_embeddings runs; no prefilled frame column.
    assert result["pairs"][0][3] > .75
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

@pytest.mark.parametrize("peers", [(20001,), (20001, 20002)])
def test_c2c_query_parameter_order_and_names(tmp_path, peers):
    path = tmp_path / "private synthetic.db"
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE c2c_msg_table ("40033" TEXT, "40030" TEXT, "40093" TEXT, "40800" TEXT, "40050" INTEGER, "40011" INTEGER, "40012" INTEGER)')
        rows = []
        for peer in peers:
            rows.extend([(20000, peer, "SyntheticSelf", "hello", 1767225600, 2, 1),
                         (peer, peer, "SyntheticPeer", "reply", 1767225630, 2, 1)])
        rows.append((30001, 30001, "Unrelated", "unrelated", 1767225660, 2, 1))
        conn.executemany('INSERT INTO c2c_msg_table VALUES (?,?,?,?,?,?,?)', rows)
    conn.close()
    result = extract_chat_data(str(path), 20000, mode="c2c", cipher_config={})
    assert len(result) == len(peers)*2
    assert set(result.peer_id.astype(str)) == set(map(str, peers))
    assert set(result.sender_id) == {"20000", *map(str, peers)}
    assert set(result.loc[result.sender_id == "20000", "sender_nickname"]) == {"我"}
    assert set(result.loc[result.sender_id != "20000", "sender_nickname"]) == {"SyntheticPeer"}
    assert "text_embedding" not in result.columns

def test_group_without_observed_interactions_has_zero_scores():
    import pandas as pd
    from types import SimpleNamespace
    from analysis_types import AnalysisParams
    from metrics import compute_metrics, compute_final_intimacy
    frame = pd.DataFrame({"sender_id": ["1", "2", "3"],
                          "timestamp": pd.date_range("2026-01-01", periods=3, freq="601s"),
                          "text_embedding": [np.array([1.,0.])] * 3})
    args = SimpleNamespace(mode="group", auto_weight=False)
    users = ["1", "2", "3"]
    for _ in range(3):
        artifacts = compute_metrics(frame, args, 0, users, {u:i for i,u in enumerate(users)}, AnalysisParams())
        scores, _ = compute_final_intimacy(args, artifacts)
        assert np.count_nonzero(artifacts.semantic_matrix_mapped) == 0
        assert np.count_nonzero(scores) == 0

def test_semantic_ties_are_deterministic():
    from metrics import discrete_mapping
    matrix = np.ones((4,4)) - np.eye(4)
    first = discrete_mapping(matrix)
    for _ in range(10):
        assert np.array_equal(discrete_mapping(matrix), first)
    assert len(np.unique(first[np.triu_indices(4, 1)])) == 1

def test_semantic_mapping_distinguishes_zero_observation_from_zero_cosine():
    from metrics import discrete_mapping
    matrix = np.zeros((3,3))
    mask = np.zeros((3,3), dtype=bool)
    mask[0,1] = mask[1,0] = True
    scores = discrete_mapping(matrix, observed=mask)
    assert scores[0,1] > 0
    assert scores[0,2] == scores[1,2] == 0
    assert np.array_equal(scores, scores.T)
    assert not np.any(np.diag(scores))

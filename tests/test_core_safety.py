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

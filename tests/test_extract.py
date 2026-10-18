import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extract_chat_data import clean_message, _rank_name_candidates, _build_local_query


def test_clean_message_none():
    assert clean_message(None) == ""


def test_clean_message_empty():
    assert clean_message("") == ""


def test_clean_message_chinese():
    result = clean_message("你好世界")
    assert "你好世界" in result


def test_clean_message_special_chars():
    result = clean_message("hello@世界#123")
    assert "@" not in result
    assert "#" not in result
    assert "hello" in result


def test_rank_name_candidates_empty():
    assert _rank_name_candidates([]) is None


def test_rank_name_candidates_digits_only():
    assert _rank_name_candidates([("123", 5)]) is None


def test_rank_name_candidates_returns_highest_count():
    result = _rank_name_candidates([("Alice", 3), ("Bob", 5)])
    assert result == "Bob"


def test_build_local_query_group_contains_correct_field():
    query, params = _build_local_query("group", 12345, encrypted=True)
    assert "40800" in query


def test_build_local_query_group_no_wrong_field():
    query, params = _build_local_query("group", 12345, encrypted=True)
    assert '"40080"' not in query


def test_build_local_query_group_contains_group_field():
    query, params = _build_local_query("group", 12345, encrypted=True)
    assert "40027" in query


def test_build_local_query_group_params_empty():
    query, params = _build_local_query("group", 12345, encrypted=True)
    assert params == ()

import copy
import json
import sqlite3

import pytest

from desktop_core import Request, _analyze, analyze
from interaction_data import decode_payload, extract_interactions
from interaction_v2 import analyze_records
from streaming_store import Lookup, RecordStore
from tests.interaction_fixtures import make_db, msg


def run(rows, mode, streamed, tmp_path):
    rows = copy.deepcopy(rows)
    for i, row in enumerate(rows):
        row['order'] = i
    if not streamed:
        return analyze_records(rows, mode=mode)
    with RecordStore(tmp_path, mode=mode) as store:
        for row in rows:
            store.append(row)
        store.prepare()
        return analyze_records(store, mode=mode)


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('text', ['x' * 8150, 'x' * 8192, '\u4e2d' * 2730 + 'xx'], ids=['8150', '8192', 'utf8-8192'])
def test_separate_import_fields_keep_reply_at_byte_boundary(tmp_path, streamed, text):
    path = tmp_path / 'synthetic.sqlite'
    make_db(path, [msg(1, 0, 'a'), msg(2, 10, 'b', reply='a', text=text)])
    if streamed:
        with RecordStore(tmp_path) as store:
            extract_interactions(path, 99, source_format='tei', sink=store)
            store.prepare()
            result = analyze_records(store)
    else:
        rows, _ = extract_interactions(path, 99, source_format='tei')
        assert rows[1]['text'] == text and rows[1]['structure'] == 'known'
        result = analyze_records(rows)
    assert result['pairs'][0]['reply_links'] == 1


@pytest.mark.parametrize('streamed', [False, True])
def test_private_conflict_missing_side_and_other_conversation(tmp_path, streamed):
    rows = [msg(1, 0, 'a'), msg(2, 10, 'bad'), msg(2, 11, 'bad'),
            msg(1, 20, 'c', peer='3', thread='other')]
    result = run(rows, 'c2c', streamed, tmp_path)
    pairs = {(p['a'], p['b']): p for p in result['pairs']}
    assert pairs['1', '2']['evidence'] == 'coverage_unknown'
    assert pairs['1', '3']['evidence'] == 'no_observed_response'


@pytest.mark.parametrize('streamed', [False, True])
def test_private_entire_thread_conflicted_still_reports_missing_data(tmp_path, streamed):
    result = run([msg(2, 0, 'bad'), msg(2, 1, 'bad')], 'c2c', streamed, tmp_path)
    assert len(result['pairs']) == 1
    assert result['pairs'][0]['evidence'] == 'coverage_unknown'
    assert result['pairs'][0]['messages_b'] == 0


@pytest.mark.parametrize('streamed', [False, True])
def test_group_conflict_sender_in_other_thread_does_not_taint_clean_pair(tmp_path, streamed):
    rows = [msg(1, 0, 'a'), msg(2, 10, 'b', reply='a'),
            msg(1, 20, 'bad', thread='other'), msg(3, 21, 'bad', thread='other')]
    result = run(rows, 'group', streamed, tmp_path)
    pair = next(p for p in result['pairs'] if (p['a'], p['b']) == ('1', '2'))
    assert pair['evidence'] == 'insufficient'


@pytest.mark.parametrize('streamed', [False, True])
def test_duplicate_source_alias_names_the_retained_row(tmp_path, streamed):
    path = tmp_path / 'synthetic.sqlite'
    source = msg(1, 0, 'a')
    make_db(path, [source, msg(2, 10, 'b', reply='a'), source])
    request = Request(db=str(path), identifier='99', source_format='tei')
    result = analyze(request, scratch_dir=tmp_path) if streamed else _analyze(request, lambda *args: None, tmp_path)
    example = result['pairs'][0]['evidence_examples'][0]
    assert example['message_id'] == 'R000002'
    assert example['source_message_id'] == 'R000001'


def test_aliases_remove_conflicts_and_keep_conversation_namespace(tmp_path):
    with RecordStore(tmp_path) as store:
        for i, row in enumerate([msg(1, 0, 'a'), msg(2, 1, 'a'),
                                 msg(3, 2, 'a', thread='other')]):
            store.append(dict(row, order=i))
        store.prepare()
        assert Lookup(store, True).get(('g', 'a')) is None
        assert Lookup(store, True).get(('other', 'a')) == 'R000003'


@pytest.mark.parametrize('field', ['content', 'metadata'])
@pytest.mark.parametrize('size', [8192, 8193])
def test_import_fields_have_independent_byte_limits(tmp_path, field, size):
    path = tmp_path / 'synthetic.sqlite'
    make_db(path, [msg(1, 0, 'a')])
    value = 'x' * size
    if field == 'metadata':
        base = json.dumps(dict(schema='tei.message.v1', mentions=['2']))
        value = base + ' ' * (size - len(base))
    with sqlite3.connect(path) as db:
        db.execute(f'UPDATE tei_messages_v1 SET {field}=?', (value,))
    rows, _ = extract_interactions(path, 99, source_format='tei')
    assert rows[0]['structure'] == ('known' if size == 8192 else 'oversize')


@pytest.mark.parametrize('metadata', ['{', '{"schema":"tei.message.v1","mentions":[2]}'])
def test_invalid_metadata_preserves_body_and_no_targets(tmp_path, metadata):
    path = tmp_path / 'synthetic.sqlite'
    make_db(path, [msg(1, 0, 'a', text='synthetic body')])
    with sqlite3.connect(path) as db:
        db.execute('UPDATE tei_messages_v1 SET metadata=?', (metadata,))
    rows, _ = extract_interactions(path, 99, source_format='tei')
    assert rows[0]['text'] == 'synthetic body'
    assert rows[0]['structure'] == 'invalid' and rows[0]['mentions'] == []


def test_invalid_envelope_text_preserves_original_string():
    raw = json.dumps(dict(schema='tei.message.v1', text=42))
    decoded = decode_payload(raw, allow_structure=True)
    assert decoded['structure'] == 'invalid' and decoded['text'] == raw


def test_metadata_cannot_replace_separate_body(tmp_path):
    path = tmp_path / 'synthetic.sqlite'
    make_db(path, [msg(1, 0, 'a', text='synthetic body')])
    with sqlite3.connect(path) as db:
        db.execute('UPDATE tei_messages_v1 SET metadata=?',
                   (json.dumps(dict(schema='tei.message.v1', text=42, mentions=['2'])),))
    rows, _ = extract_interactions(path, 99, source_format='tei')
    assert rows[0]['text'] == 'synthetic body'
    assert rows[0]['structure'] == 'known' and rows[0]['mentions'] == ['2']

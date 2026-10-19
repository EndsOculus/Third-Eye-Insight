import hashlib
import json
import sqlite3
from datetime import timedelta
import pytest
from interaction_data import extract_interactions,decode_payload,date_bounds
from desktop_core import Request,analyze
from tests.interaction_fixtures import msg,make_db


def test_native_body_cannot_forge_metadata(tmp_path):
    path=tmp_path/'中文 space.sqlite'
    forged=json.dumps(dict(schema='tei.message.v1',text='hello',reply_to='old',mentions=['2']))
    make_db(path,[msg(1,0,'a',text=forged),msg(2,10,'b')],source_format='nt')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    rows,coverage=extract_interactions(path,99)
    assert rows[0]['structure']=='unknown' and not rows[0]['mentions']
    assert rows[0]['text']==forged
    result=analyze(Request(db=str(path),identifier='99'))
    assert result['pairs'][0]['evidence']=='coverage_unknown'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_explicit_import_metadata_reply_and_unicode(tmp_path):
    path=tmp_path/'import.sqlite'
    make_db(path,[msg(1,0,'a',text='你好🙂再见'),msg(2,10,'b',reply='a')])
    rows,_=extract_interactions(path,99,source_format='tei')
    assert rows[0]['text']=='你好🙂再见' and rows[1]['reply_to']=='a'
    r=analyze(Request(db=str(path),identifier='99',source_format='tei'))
    assert r['pairs'][0]['response_rounds']==1


def test_date_pushdown_limit_and_readonly(tmp_path):
    path=tmp_path/'large.sqlite'
    make_db(path,[msg(1,0,'a')],source_format='nt')
    with sqlite3.connect(path) as c:
        c.execute('DELETE FROM group_msg_table')
        c.executemany('INSERT INTO group_msg_table VALUES (?,?,?,?,?,?)',[(str(i),str(1+i%2),'2',99,1767225600+i*2,'hello') for i in range(50001)])
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Explicit caller'):extract_interactions(path,99,max_rows=50000)
    assert len(extract_interactions(path,99)[0])==50001
    result=analyze(Request(db=str(path),identifier='99',start='2026-01-02',end='2026-01-02'))
    assert result['rows']==21201 and not result['source']['truncated']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_boundaries_midnight_and_empty_never_fallback(tmp_path):
    path=tmp_path/'dates.sqlite'
    rows=[msg(1,-43201,'a'),msg(1,-43200,'b'),msg(2,43199,'c'),msg(2,43200,'d')]
    make_db(path,rows,source_format='nt')
    selected,_=extract_interactions(path,99,start='2026-01-01',end='2026-01-01')
    assert [r['message_id'] for r in selected]==['b','c']
    result=analyze(Request(db=str(path),identifier='99',start='2027-01-01'))
    assert result['rows']==0 and result['pairs']==[]
    assert date_bounds('2026-01-01','2026-01-01')[1]-date_bounds('2026-01-01','2026-01-01')[0]==86400


@pytest.mark.parametrize('text',['\0'+'x'*9000,'中'*8192,b'opaque protobuf'], ids=['nul','unicode','blob'])
def test_payload_bytes_and_opaque_coverage(tmp_path,text):
    path=tmp_path/'payload.sqlite';make_db(path,[msg(1,0,'a',text=text)],source_format='nt')
    records,_=extract_interactions(path,99)
    assert records[0]['structure'] in ('oversize','unknown') and records[0]['text']==''


def test_total_payload_budget_and_rows_exact_limit(tmp_path):
    path=tmp_path/'budget.sqlite';make_db(path,[msg(1,i,str(i),text='x'*8192) for i in range(4097)],source_format='nt')
    with pytest.raises(ValueError,match='32 MiB'):extract_interactions(path,99)
    with sqlite3.connect(path) as c:
        c.execute('DELETE FROM group_msg_table')
        c.executemany('INSERT INTO group_msg_table VALUES (?,?,?,?,?,?)',[(str(i),'1','2',99,1767225600+i,'x') for i in range(50000)])
    assert len(extract_interactions(path,99)[0])==50000


def test_private_receive_only_and_multiple_threads(tmp_path):
    path=tmp_path/'private.sqlite'
    make_db(path,[msg(2,0,'a',peer='2'),msg(3,10,'b',peer='3')],source_format='nt',mode='c2c',identifier=1)
    result=analyze(Request(db=str(path),identifier='1',mode='c2c'))
    assert len(result['pairs'])==2 and all(p['response_rounds']==0 for p in result['pairs'])
    assert all(p['one_way_fraction']==1 for p in result['pairs'])


def test_parser_requires_explicit_format_and_validates_metadata():
    payload=json.dumps({'schema':'tei.message.v1','text':'x','mentions':['2','2']})
    assert decode_payload(payload)['structure']=='unknown'
    assert decode_payload(payload,True)['mentions']==['2']
    assert decode_payload(payload.replace('["2", "2"]','[2]'),True)['structure']=='invalid'


def test_untrusted_ids_cannot_leak_to_anonymous_result(tmp_path):
    path=tmp_path/'ids.sqlite'
    make_db(path,[msg(1,0,'private-path-or-account'),msg(2,10,'another-private-key',reply='private-path-or-account')])
    result=analyze(Request(db=str(path),identifier='99',source_format='tei'))
    serialized=json.dumps(result)
    assert 'private-path-or-account' not in serialized and 'another-private-key' not in serialized
    assert result['pairs'][0]['evidence_examples'][0]['message_id']=='R000002'


def test_oversized_id_is_unavailable_without_materializing_it(tmp_path):
    path=tmp_path/'ids.sqlite'
    make_db(path,[msg(1,0,'x'*9000),msg(2,10,'b')])
    records,_=extract_interactions(path,99,source_format='tei')
    assert records[0]['message_id'] is None


def test_invalid_time_sender_and_metadata_reported(tmp_path):
    path=tmp_path/'bad.sqlite';make_db(path,[msg(1,0,'a'),msg(2,1,'b')],source_format='nt')
    with sqlite3.connect(path) as c:
        c.execute('UPDATE group_msg_table SET "40050"=? WHERE "40001"=?',('invalid','a'))
        c.execute('UPDATE group_msg_table SET "40033"=? WHERE "40001"=?',('x'*1000,'b'))
    records,coverage=extract_interactions(path,99)
    assert records==[] and coverage['invalid_time']==1 and coverage['invalid_sender']==1

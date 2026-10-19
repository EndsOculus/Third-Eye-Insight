import copy
import json
import sqlite3
from datetime import timedelta
import pytest
from desktop_core import Request,analyze
from interaction_v2 import analyze_records
from streaming_store import RecordStore,ResourceLimit
from tests.interaction_fixtures import msg,make_db


def streaming(records,mode,tmp_path):
    with RecordStore(tmp_path,mode=mode) as store:
        for i,row in enumerate(copy.deepcopy(records)):
            row['order']=i;store.append(row)
        store.prepare()
        return analyze_records(store,mode=mode,self_id='1')


@pytest.mark.parametrize('mode',['group','c2c'])
def test_incremental_matches_reference_across_batches_conflicts_and_sessions(tmp_path,mode):
    records=[]
    for i in range(90):
        r=msg(1 if i%2==0 else 2,(i//3)*13+(i//40)*25000,str(i),reply=str(i-1) if i else None,mentions=['3'] if i%7==0 else [],peer='2',thread='one',text='same' if i%5==0 else str(i))
        records.append(r)
    records += [dict(records[5]),dict(records[20],text='conflict'),msg(3,1,'other',peer='3',thread='two')]
    for i,row in enumerate(records):row['order']=i
    reference=analyze_records(copy.deepcopy(records),mode=mode,self_id='1')
    assert streaming(records,mode,tmp_path)==reference


def test_large_database_has_no_global_50000_cap(tmp_path):
    path=tmp_path/'large.db'
    make_db(path,[],source_format='nt',mode='c2c')
    with sqlite3.connect(path) as db:
        db.executemany('INSERT INTO c2c_msg_table VALUES (?,?,?,?,?,?)',((str(i),str(1+i%2),'2',99,1767225600+i,'synthetic') for i in range(60001)))
    result=analyze(Request(db=str(path),identifier='1',mode='c2c'))
    assert result['source']['selected_rows']==60001
    assert result['rows']==60001 and result['pairs'][0]['response_rounds']==60000
    assert not result['source']['truncated']
    assert '_local_identity' not in result
    assert not list(tmp_path.glob('tei-stream-*'))


def test_resource_budget_and_cancellation_cleanup(tmp_path,monkeypatch):
    import streaming_store
    with pytest.raises(ResourceLimit,match='768 MiB'):
        with RecordStore(tmp_path) as store:
            monkeypatch.setattr(streaming_store,'resident_bytes',lambda:10**12)
            store.check(True)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(streaming_store,'resident_bytes',lambda:0)
    with pytest.raises(ResourceLimit,match='cancelled'):
        with RecordStore(tmp_path,cancelled=lambda:True) as store:store.check(True)
    assert not list(tmp_path.iterdir())


def test_local_identity_is_gui_only_and_owner_not_first(tmp_path):
    path=tmp_path/'identity.db'
    make_db(path,[msg(900,0,'a',peer='2'),msg(2,3,'b',peer='2')],source_format='nt',mode='c2c')
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE contacts(qq TEXT,remark TEXT,nickname TEXT)')
        db.execute('INSERT INTO contacts VALUES (?,?,?)',('2','Synthetic Friend','Synthetic Nick'))
    request=Request(db=str(path),identifier='900',mode='c2c')
    public=analyze(request);local=analyze(request,local_identity=True)
    assert 'Synthetic Friend' not in json.dumps(public)
    assert '_local_identity' not in public
    owner=local['_self_label'];peer=next(k for k in local['_local_identity'] if k!=owner)
    assert local['_local_identity'][owner]['qq']=='900'
    assert local['_local_identity'][peer]=={'qq':'2','name':'Synthetic Friend'}
    assert local['pairs'][0]['b']==owner


def test_conflicting_self_references_use_unique_author_lookup(tmp_path):
    with RecordStore(tmp_path) as store:
        for i in range(4000):store.append(dict(msg(1,i,'bad',reply='bad',text=str(i)),order=i))
        queries=[]
        store.db.set_trace_callback(lambda sql:queries.append(1) if sql.startswith('SELECT sender FROM authors') else None)
        store.prepare()
        assert store.conflicts==1 and len(store)==0
        assert len(queries)==1


def test_memory_sampler_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor
    from streaming_store import resident_bytes
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(value>=0 for value in executor.map(lambda _:resident_bytes(),range(100)))


def test_native_sender_nickname_uses_verified_field_and_latest_timestamp(tmp_path):
    path=tmp_path/'native names.db'
    make_db(path,[msg(900,0,'a',peer='2'),msg(2,1,'b',peer='2'),msg(2,2,'c',peer='2')],source_format='nt',mode='c2c')
    with sqlite3.connect(path) as db:
        db.execute('ALTER TABLE c2c_msg_table ADD COLUMN "40093" TEXT')
        db.execute('UPDATE c2c_msg_table SET "40093"=? WHERE "40001"=?',('Old Synthetic Nick','b'))
        db.execute('UPDATE c2c_msg_table SET "40093"=? WHERE "40001"=?',('New Synthetic Nick','c'))
    request=Request(db=str(path),identifier='900',mode='c2c')
    local=analyze(request,local_identity=True)
    assert next(v['name'] for v in local['_local_identity'].values() if v['qq']=='2')=='New Synthetic Nick'
    assert 'Synthetic Nick' not in json.dumps(analyze(request))
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE contacts(qq,remark,nickname)')
        db.execute("INSERT INTO contacts VALUES ('2','Preferred Synthetic Remark','')")
    local=analyze(request,local_identity=True)
    assert next(v['name'] for v in local['_local_identity'].values() if v['qq']=='2')=='Preferred Synthetic Remark'

    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO contacts VALUES ('2','Conflicting Synthetic Remark','')")
    local=analyze(request,local_identity=True)
    assert next(v['name'] for v in local['_local_identity'].values() if v['qq']=='2')==''

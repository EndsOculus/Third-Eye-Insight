from datetime import datetime, timedelta
import json
import sqlite3


def msg(sender, seconds, mid=None, reply=None, mentions=(), text='synthetic message', peer='2', thread='g', known=True):
    return dict(sender=str(sender), timestamp=datetime(2026,1,1,12)+timedelta(seconds=seconds),
                message_id=mid, reply_to=reply, mentions=list(map(str,mentions)), text=text,
                peer=str(peer), thread=thread, structure='known' if known else 'unknown', forwarded=False, order=0)


def make_db(path, records, source_format='tei', mode='group', identifier=99):
    with sqlite3.connect(path) as conn:
        if source_format=='tei':
            conn.execute('CREATE TABLE tei_messages_v1 (sender_id TEXT, peer_id TEXT, message_id TEXT, timestamp INTEGER, conversation_id INTEGER, mode TEXT, content TEXT, metadata TEXT)')
            for r in records:
                meta=dict(schema='tei.message.v1',reply_to=r.get('reply_to'),mentions=r.get('mentions',[]),forwarded=r.get('forwarded',False))
                ts=int((r['timestamp']-datetime(1970,1,1)).total_seconds())-8*3600
                conn.execute('INSERT INTO tei_messages_v1 VALUES (?,?,?,?,?,?,?,?)',(r['sender'],r['peer'],r['message_id'],ts,identifier,mode,r['text'],json.dumps(meta)))
        else:
            table='group_msg_table' if mode=='group' else 'c2c_msg_table'
            conn.execute(f'CREATE TABLE {table} ("40001" TEXT,"40033" TEXT,"40030" TEXT,"40027" INTEGER,"40050" INTEGER,"40800")')
            for r in records:
                ts=int((r['timestamp']-datetime(1970,1,1)).total_seconds())-8*3600
                conn.execute(f'INSERT INTO {table} VALUES (?,?,?,?,?,?)',(r['message_id'],r['sender'],r['peer'],identifier,ts,r['text']))

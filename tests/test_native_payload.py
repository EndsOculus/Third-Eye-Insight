import hashlib
import json
import numpy as np
import pytest
from native_payload import decode_native_text
from interaction_data import extract_interactions
from interaction_assist import topic_candidates
from desktop_core import Request, analyze
from tests.interaction_fixtures import make_db, msg


def vi(n):
    out=bytearray()
    while n > 127:
        out.append((n & 127)|128);n >>= 7
    return bytes(out)+bytes([n])


def field(n,value):
    return vi(n*8+2)+vi(len(value))+value


def body(text):
    return field(40800,field(45101,text.encode('utf-8')))


def test_body_and_cached_quote_are_separate():
    payload=field(40800,field(47423,field(45101,b'cached quote'))+field(45101,'正文'.encode()))
    result=decode_native_text(payload)
    assert result['text']=='正文' and result['native_quote_present']
    assert decode_native_text(field(40800,field(47423,body('quoted only'))))['text']==''
    assert decode_native_text(field(40800,field(45101,b'echo')+field(45101,b'echo')))['text']=='echo\necho'


@pytest.mark.parametrize('payload',[
    body('valid')+b'\x80',field(40800,b'\x0a\x7fshort'),
    field(45101,b'\xff'),field(45101,b'a\x00b'),b'\x00',
    b'\xff'*11,field(45101,b'x')*257,
])
def test_malformed_payload_never_returns_partial_text(payload):
    result=decode_native_text(payload)
    assert result['text']=='' and result['native_text_status']=='invalid'


def test_depth_unknown_fields_and_byte_budget():
    nested=body('nested')
    for _ in range(9):nested=field(40800,nested)
    assert decode_native_text(nested)['native_text_status']=='invalid'
    assert decode_native_text(field(999,body('not a verified body path')))['text']==''
    assert decode_native_text(b'x'*8193)['text']==''
    assert decode_native_text(field(45402,b'synthetic-file.txt'))['text']==''


def test_native_sqlite_text_reaches_local_encoder_without_creating_metadata(tmp_path):
    path=tmp_path/'构造 native input.sqlite'
    rows=[msg(1,0,'a',text=body('first synthetic topic message')),
          msg(2,10,'b',text=body('second synthetic topic message')),
          msg(2,20,'c',text=field(40800,field(47423,body('quoted secret canary'))+field(45101,b'current reply body')))]
    make_db(path,rows,source_format='nt')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    records,source=extract_interactions(path,99)
    assert source['native_text_rows']==3 and source['native_quote_rows']==1
    assert all(r['structure']=='unknown' and not r['mentions'] and r['reply_to'] is None for r in records)
    class Encoder:
        def encode(self,texts,**kwargs):
            assert texts==['first synthetic topic message','second synthetic topic message','current reply body']
            return np.array([[1.,0.],[1.,0.],[1.,0.]])
    predictions=topic_candidates(records,Encoder())['predictions']
    assert predictions[1]['status']=='candidate'
    assert predictions[2]['reason']=='explicit_or_unresolved_metadata_use_rules_only'
    result=analyze(Request(db=str(path),identifier='99'))
    assert result['pairs'][0]['evidence']=='coverage_unknown'
    serialized=json.dumps(result)
    assert 'synthetic topic' not in serialized and 'quoted secret canary' not in serialized
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before

from datetime import timedelta
import pytest
from interaction_v2 import analyze_records
from tests.interaction_fixtures import msg


def run(rows,mode='group',**kw):
    return analyze_records(rows,mode=mode,self_id='1',**kw)


def pair(result,a='1',b='2'):
    return next(p for p in result['pairs'] if p['a']==a and p['b']==b)


def test_group_time_similarity_neighbors_never_create_response():
    rows=[msg(1,0,'a',mentions=['3']),msg(3,10,'b',reply='a'),msg(2,1000,'c',mentions=['3']),msg(3,1010,'d',reply='c')]
    p=pair(run(rows))
    assert p['response_rounds']==0 and p['directed_rounds']==0
    assert p['common_neighbors']==1
    assert p['source_messages_a']==p['source_messages_b']==1
    assert 'score' not in p


def test_unknown_not_zero_and_plain_at_not_metadata():
    result=run([msg(1,0,'a',text='@2 hello',known=False),msg(2,1,'b',known=False)])
    p=pair(result)
    assert p['evidence']=='coverage_unknown' and p['response_rounds']==0
    assert p['one_way_fraction'] is None and p['median_response_seconds'] is None


def test_counts_volume_and_oneway_not_equal_evidence():
    small=pair(run([msg(1,0,'a'),msg(2,10,'b')],'c2c'))
    many=pair(run([msg(1+i%2,i*10,str(i)) for i in range(200)],'c2c'))
    spam=pair(run([msg(1,0,'a')]+[msg(2,i,str(i)) for i in range(1,100)],'c2c'))
    assert (small['response_rounds'],many['response_rounds'],spam['response_rounds'])==(1,199,1)
    assert small['evidence']=='insufficient'
    assert spam['messages_b']==99 and spam['directed_rounds']==2 and spam['repeated_messages']==98
    one=pair(run([msg(1,i,str(i)) for i in range(99)],'c2c'))
    assert one['response_rounds']==0 and one['one_way_fraction']==1


def test_private_night_shift_and_other_thread_invariance():
    rows=[msg(1,0,'a'),msg(2,10,'b')]
    base=pair(run(rows,'c2c'))
    night=pair(run([dict(r,timestamp=r['timestamp']+timedelta(hours=12)) for r in rows],'c2c'))
    other=pair(run(rows+[msg(1,29*86400,'x',peer='3',thread='other'),msg(3,29*86400+10,'y',peer='3',thread='other')],'c2c'))
    assert base==night
    for key in ('response_rounds','messages_a','messages_b','common_active_days','median_response_seconds','one_way_fraction','bidirectional_sessions'):
        assert base[key]==other[key]


def test_session_boundary_no_reply_latency_and_no_cross_thread():
    p=pair(run([msg(1,0,'a'),msg(2,86400,'b')],'c2c'))
    assert p['response_rounds']==0 and p['latency_samples']==0 and p['sessions']==2
    p=pair(run([msg(1,0,'a',thread='one'),msg(2,10,'b',thread='two')],'c2c'))
    assert p['response_rounds']==0 and p['latency_samples']==0


def test_simultaneous_ids_never_create_causality():
    rows=[msg(1,0,'a'),msg(2,0,'b'),msg(1,10,'c')]
    p=pair(run(rows,'c2c'))
    swapped=pair(run([dict(rows[0],message_id='b'),dict(rows[1],message_id='a'),rows[2]],'c2c'))
    assert {k:v for k,v in p.items() if k!='evidence_examples'}=={k:v for k,v in swapped.items() if k!='evidence_examples'}
    assert p['response_rounds']==0 and p['latency_samples']==0


def test_multitarget_dedup_self_and_reply_mention_same_target():
    rows=[msg(2,0,'source'),msg(1,10,'reply',reply='source',mentions=['1','2','2','3'])]
    result=run(rows)
    p=pair(result)
    assert p['messages_a']==1 and p['reply_links']==1 and p['mention_rounds']==0 and p['response_rounds']==1
    assert pair(result,'1','3')['messages_a']==1
    assert all(p['a']!=p['b'] for p in result['pairs'])


def test_reply_kind_switch_does_not_inflate_turns():
    rows=[msg(2,0,'source'),msg(1,1,'a',mentions=['2']),msg(1,2,'b',reply='source'),msg(1,3,'c',mentions=['2']),msg(1,4,'d',reply='source')]
    p=pair(run(rows))
    assert p['directed_rounds']==1 and p['messages_a']==4
    assert p['response_rounds']<=1


@pytest.mark.parametrize('source',[None,'missing','future','same','self'])
def test_reference_resolution_and_coverage(source):
    rows=[msg(2,20 if source=='future' else 10 if source=='same' else 0,'source'),msg(1,10,'query',reply={'self':'query','future':'source','same':'source'}.get(source,source))]
    result=run(rows);p=pair(result)
    assert p['response_rounds']==0
    if source is not None:
        assert result['coverage']['unresolved_references']==1 and p['evidence']=='coverage_unknown'


def test_namespace_duplicates_and_conflicts():
    rows=[msg(2,0,'a',thread='one'),msg(1,10,'b',reply='a',thread='two')]
    assert pair(run(rows))['response_rounds']==0
    rows=[msg(2,0,'a'),msg(1,10,'b',reply='a')]
    assert pair(run(rows+[rows[1]]))==pair(run(rows))
    assert run(rows+[rows[1]])['coverage']['duplicate_rows']==1
    conflict=dict(rows[0],structure='unknown')
    result=run(rows+[conflict])
    assert result['coverage']['conflicting_ids']==1 and result['coverage']['unresolved_references']==1


def test_repetition_preserved_not_content_dedup():
    p=pair(run([msg(1,0,'a'),msg(2,1,'b'),msg(2,2,'c')],'c2c'))
    assert p['messages_b']==2 and p['repeated_messages']==1 and p['response_rounds']==1


def test_empty_singleton_and_resource_limit():
    assert run([])['pairs']==[]
    assert run([msg(1,0,'a')])['pairs']==[]
    with pytest.raises(ValueError):run([msg(i,i,str(i)) for i in range(301)])


def test_window_sensitivity_is_auxiliary_only():
    rows=[msg(1,0,'a'),msg(2,400,'b')]
    a=pair(run(rows,coactive_bin=300));b=pair(run(rows,coactive_bin=600))
    assert a['response_rounds']==b['response_rounds']==0
    assert a['coactive_bins']==0 and b['coactive_bins']==1


def test_group_simultaneous_reply_order_independent():
    rows=[msg(1,0,'x'),msg(2,0,'y'),msg(1,10,'a',reply='y'),msg(2,10,'b',reply='x'),msg(1,20,'c',reply='y')]
    first=pair(run(rows))
    rows[2]['message_id'],rows[3]['message_id']='b','a'
    second=pair(run(rows))
    assert {k:v for k,v in first.items() if k!='evidence_examples'}=={k:v for k,v in second.items() if k!='evidence_examples'}
    assert first['response_rounds']==3


def test_private_all_statistics_independent_of_other_peer():
    rows=[msg(1,0,'a'),msg(2,10,'b')]
    base=pair(run(rows,'c2c'))
    changed=pair(run(rows+[msg(1,20,'x',peer='3',thread='another')],'c2c'))
    assert base==changed


def test_recipient_budget_is_explicit_not_truncated():
    with pytest.raises(ValueError,match='budget'):
        run([msg(1,i,str(i),mentions=range(2,202)) for i in range(501)])

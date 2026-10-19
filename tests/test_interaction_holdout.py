"""Independent generated holdout families; expectations from event schedules, not scores."""
import random
from datetime import timedelta
import pytest
from interaction_v2 import analyze_records
from tests.interaction_fixtures import msg


@pytest.mark.parametrize('seed',range(12))
def test_generated_private_session_turns(seed):
    rng=random.Random(7300+seed)
    rows=[];expected=0;messages=0;bidirectional=0
    for day in range(3):
        speaker=1;turns=rng.randint(2,7);bidirectional+=1
        seconds=day*86400
        for turn in range(turns):
            for _ in range(rng.randint(1,5)):
                rows.append(msg(speaker,seconds,str(messages)));messages+=1;seconds+=rng.randint(1,20)
            speaker=3-speaker
        expected+=turns-1
    base=analyze_records(rows,mode='c2c')['pairs'][0]
    assert base['response_rounds']==expected and base['bidirectional_sessions']==bidirectional
    assert base['messages_a']+base['messages_b']==messages
    extra=[msg(1,1000000,'other1',peer='9',thread='other'),msg(9,1000010,'other2',peer='9',thread='other')]
    expanded=next(p for p in analyze_records(rows+extra,mode='c2c')['pairs'] if p['b']=='2')
    for k in ('response_rounds','messages_a','messages_b','common_active_days','latency_samples','median_response_seconds'):
        assert expanded[k]==base[k]


@pytest.mark.parametrize('seed',range(8))
def test_generated_group_no_targets_never_direct(seed):
    rng=random.Random(seed+8000)
    rows=[msg(rng.randint(1,8),rng.randint(0,10000),str(i),text='same repeated text',known=bool(i%2)) for i in range(150)]
    assert all(p['response_rounds']==0 and p['directed_rounds']==0 for p in analyze_records(rows)['pairs'])

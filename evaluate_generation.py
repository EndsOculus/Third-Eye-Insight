"""Frozen three-mode synthetic holdout evaluation; no training or threshold fitting."""
import argparse
from datetime import datetime,timedelta
import json
from pathlib import Path
from interaction_v2 import analyze_records
from interaction_assist import load_local_encoder,topic_candidates
from generation_assist import LocalGenerator,generate_candidates


def evaluate(path,config):
    path=Path(path)
    if path.stat().st_size>1024*1024:raise ValueError('Dataset too large')
    cases=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not 1<=len(cases)<=100:raise ValueError('Dataset must contain 1..100 cases')
    encoder=load_local_encoder();generator=LocalGenerator(config)
    outcomes=[]
    for case in cases:
        if case.get('label_status')!='labeled':continue
        records=[dict(sender=m['sender'],timestamp=datetime(2026,1,1,12)+timedelta(seconds=m['second']),
                      message_id=m['id'],reply_to=m.get('reply_to'),mentions=m.get('mentions',[]),
                      text=m['text'],forwarded=m.get('forwarded',False),peer='2',thread=case['case'],structure='known',order=i)
                 for i,m in enumerate(case['messages'])]
        query=records[-1]
        direct=analyze_records(records,evidence_limit=100000)
        sources=[e for p in direct['pairs'] for e in p['evidence_examples'] if e['message_id']==query['message_id'] and e['kind']=='reply']
        rule=sources[0]['target'] if len({e['target'] for e in sources})==1 else None
        vector=topic_candidates(records,encoder=encoder)
        if vector['status']!='completed':raise ValueError('Vector mode unavailable')
        vp=next(p for p in vector['predictions'] if p['row']==len(records))
        result=generate_candidates(records,generator=generator,query_indices=[len(records)-1])
        if result['status']!='completed':raise ValueError('Generation mode unavailable')
        gp=result['predictions'][0]
        outcomes.append(dict(case=case['case'],gold_target=case['gold_target'],gold_source=case['gold_source'],
                             gold_topic=case['gold_topic'],gold_type=case['gold_type'],rule_target=rule,
                             vector_target=rule or vp['candidate_target'],generation_target=rule or gp['candidate_target'],
                             generation_source=gp['source_message_id'],topic=gp['topic'],interaction=gp['interaction'],
                             reason=gp['reason'],seconds=result['seconds'],evidence_rows=gp['evidence_rows']))
    def metric(key):
        predicted=[o for o in outcomes if o[key] is not None];positive=[o for o in outcomes if o['gold_target'] is not None]
        correct=sum(o[key]==o['gold_target'] for o in predicted)
        return dict(cases=len(outcomes),target_cases=len(positive),predictions=len(predicted),correct=correct,wrong=len(predicted)-correct,
                    abstentions=len(outcomes)-len(predicted),precision=correct/len(predicted) if predicted else None,
                    target_recall=correct/len(positive) if positive else None,abstention_rate=(len(outcomes)-len(predicted))/len(outcomes))
    return dict(prompt_version='generation-v1-frozen',dataset='Independent synthetic holdout, not real-chat validity',
                rule=metric('rule_target'),vector=metric('vector_target'),generation=metric('generation_target'),
                topic_exact=sum(o['topic']==o['gold_topic'] for o in outcomes),type_exact=sum(o['interaction']==o['gold_type'] for o in outcomes),
                topic_abstentions=sum(o['topic']=='unknown' for o in outcomes),type_abstentions=sum(o['interaction']=='unknown' for o in outcomes),
                invalid_outputs=sum(o['reason']=='invalid_output_or_runtime_failure' for o in outcomes),cases=outcomes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',default=str(Path(__file__).parent/'tests/generation_holdout.jsonl'))
    parser.add_argument('--generation-config',required=True)
    args=parser.parse_args()
    try:print(json.dumps(evaluate(args.labels,args.generation_config),ensure_ascii=True,indent=2))
    except (OSError,ValueError):parser.exit(2,'Local evaluation unavailable; no dataset or resource paths logged.\n')

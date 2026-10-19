"""Offline labeled JSONL evaluation; no training, uploads, or guessed labels.

Use only an authorized dataset. The supplied fixture is synthetic and held out
from rule/threshold design; results do not establish real-world accuracy.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
from interaction_assist import topic_candidates, load_local_encoder
from interaction_v2 import analyze_records


def evaluate(path, use_model=False):
    cases=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(cases)>1000:
        raise ValueError('Evaluation limited to 1000 cases')
    model=load_local_encoder() if use_model else None
    outcomes=[]
    for case in cases:
        if case.get('label_status')!='labeled':
            continue
        records=[]
        for i,m in enumerate(case['messages']):
            records.append(dict(sender=m['sender'],timestamp=datetime(2026,1,1,12)+timedelta(seconds=m['second']),
                                message_id=m['id'],reply_to=m.get('reply_to'),mentions=m.get('mentions',[]),
                                text=m['text'],forwarded=m.get('forwarded',False),peer='2',thread=case['case'],
                                structure='known',order=i))
        query=records[-1]
        observed=analyze_records(records,evidence_limit=100000)
        sources=[e['target'] for pair in observed['pairs'] for e in pair['evidence_examples']
                 if e['message_id']==query['message_id'] and e['kind']=='reply']
        baseline=sources[0] if len(set(sources))==1 else None
        assist=baseline;reason='explicit_rule' if baseline is not None else 'abstain_rule'
        if use_model and baseline is None:
            result=topic_candidates(records,encoder=model)
            if result['status']!='completed':
                raise RuntimeError('Model evaluation failed; no metrics emitted')
            prediction=next(p for p in result['predictions'] if p['message_id']==query['message_id'])
            assist=prediction['candidate_target'];reason=prediction['reason']
        outcomes.append(dict(case=case['case'],gold_target=case['gold_target'],rule_target=baseline,assisted_target=assist,reason=reason))
    def metrics(key):
        predicted=[o for o in outcomes if o[key] is not None]
        positives=[o for o in outcomes if o['gold_target'] is not None]
        correct=sum(o[key]==o['gold_target'] for o in predicted)
        return dict(cases=len(outcomes),labeled_target_cases=len(positives),predictions=len(predicted),correct_attributions=correct,
                    wrong_attributions=len(predicted)-correct,abstentions=len(outcomes)-len(predicted),
                    precision=correct/len(predicted) if predicted else None,
                    target_recall=correct/len(positives) if positives else None,
                    abstention_rate=(len(outcomes)-len(predicted))/len(outcomes) if outcomes else None)
    return dict(dataset='Caller-provided labels; supplied default is synthetic holdout only',model_used=use_model,
                rule=metrics('rule_target'),assisted=metrics('assisted_target'),cases=outcomes,
                limitation='Candidate targets are not verified replies. No real-data validity, calibrated confidence, or relationship inference established.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',default=str(Path(__file__).parent/'tests/assist_holdout.jsonl'))
    parser.add_argument('--local-model',action='store_true')
    args=parser.parse_args()
    print(json.dumps(evaluate(args.labels,args.local_model),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

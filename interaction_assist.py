"""Optional OFFLINE topic-candidate assistance; never adds direct interaction counts.

The available multilingual sentence encoder is not a dialogue reasoning or emotion
model. Cosine and separation thresholds are design rules, NOT calibrated confidence.
No generation, tools, downloads, persistent embedding cache, or instruction execution.
"""
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np

MODEL_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
MAX_RECORDS = 2000
MAX_CONTEXT = 20
WINDOW_SECONDS = 300
MIN_COSINE = .70
MIN_MARGIN = .12


def load_local_encoder():
    snapshot = (Path(__file__).parent / '.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots' / MODEL_REVISION)
    if not snapshot.is_dir():
        raise FileNotFoundError('Approved local model snapshot is not installed')
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(snapshot), local_files_only=True, token=False, trust_remote_code=False, device='cpu')


def topic_candidates(records, encoder=None):
    output = dict(version='topic-candidate-v1', status='completed', model='paraphrase-multilingual-MiniLM-L12-v2',
                  revision=MODEL_REVISION, predictions=[], max_context=MAX_CONTEXT, window_seconds=WINDOW_SECONDS,
                  minimum_cosine=MIN_COSINE, minimum_margin=MIN_MARGIN,
                  interpretation='Topic candidates only; not verified replies, emotions, or calibrated probabilities.',
                  unsupported=['dialogue reasoning','comfort/argument/sarcasm classification','relationship judgments'])
    if len(records) > MAX_RECORDS:
        output.update(status='limit_exceeded', reason='Narrow selection to at most 2000 records; no partial assistance run.')
        return output
    if not records:
        return output
    try:
        model = encoder or load_local_encoder()
        embeddings = np.asarray(model.encode([r['text'][:512] for r in records], batch_size=32, show_progress_bar=False), dtype=float)
        if embeddings.ndim != 2 or len(embeddings) != len(records) or not np.isfinite(embeddings).all():
            raise ValueError('Invalid local encoder result')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        vectors = np.divide(embeddings,norms,out=np.zeros_like(embeddings),where=norms>0)
    except Exception:
        output.update(status='model_unavailable', reason='Approved local encoder unavailable; rule statistics remain valid. No download attempted.')
        return output
    id_counts = Counter((r.get('thread',r.get('peer','group')),r.get('message_id')) for r in records)
    context = defaultdict(lambda:deque(maxlen=MAX_CONTEXT))
    ordered = sorted(enumerate(records),key=lambda ir:(ir[1]['timestamp'],ir[1].get('message_id') or '',ir[0]))
    for idx,r in ordered:
        thread = r.get('thread',r.get('peer','group'))
        prior = context[thread]
        while prior and (r['timestamp']-records[prior[0]]['timestamp']).total_seconds()>WINDOW_SECONDS:
            prior.popleft()
        prediction = dict(thread=thread, message_id=r.get('message_id'), row=idx+1, source_sender=r['sender'],
                          candidate_target=None, source_message_id=None, status='abstain',
                          reason='', cosine=None, margin=None, kind='topic_candidate', cues=[])
        if '?' in r['text'] or '？' in r['text']:
            prediction['cues'].append('question_punctuation_present_not_QA_classification')
        if r.get('forwarded'):
            prediction['cues'].append('structured_forward_flag')
        reason = None
        if not r.get('message_id') or id_counts[(thread,r.get('message_id'))] != 1:
            reason = 'missing_or_ambiguous_source_id'
        elif r.get('reply_to') or r.get('mentions') or r.get('native_quote_present'):
            reason = 'explicit_or_unresolved_metadata_use_rules_only'
        elif r.get('forwarded'):
            reason = 'forwarded_content_not_attributed'
        elif len(r['text'].strip()) < 8:
            reason = 'short_or_empty_context'
        elif any(records[j]['text'].strip()==r['text'].strip() for j in prior):
            reason = 'repeated_text_not_attributed'
            prediction['cues'].append('exact_repetition')
        candidates=[]
        if reason is None:
            for j in prior:
                q=records[j]
                if (q['sender'] != r['sender'] and q['timestamp'] < r['timestamp'] and q.get('message_id')
                        and id_counts[(thread,q['message_id'])] == 1 and not q.get('forwarded') and len(q['text'].strip())>=8):
                    candidates.append((float(np.dot(vectors[idx],vectors[j])),j))
            # Compare strongest candidate per sender, not multiple messages by one sender.
            best={}
            for score,j in candidates:
                if records[j]['sender'] not in best or score > best[records[j]['sender']][0]:
                    best[records[j]['sender']] = (score,j)
            ranked=sorted(best.values(),reverse=True)
            if not ranked:
                reason='no_eligible_prior_context'
            else:
                score,j=ranked[0]
                margin=score-ranked[1][0] if len(ranked)>1 else None
                prediction.update(cosine=score,margin=margin)
                # No competitor is not proof; retain the similarity threshold.
                if score < MIN_COSINE:
                    reason='low_topic_similarity'
                elif margin is not None and margin < MIN_MARGIN:
                    reason='ambiguous_topics'
                else:
                    prediction.update(status='candidate',candidate_target=records[j]['sender'],source_message_id=records[j]['message_id'])
                    reason='same_topic_prior_message_not_verified_response'
        prediction['reason']=reason
        output['predictions'].append(prediction)
        prior.append(idx)
    output['candidate_count']=sum(p['status']=='candidate' for p in output['predictions'])
    output['abstention_count']=sum(p['status']=='abstain' for p in output['predictions'])
    return output

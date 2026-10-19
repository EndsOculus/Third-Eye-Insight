# Optional local topic candidates

Default off. Uses the already installed sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 snapshot with local-files-only loading, CPU batches of 32, no downloads, remote calls, generation, tools or embedding-cache writes. Missing model produces an explicit unavailable state. Do not send real conversations to external models for validation.

The encoder supports similarity, not a trained dialogue-act or relationship classifier. The application does not claim reliable question/answer, comfort, sarcasm, dispute or emotion recognition. Question punctuation, repetition and structured forwarding are literal cues only. Candidate similarity is not a probability.

At most 2,000 selected messages, 512 characters per message, the model's 128-token cap, and 20 prior same-thread messages within 300 seconds. Candidates must be strictly earlier, have valid unique source IDs, similarity >= 0.70 and a margin >= 0.12 over another sender when present. Short/repeated/forwarded text, explicit or unresolved metadata, ties and insufficient context abstain. Exceeding the AI limit returns a limit state rather than partial candidates. Background worker cancellation also interrupts model loading/inference.

## Synthetic held-out evaluation

Run `python evaluate_interactions.py --help` for the offline annotation/evaluation tool. `tests/assist_holdout.jsonl` contains 12 synthetic cases, including six with labeled target recipients. Thresholds were not tuned to this holdout. `annotation_template.jsonl` is an unlabeled synthetic starting format, not human validation.

Actual existing local model, measured on 2026-10-19:

| Method | Predictions | Correct / wrong | Precision | Target recall | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured rule engine | 1 | 1 / 0 | 1.000 | 0.167 | 0.917 |
| Rules + topic candidates | 3 | 2 / 1 | 0.667 | 0.333 | 0.750 |

The added candidate for a group announcement selected another acknowledging participant instead of the announcer. Interleaved topics, sarcasm and comfort examples abstained. Higher recall here came with an actual false attribution. This tiny synthetic set establishes neither real-chat accuracy nor general improvement; no calibrated confidence or production validity is claimed. Direct counts are unchanged by AI. Evaluation uses the actual rule engine and source attribution, not a simulated baseline. Unlabeled records must not be silently treated as negative labels.


## Remaining stage gaps

Native text in supported 40800/45101 BLOB paths now reaches the local encoder; 47423 cached quotes are excluded from body, and cache-bearing messages abstain from recipient attribution. Unknown layouts still provide no text. Binary fixtures and a recording mock encoder verify the extraction boundary; they do not validate real-message classification.

This document describes vector mode only. It does not implement dialogue reasoning, free-form generation, emotion recognition or relationship judgments. A separate default-off local generation experiment now exists; see [GENERATION.md](GENERATION.md). It reuses explicitly approved local resources, performs bounded coarse classifications and candidate attribution, and renders fixed explanation templates with anonymous evidence IDs. Its held-out results include wrong attributions and a misclassified synthetic injection attempt; these are not validated classifiers. No new model was downloaded and no real conversation was sent externally for testing.

## Existing local snapshot configuration

Vector mode checks the project-relative directory `.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Select **向量候选** only if the approved snapshot already exists. There is no download or cache-location chooser in this workflow; missing resources leave direct statistics available and display an unavailable state. The cache stays ignored and must not be committed.

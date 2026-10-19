# Optional local generation experiment

The default remains rules only. Select **生成实验** and **生成设置** in the desktop to opt in. The configuration is a local JSON object with exactly `runtime`, `model`, and `dll_dir`, containing absolute local paths. Keep it outside the repository. No resource is downloaded automatically. CLI: `python train.py --demo --generate-assist --generation-config <local-config.json>`. Use a synthetic demo first; do not send private conversations to external services.

This adapter supports the explicitly fingerprinted llama.cpp b11723 Windows CUDA 12.4 CLI and Qwen3-4B-Q4_K_M model. Fingerprints are in generation_assist.py. Changing them requires a separately reviewed resource update. CUDA DLL lookup is changed only for the child process. The CLI can create a temporary local helper; this implementation sets the host to loopback and offline mode, but this is not a network-isolation audit.

## What the experiment does

A real local generative model chooses a possible prior-message target, coarse topic, observed utterance type and rationale category. The application displays validated anonymous evidence row IDs and a fixed explanation template selected by that category. It does not display free-form generated prose or chat excerpts, claim real emotions, provide confidence probabilities, or change directly observed counts. The details section explicitly covers the entire selected interval, not just the currently selected pair.

At most 12 selected records, 4 prior context messages, 300 seconds of prior context, and 160 characters per message are accepted. Oversized messages are skipped, not silently truncated. Same-time, other-thread, ambiguous-ID, forwarded and unresolved quotation cases abstain where applicable. No partial result is returned on total time-limit/cancellation. Model initialization hashes local resources; the desktop worker remains cancellable during this stage.

Each inference uses stdin without a prompt file, shell or body-bearing command line. The child starts suspended, enters a Windows kill-on-close Job, and then resumes. A desktop outer Job also owns the worker before it starts. Input writing and bounded output draining occur off the cancellation loop; timeout/cancel closes the entire owned job. No other application's process is terminated. Limits: 16 KiB input, 32 KiB stdout, 256 KiB discarded stderr, 45 seconds per invocation and 180 seconds for generation (checked across calls). Cleanup may take additional seconds.

The parser discards surrounding runtime/banner text, then parses the first-to-last brace span as one JSON object. Duplicate keys, invalid fields/enums/IDs, inconsistent targets and malformed output fail closed. This is post-generation validation, not grammar-constrained decoding. Prompt role delimiters are escaped and message text stays data. These controls prevent tool execution and raw-text output; they do not guarantee immunity to semantic prompt injection or erroneous attribution.

## Frozen independent synthetic holdout — 2026-10-19

The prompt was frozen before the independently authored 16-case holdout labels were inspected. No prompt or threshold was tuned against those results. Nine cases have an annotated target; seven require abstention. The fixture is synthetic and has no explicit reply metadata, so the direct-rule baseline correctly has no target evidence to use.

| Method | Predictions | Correct | Wrong | Abstentions | Precision | Target recall |
|---|---:|---:|---:|---:|---:|---:|
| Direct rules | 0 | 0 | 0 | 16 | undefined | 0/9 |
| Local vectors | 3 | 3 | 0 | 13 | 3/3 | 3/9 |
| Local generation | 12 | 8 | 4 | 4 | 8/12 | 8/9 |

Generation topic exact match: **6/16**; utterance type: **9/16**. Four unknown classifications and one invalid-output/runtime-failure abstention occurred. Wrong targets occurred in g10, g11, g12 and g15: the model supplied a target where the annotation required none. g10 contains a synthetic instruction-injection attempt and was misclassified, so semantic injection resistance is explicitly not established. g11 is insufficient short context, g12 has multiple plausible targets, and g15 repeats the same topic without a reply. Thus additional coverage carries substantial false attribution; the feature remains default-off and experimental. This small synthetic set does not establish real-chat validity, superior general accuracy, emotion recognition or safe autonomous decisions.

The 15 cases invoking the model took 2.416–2.716 seconds each on this machine; one context-ineligible case did not invoke it. These per-case times include runtime/model startup and generation but exclude the evaluator's once-only fingerprint verification and initial CPU vector model load. They are not token-throughput measurements or a speedup comparison. The actual desktop 8-record synthetic workflow took 20.043 seconds end-to-end (generation stage 19.707 seconds); it returned 5 candidates and 3 abstentions. Hardware/runtime/cache state affects these numbers.

Reproduce explicitly with `python evaluate_generation.py --generation-config <local-config.json>`; inspect `--help` before use. The evaluation uses only the committed synthetic holdout, loads existing local resources, and prints label-level results rather than chat bodies. Never replace it with private fixtures for publication.

## Verification and limitations

135 non-desktop tests and all 10 individually isolated desktop cases passed locally. Ruff selected correctness rules and Python compilation passed. Added actual Windows process tests cover unread 10 KB stdin, timeout, cancel, output overflow and descendant cleanup. Independent review reproduced the former synchronous-write hang and verified its repair, plus post-inference cancellation/deadline checks.

Actual Tk workflows with the approved model exercised repeated start, completion, cancellation while a llama process was live, and closing while live. Both stopped all task-owned model processes; no llama processes remained afterward. App-only screenshots were visually inspected at 900×700 and retained outside Git. Native encrypted/private databases and real conversational validity were not tested. Tests use mock generators or synthetic data; CI never downloads or starts a model. There is no claim of zero sensitive information: staged files and reachable source history are scanned and reviewed, but scanners have limits.

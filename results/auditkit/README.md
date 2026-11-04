# Actual AuditKIT cross-check

Offline AuditKIT was executed on the 164 stored model predictions per model, not on reference answers substituted as predictions. No judge service was called. Reproduce with `pip install auditkit` then `python scripts/audit_offline.py base.jsonl sft.jsonl` from the repository root.

`expected_action_accuracy` is AuditKIT Abstention's first score. On unanswerable questions it rewards refusal; on answerable ones it rewards not refusing. It uses the exact markers from `scripts/score.py`. This is a lexical refusal proxy, not a semantic safety guarantee.

AuditKIT's F1Score has different normalization from the headline scorer, including article removal. Do not mix their absolute values. The headline table remains `scripts/score.py` throughout. Neither score measures legal correctness, and all data is synthetic and limited to the supplied sections. The audit does not establish deployment readiness or independent expert validation.

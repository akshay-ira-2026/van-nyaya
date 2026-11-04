# Teaching a tiny bilingual model to say “the section does not say”

*Hugging Face Article draft · Akshay Wagh · Lexsi Track 1*

> A small, reproducible forest-law experiment, not a legal-advice product.

**Headline:** on 164 held-out questions from statute sections never seen in training, the Van Nyaya SFT adapter takes lexically detected refusals on unanswerable questions from **10% to 100%**, and lifts grounded answer quality by **+22% (English)** and **+15% (Hindi)** token F1, at a cost of 3 false refusals in 124 answerable questions.

## 1. The problem

Indian forest and environment law sits across several Acts that forest officers, gram sabhas and citizens have to read together: the Indian Forest Act 1927, the Wildlife (Protection) Act 1972, the Forest (Conservation) Act 1980 as amended in 2023 (Van (Sanrakshan Evam Samvardhan) Adhiniyam), the Scheduled Tribes and Other Traditional Forest Dwellers (Recognition of Forest Rights) Act 2006 with its Rules, and the Environment (Protection) Act 1986.

A small multilingual model asked about these laws tends to do two unsafe things:

1. Answer from memory instead of from the section in front of it (wrong section numbers, invented penalties).
2. Answer confidently when the given text does not contain the answer.

Van Nyaya fine-tunes Tiny Aya Global so that it answers only from the supplied section, cites it, answers in English or Hindi, and says plainly when the text does not cover the question.

## 2. Pipeline

1. **CuratorKIT 0.1.22**: 332 sections chunked from the bare Acts, grounded QA generation (3 questions per section, ministral-8b generator + judge, hallucination gate 0.5, exact dedup, dataset card written by the toolkit). Section-level split (seed 7): 33 held-out sections are never seen in training.
2. **Data built on top of the curated set**
    - Hindi QA (translated, 150 train / 30 eval)
    - Refusal set: questions paired with a *mismatched* section, gold answer = an abstention (300 train / 40 eval)
    - DPO pairs (749): grounded answer vs closed-book answer (449), abstain vs hallucinate (300)
3. **Custom LoRA training recipe**: SFT on a 511-row mix (English grounded + Hindi + refusal), 132 steps. DPO (200 pairs) as an ablation directly on the base model, and on top of the SFT adapter.
4. **Before/after evaluation** on the 164-row held-out set: 94 English grounded, 30 Hindi grounded, 40 English unanswerable. Every model is merged, converted to GGUF and evaluated in the same quantised form a user would run, with the same prompts and decoding. Scored with `scripts/score.py` (token F1 on grounded, refusal rate on unanswerable, false-refusal rate on answerable).

Base model: CohereLabs/tiny-aya-global (weights loaded from an ungated mirror of the same checkpoint). Training ran entirely on free CPU (GitHub Actions 4 vCPU / 16 GB runners) with chunked loss, swap and 2-hour checkpoint-and-resume chaining: zero GPU, zero cost.

## 3. Results (held-out, n = 164)

| model | refusal proxy (40 unanswerable) | EN token F1 (94) | HI token F1 (30) | false refusals (124 answerable) |
|---|---|---|---|---|
| Tiny Aya Global (base) | 10.0% | 0.294 | 0.265 | 0 |
| DPO on base, 50 steps | 40.0% | 0.287 | 0.267 | 0 |
| DPO on base, final (200 pairs) | 45.0% | 0.285 | 0.259 | |
| **Van Nyaya SFT (511 rows, final)** | **100.0%** | **0.359** | **0.306** | 3 (2.4%) |
| SFT + DPO | running | | | |

What it shows:

- **Refusal is learned from data, not from preference tuning alone.** DPO on the base model moves abstention from 10% to 45% but never touches answer quality. Supervised fine-tuning on curated grounded + refusal data matches the lexical refusal criterion on all 40 unanswerable test questions. This does not establish universal abstention or legal accuracy.
- **Grounding improves answers in both languages.** +22% English and +15% Hindi token F1 over the base, even though Hindi made up under a third of training data.
- **The safety cost is small and visible.** 3 of 124 answerable questions get a refusal the base would not give. We report it rather than hide it.

All generations and per-model score files are public in the eval-outputs dataset, so every number above can be re-scored.

### Actual AuditKIT cross-check

AuditKIT was executed offline on the stored base and final SFT predictions using its `F1Score` and `Abstention` scorers. The script and raw report are in `scripts/audit_offline.py` and `results/auditkit/`. Native AuditKIT normalization and abstention matching differ from the headline scorer, so its scores are reported separately and never substituted into the main table. No external judge or expert legal validation is claimed.

## 4. Why it matters

A legal assistant that is wrong with confidence is worse than none. For a forest rights claim under FRA or a compounding notice under IFA, the person asking often can't check the section. Getting a tiny, phone-sized, bilingual model to stay inside the text and say "the section doesn't say" is a behavior worth testing further, not proof of deployment safety. The GGUF can run on an ordinary CPU; a 2.14 GB browser download is substantial, especially on phones.

## 5. Artifacts

- Live demo (runs in your browser, plus base vs Van Nyaya side by side on held-out questions): https://huggingface.co/spaces/AkshayWagh/van-nyaya-demo
- GGUF (merged final SFT, split into browser-size shards): https://huggingface.co/AkshayWagh/van-nyaya-tiny-aya-gguf

- Model collection: https://huggingface.co/collections/AkshayWagh/van-nyaya-models-6aeb1675438b8efca8ba323e
- Dataset collection: https://huggingface.co/collections/AkshayWagh/van-nyaya-datasets-6aeb16b634d9685329e7ce83
- SFT LoRA (headline): https://huggingface.co/AkshayWagh/van-nyaya-tiny-aya-sft-lora
- GGUF, CPU-ready: https://huggingface.co/AkshayWagh/van-nyaya-tiny-aya-gguf
- DPO LoRA (ablation): https://huggingface.co/AkshayWagh/van-nyaya-tiny-aya-dpo-lora
- Data: https://huggingface.co/datasets/AkshayWagh/van-nyaya-grounded-qa
- Eval generations and scores: https://huggingface.co/datasets/AkshayWagh/van-nyaya-eval-outputs
- Code and training workflows: https://github.com/akshay-ira-2026/van-nyaya

## 6. Limits

- The Wildlife (Protection) Act text used is the pre-2022-amendment version.
- QA was generated and judged by an LLM with a hallucination gate; Hindi was machine-translated.
- Small eval set (164); token F1 underrates correct paraphrases.
- An early submission draft reported numbers from a quicker scorer (base refusal 20%, EN F1 0.450); all numbers above use one scorer, `scripts/score.py`, on every model.
- Refusal is a lexical proxy, not a semantic assessment. A response can contain a refusal phrase and still hallucinate.
- Token F1 measures lexical overlap, not legal correctness.
- No independent expert validation or production-safety claim.
- Not legal advice.

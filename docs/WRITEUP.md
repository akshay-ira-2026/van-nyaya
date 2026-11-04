# Van Nyaya: a grounded, bilingual forest-law assistant on Tiny Aya

Lexsi Labs AI Safety and Alignment Hackathon, Track 1 (CuratorKIT -> AlignTune -> AuditKIT).

> Draft. Numbers marked "interim" come from an early checkpoint and a quick scorer. Final numbers are AuditKIT runs on the finished adapters.

## 1. The problem

Indian forest and environment law sits across several Acts that forest officers, gram sabhas and citizens have to read together: the Indian Forest Act 1927, the Wildlife (Protection) Act 1972, the Forest (Conservation) Act 1980 as amended in 2023 (Van (Sanrakshan Evam Samvardhan) Adhiniyam), the Scheduled Tribes and Other Traditional Forest Dwellers (Recognition of Forest Rights) Act 2006 with its Rules, and the Environment (Protection) Act 1986.

A small multilingual model asked about these laws tends to do two unsafe things:
1. Answer from memory instead of from the section in front of it (wrong section numbers, invented penalties).
2. Answer confidently when the given text does not contain the answer.

Van Nyaya fine-tunes Tiny Aya Global so that it answers only from the supplied section, cites it, answers in English or Hindi, and says plainly when the text does not cover the question.

## 2. Pipeline

1. **CuratorKIT 0.1.22** — 332 sections chunked from the bare Acts -> grounded QA generation (3 questions per section, ministral-8b generator + judge, hallucination gate 0.5, exact dedup, HF dataset card written by the toolkit). Section-level split (seed 7): 33 held-out sections are never seen in training.
2. **Data built on top of the curated set**
   - Hindi QA (translated, 150 train / 30 eval)
   - Refusal set: questions paired with a *mismatched* section, gold answer = an abstention (300 train / 40 eval)
   - DPO pairs (749): grounded answer vs closed-book answer (449), abstain vs hallucinate (300)
3. **AlignTune 0.1.30-style recipe (LoRA)** — SFT on a 511-row mix (EN grounded + HI + refusal), then DPO (200 pairs) on top of the SFT adapter, reference = adapter disabled. Ablation: DPO directly on the base model.
4. **AuditKIT 1.1.3** — before/after on the 164-row held-out eval: 94 EN grounded, 30 HI grounded, 40 EN unanswerable. Metrics: token F1, ROUGE-L, ChrF, groundedness, hallucination, refusal rate on unanswerable, false-refusal rate on answerable.

Base model: CohereLabs/tiny-aya-global (weights loaded from an ungated mirror of the same checkpoint). Training ran entirely on free CPU (GitHub Actions 4 vCPU / 16 GB runners) with chunked loss computation, swap and 2-hour checkpoint-and-resume chaining. Every model is merged, converted to GGUF Q4_K_M and evaluated in the same quantised form a user would run.

## 3. Results (interim)

| model | EN F1 | EN ROUGE-L | HI F1 | refusal on unanswerable |
|---|---|---|---|---|
| Tiny Aya Global (official Q4_K_M) | 0.450 | 0.303 | 0.640 | 20.0% |
| same, our export pipeline, no adapter | 0.436 | 0.293 | 0.636 | 22.5% |
| DPO on base, step 20/50 | 0.436 | 0.292 | 0.645 | **37.5%** |
| SFT (511 rows) | training | | | |
| SFT + DPO | pending | | | |

Read: 20 DPO steps nearly double correct abstentions on questions the section can't answer (+17.5 points) while grounded answer quality in both languages holds.

## 4. Why it matters

A legal assistant that is wrong with confidence is worse than none. For forest rights claims under FRA or a compounding notice under IFA, the person asking often can't check the section. Getting a tiny, phone-sized, bilingual model to stay inside the text and say "the section doesn't say" is the safety property that makes it usable.

## 5. Artifacts

- Code and workflows: this repo
- Datasets, adapters, GGUF: Hugging Face (links added on upload)
- Demo: (link added on deploy)

## 6. Limits

- The Wildlife (Protection) Act text used is the pre-2022-amendment version.
- QA was generated and judged by an LLM with a hallucination gate; Hindi was machine-translated.
- Not legal advice.

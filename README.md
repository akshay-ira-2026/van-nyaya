# Van Nyaya

Grounded bilingual (English + Hindi) assistant for Indian forest and environment law, fine-tuned from Tiny Aya Global. Lexsi Labs AI Safety and Alignment Hackathon, Track 1 (CuratorKIT -> AlignTune -> AuditKIT).

Held-out results (n = 164, sections never seen in training): correct refusals 10% -> 100%, English token F1 0.294 -> 0.359 (+22%), Hindi token F1 0.265 -> 0.306 (+15%), 3/124 false refusals. Trained on free CPU only (GitHub Actions).

- Write-up: docs/WRITEUP.md and https://huggingface.co/spaces/AkshayWagh/van-nyaya
- Models: https://huggingface.co/collections/AkshayWagh/van-nyaya-models-6aeb1675438b8efca8ba323e
- Data: https://huggingface.co/collections/AkshayWagh/van-nyaya-datasets-6aeb16b634d9685329e7ce83
- Scoring: `python3 scripts/score.py --eval eval_all.jsonl --gen <gens.jsonl> --tag X --output-dir res`

## Live demo

- Live demo (runs in your browser, plus base vs Van Nyaya side by side on held-out questions): https://huggingface.co/spaces/AkshayWagh/van-nyaya-demo
- GGUF (merged final SFT, split into browser-size shards): https://huggingface.co/AkshayWagh/van-nyaya-tiny-aya-gguf

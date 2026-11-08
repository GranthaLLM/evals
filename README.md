# evals

Open-model evals on the Grantha Sanskrit corpus. Every model is reached through an OpenAI-compatible API, so nothing here needs a GPU.

Order of work: sandhi splitting (SandhiKosh), chrF++ translation score, semantic similarity against the English reference, then a GEMBA-MQM style judge that also reads the commentary, checked against a human-labelled set.

## Setup

    uv venv && uv pip install -e ".[dev]"
    pytest

Models live in `models.toml`. Keys come from the env var named there, then the key file. Responses are cached on disk under `cache/`, so reruns are free.

## Run

    grantha-evals sandhi <model> [--per-tag 100]
    grantha-evals translate <model> [--per-work 10] [--works philosophical/samkhyakarika ...]

Sandhi needs SandhiKosh (research use only, not redistributed here): `--data` defaults to `../tokenizer/data/SandhiKosh`. Translation reads `../corpus` by default. Summaries land in `results/`, per-item outputs in `results/raw/` (ignored).

## Scoring notes

- chrF++ is sacreBLEU `word_order=2`; the signature is written next to every score.
- Sandhi reports exact component match and boundary F1 (a boundary counts within one letter, as in the tokenizer eval). Five fixed few-shot demos come from the UoH sheet and are excluded from scoring.
- The English reference is one translator's rendering, not gold. Read scores against the human-human ceiling, not against 100.

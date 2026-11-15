# Tasks for the 3090 machine

You are on the Grantha evals repo, branch `pc`. This machine has an NVIDIA RTX 3090 (24 GB). One job: score Dharmamitra's `buddhist-nlp/mitra-qwen35-translate` (Apache-2.0, Qwen3.5-9B fine-tuned for Sanskrit/Tibetan/Chinese to English, top of their leaderboard) on the same verses as Gemma-4-31B and Qwen3.8-27B, and report results back.

Rules: commits are authored as the repo owner (use the repo's git identity, do not set an author), no co-author lines, conventional commit subjects of at most 50 characters, no em dashes anywhere in code, comments or messages. Do not change the eval logic. Only fix bugs that stop things from running, and say what you changed.

## 0. Setup

The evals read `../corpus` by default, so clone it next to this repo, then install.

```bash
git clone https://github.com/GranthaLLM/corpus.git ../corpus
uv venv && uv pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv
```

If `nvidia-smi` shows no GPU, stop and tell the user. The Gemma and Qwen translations are cached on the Mac (`cache/`, 21 MB, not in git). Check `cache/` exists and is non-empty. If missing, ask the user to run on the Mac: `rsync -av cache/ <pc>:<repo path>/cache/`. Without it the Gemma and Qwen rows cannot be rebuilt, the comparison will be incomplete.

## 1. Serve the model

```bash
python -m venv ~/vllm-env && source ~/vllm-env/bin/activate
pip install -U vllm
vllm serve buddhist-nlp/mitra-qwen35-translate --dtype bfloat16 --max-model-len 4096 --port 8000
```

The repo is public and ungated, no token needed. If an expired Hugging Face token is cached, it makes the lookup fail with "not a valid model identifier". Run the server with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and no `HF_TOKEN` set, or run `hf auth logout` first. Do not read or print the cached token.

Run it in tmux. About 18 GB of weights, fits the 3090. Qwen3.5 needs a recent vllm; if loading fails, upgrade vllm and transformers. Wait for the "Application startup complete" line.

`models.toml` already has the entry `mitra-qwen-9b` pointing at `http://localhost:8000/v1`. Set a dummy key in the eval shell: `export LOCAL_KEY=x`.

## 2. Check one verse before the full run

```bash
grantha-evals translate mitra-qwen-9b --per-work 1 --workers 1
```

Read one output in `results/raw/`. If the model returns reasoning text or Japanese/Tibetan instead of an English translation, stop and tell the user what it printed. Do not edit the prompt on your own.

## 3. Full run (same verses as the other models)

```bash
grantha-evals translate mitra-qwen-9b --per-work 150 --workers 16
grantha-evals report gemma-4-31b qwen3.8-27b mitra-qwen-9b
grantha-evals similarity gemma-4-31b qwen3.8-27b mitra-qwen-9b
```

Expect about 1,100 verses and 5 to 15 minutes. `report` and `similarity` print chrF++ and BGE cosine with 95% bootstrap intervals and paired differences, and write `results/translate_report.json` and `results/similarity_report.json`.

Commit and push the two report JSONs and `results/translate_mitra-qwen-9b.json`:

```bash
git add results/translate_report.json results/similarity_report.json results/translate_mitra-qwen-9b.json
git commit -m "results: MITRA-Qwen 9B translation scores" && git push
```

If the push fails because of credentials, show the numbers and tell the user. Then copy the cache back so the Mac can rebuild everything: `rsync -av <pc>:<repo path>/cache/ cache/` (user runs this).

## What to report

1. GPU name and that the server loaded.
2. One sample translation from step 2.
3. chrF++ and cosine for mitra-qwen-9b, with intervals, and the paired differences against Gemma-4-31B and Qwen3.8-27B.
4. Anything you had to change.

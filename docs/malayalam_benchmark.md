# Malayalam STT benchmark

This benchmark compares Groq Whisper Large v3 with Sarvam Saaras v3 on the same local, consented audio. It is intentionally **dry-run by default** and does not spend provider credits unless `--run` is explicitly supplied.

## Prepare samples

1. Record one speaker per file in a quiet space, then repeat selected phrases with realistic room noise.
2. Keep every clip under 30 seconds; WAV at 16 kHz mono is preferred.
3. Save files under `benchmarks/samples/`. This folder is ignored by Git.
4. Fill the matching `expected_transcript` in `benchmarks/malayalam_manifest.jsonl` with the exact intended Malayalam or code-mixed wording.
5. Do not include private conversations, passwords, financial details, or anyone's voice without consent.

Use at least six cases before making a provider decision: two clean, two noisy, and two Malayalam-English code-mixed samples. The included manifest provides eight case slots.

## Validate without credits

```powershell
python scripts/benchmark_stt.py
```

This validates the manifest and prints the selected case IDs. It makes no network requests.

## Run a controlled evaluation

Start with three cases and both providers:

```powershell
python scripts/benchmark_stt.py --run --max-cases 3 --providers groq,sarvam
```

Results are saved to `benchmarks/results/stt_benchmark_results.json`, which is ignored by Git. Each result includes latency, character error rate (CER), and word error rate (WER). Lower is better.

Only increase `--max-cases` after reviewing the first result file. Use the `ml-IN` language hint rather than automatic language detection for this Malayalam benchmark. Sarvam's `codemix` mode is selected for the code-mixed cases.

## Decision rule

Prefer Sarvam for Malayalam only if it materially improves both accuracy and practical latency across the noisy and code-mixed cases. A single clean sample is not enough evidence to change the live provider.

# Malayalam STT benchmark

This benchmark compares Groq Whisper Large v3 with Sarvam Saaras v3 on the same local, consented audio. It is intentionally **dry-run by default** and does not spend provider credits unless `--run` is explicitly supplied.

## Prepare samples

1. Record one speaker per file in a quiet space, then repeat selected phrases with realistic room noise.
2. Keep every clip under 30 seconds; WAV at 16 kHz mono is preferred.
3. Save files under `benchmarks/samples/`. This folder is ignored by Git.
4. Fill the matching `expected_transcript` in `benchmarks/malayalam_manifest.jsonl` with the exact intended Malayalam or code-mixed wording.
5. Do not include private conversations, passwords, financial details, or anyone's voice without consent.

Use at least six cases before making a provider decision: two clean, two noisy, and two Malayalam-English code-mixed samples. The included manifest provides eight case slots.

## Optional Mozilla Data Collective reference dataset

The project can inspect the Mozilla Data Collective API for the Malayalam Common Voice
Scripted Speech 26.0 dataset. This is useful for a standardised reference corpus, but it
does not replace local, consented noisy and Malayalam-English code-mixed recordings.

Create an API key in Mozilla Data Collective, accept the dataset's terms in its web UI,
then keep the key locally in `.env`:

```env
MDC_API_KEY=your_key_here
```

Inspect only (the default):

```powershell
python scripts/mdc_dataset.py
```

This retrieves metadata only. It does **not** create a download session, consume the
dataset-download allowance, or download any audio.

Downloading is intentionally separate. It requires the exact dataset ID to be repeated
and applies a 256 MB size limit by default:

```powershell
python scripts/mdc_dataset.py --download --confirm-dataset-id cmqiglff100innq07ytefmv64
```

The resulting archive is saved under `benchmarks/downloads/`, which is Git-ignored. Do
not re-host or commit Mozilla dataset archives. Only extract or select clips after
reviewing the dataset terms and documenting their references in a separate benchmark
manifest.

Prepare a deterministic 20-clip subset from the downloaded archive:

```powershell
python scripts/prepare_mdc_benchmark.py
```

It selects 2.5-15 second clips from the official test split, prioritises speaker variety,
copies only those MP3 files to `benchmarks/samples/mdc_common_voice_test/`, and writes a
transcript-backed manifest under `benchmarks/generated/`. Both locations are Git-ignored.
Run Groq only for the first review:

```powershell
python scripts/benchmark_stt.py --manifest benchmarks/generated/malayalam_mdc_test_manifest.jsonl --providers groq --run --max-cases 3
```

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

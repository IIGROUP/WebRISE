# WebRISE Code Release

This folder contains the core code used to construct and evaluate WebRISE interaction contracts.

## Contents

- `generation/icg_pipeline.py`: main pipeline for generating Test Data Contracts, test items, and Interaction Contract Graphs (ICGs).
- `generation/gen_input/`: utilities for building Text/MD/Sketch/Image/Video inputs, including recorded Video demonstrations.
- `evaluation/`: contract-guided browser evaluation with the adaptive agent, indexed DOM observation (`dom_observation.py`), DOM evidence capture (`dom_assert.py`), DOM/visual scorers, metrics, a vendored browser-use package, and test upload assets.

## Setup

```bash
pip install -r code_release/requirements.txt
python -m playwright install chromium
cp code_release/.env.example .env
```

Set `OPENAI_API_KEY` and, if needed, `OPENAI_BASE_URL` in `.env` or in the shell. `WEBRISE_ICG_MODEL` controls ICG construction, while `WEB_EVAL_*` controls agent evaluation and scoring.

## Generate ICGs

```bash
python code_release/generation/icg_pipeline.py \
  --input data_release/requirements_full.json \
  --data-dir generated_icg \
  --api-call-log-dir generated_icg/api_calls
```

Use `--filter TASK_ID` to regenerate a single task and `--force-all` to overwrite existing intermediate artifacts.

## Evaluate GT Artifacts

Run one task:

```bash
python code_release/evaluation/eval_agentmode.py \
  --html data_release/D01_S01_T003/Search_Result_Tabs.html \
  --icg data_release/D01_S01_T003/icg.json \
  --output eval_runs/D01_S01_T003 \
  --record-operations
```

Run the minimal shell wrapper:

```bash
bash code_release/evaluation/eval_agentmode.sh \
  data_release/D01_S01_T003/Search_Result_Tabs.html \
  data_release/D01_S01_T003/icg.json
```

## Build Input Modalities

```bash
python code_release/generation/gen_input/build_text_md_sketch_inputs.py --seed-root data_release --tasks D01_S01_T003
python code_release/generation/gen_input/build_image_inputs.py D01_S01_T003 --dir data_release --force
python code_release/generation/gen_input/build_video_inputs.py D01_S01_T003 --seed-root data_release --passed-only --force
python code_release/generation/gen_input/build_video_inputs.py D01_S01_T003 --seed-root data_release --input-only
```

Prompts used by the ICG generation, modality construction, adaptive agent, and DOM/visual scoring are included inline in the released scripts.

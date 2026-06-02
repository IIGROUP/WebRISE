# WebRISE: Requirement-Induced State Evaluation for MLLM-Generated Web Artifacts

[![Project Page](https://img.shields.io/badge/Project-Page-2f64c8)](https://iigroup.github.io/WebRISE/)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](#)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-ffcc4d)](https://huggingface.co/datasets/IIGroup/WebRISE)

**WebRISE** is a benchmark for evaluating whether MLLM-generated web artifacts actually work, rather than only look plausible. WebRISE compiles task requirements into Interaction Contract Graphs (ICGs) of observable UI states, user-intent transitions, and DOM/visual assertions, then evaluates generated HTML pages through browser execution.

## Overview

WebRISE targets executable, interactive web artifacts generated from multimodal specifications. It evaluates behavior at the level of requirement-induced state transitions, supporting diagnosis of explicit user-facing functions and implicit product-level constraints such as state synchronization, boundary feedback, stale-state cleanup, and state preservation.

The released benchmark includes:

- **442** web tasks across diverse domains and scenarios.
- **5** input modalities: Text, Markdown, Sketch, Image, and Video.
- **5,495** interaction transitions.
- **5,271** requirement checks.
- Contract-guided browser evaluation with DOM and visual evidence.

## Benchmark Design

- **Requirement-induced state contracts**: Each task is represented by an ICG that links explicit and implicit requirements to observable states, user-intent transitions, DOM/visual assertions, and coverage mappings.
- **Contract-guided adaptive execution**: The ICG specifies what to verify, while an adaptive browser agent decides how to execute each transition on diverse generated pages.
- **Implementation-agnostic observations**: The agent acts over indexed DOM observations instead of fixed CSS selectors, reference DOM paths, or hand-written scripts.
- **DOM/visual dual oracle**: DOM assertions capture process and element-level evidence, while visual postconditions verify user-visible state changes from screenshots.
- **Diagnostic metrics**: WebRISE reports state reachability, transition validity, explicit/implicit requirement coverage, and auxiliary visual quality diagnostics.

## Main Results

Each modality cell reports `T / R / V`, where `T` is transition validity, `R` is overall requirement coverage, and `V` is the auxiliary visual score.

| Model | Text | MD | Sketch | Image | Video | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 26.8 / 30.5 / 78.2 | 15.5 / 19.2 / 80.8 | 41.2 / 45.4 / 77.0 | 46.6 / 49.6 / 71.7 | 49.5 / 52.2 / 72.8 | 50.5 |
| Qwen3.5-122B-A10B | 38.0 / 41.2 / 56.8 | 42.5 / 45.9 / 72.0 | 38.0 / 42.3 / 74.0 | 40.2 / 43.8 / 70.7 | 42.8 / 47.1 / 71.3 | 51.1 |
| Qwen3.5-27B | 36.3 / 40.0 / 59.9 | 41.7 / 45.5 / 72.1 | 38.6 / 42.7 / 76.8 | 42.6 / 46.7 / 70.6 | 43.1 / 46.9 / 71.8 | 51.7 |
| Qwen3.5-397B-A17B | 45.7 / 49.2 / 64.8 | 51.1 / 54.5 / 75.7 | 46.8 / 50.5 / 78.9 | 48.4 / 51.4 / 72.8 | 49.3 / 52.8 / 72.1 | 57.6 |
| Kimi-K2.5 | 48.5 / 51.9 / 68.9 | 57.0 / 59.6 / 73.8 | 47.8 / 50.4 / 79.9 | 56.9 / 59.1 / 72.6 | 58.6 / 60.3 / 72.9 | 61.2 |
| Qwen3.6-27B | 47.9 / 50.9 / 75.3 | 57.5 / 60.1 / 83.0 | 50.4 / 53.3 / 87.2 | 55.2 / 57.8 / 74.1 | 54.2 / 57.2 / 74.1 | 62.5 |
| Kimi-K2.6 | 44.6 / 47.3 / 83.1 | 51.7 / 54.9 / 87.1 | 47.8 / 51.5 / 86.3 | 58.5 / 60.4 / 73.2 | 63.7 / 65.4 / 73.5 | 63.3 |
| Claude Opus 4.6 | 43.3 / 45.5 / 56.6 | 54.3 / 56.3 / 73.9 | 52.3 / 55.0 / 72.2 | 57.7 / 59.5 / 70.2 | 52.6 / 54.9 / 70.7 | 58.3 |
| Gemini 3 Flash | 44.7 / 48.2 / 71.9 | 50.0 / 54.1 / 79.3 | 46.1 / 49.3 / 85.4 | 54.1 / 57.5 / 72.4 | 45.6 / 48.5 / 70.8 | 58.5 |
| Claude Opus 4.7 | 48.8 / 50.9 / 68.3 | 54.5 / 56.5 / 76.2 | 49.7 / 52.4 / 77.4 | 57.0 / 58.5 / 70.5 | 65.0 / 66.1 / 72.7 | 61.6 |
| Gemini 3.1 Pro | 50.7 / 53.6 / 69.7 | 58.9 / 61.5 / 79.2 | 52.2 / 54.9 / 84.8 | 54.5 / 57.1 / 72.2 | 52.0 / 54.9 / 71.6 | 61.9 |
| Qwen3.6-Plus | 49.3 / 51.9 / 68.2 | 51.7 / 54.6 / 74.5 | 53.8 / 56.4 / 86.3 | 57.5 / 59.4 / 73.8 | 61.7 / 63.4 / 74.8 | 62.5 |
| GPT-5.4 | 59.7 / 61.4 / 78.4 | 60.5 / 62.2 / 79.8 | 57.8 / 60.3 / 86.6 | 60.0 / 62.1 / 71.5 | 63.1 / 64.8 / 73.7 | 66.8 |
| GPT-5.5 | 60.3 / 62.3 / 85.6 | 64.4 / 66.1 / 83.3 | 60.6 / 62.9 / 86.1 | 61.8 / 63.4 / 74.1 | 65.6 / 66.3 / 73.9 | 69.1 |

## Repository Structure

```text
.
├── generation/
│   ├── icg_pipeline.py
│   └── gen_input/
├── inference/
│   └── generate_html.py
├── evaluation/
│   ├── eval_agentmode.py
│   ├── dom_observation.py
│   ├── dom_assert.py
│   ├── dom_scorer.py
│   ├── scorer.py
│   ├── metrics.py
│   ├── test_assets/
│   └── browser-use/
├── requirements.txt
└── .env.example
```

## Dataset

The WebRISE data release is hosted on Hugging Face:

- **Dataset**: [IIGroup/WebRISE](https://huggingface.co/datasets/IIGroup/WebRISE)

Download the dataset with the Hugging Face CLI:

```bash
hf download IIGroup/WebRISE \
  --repo-type dataset
```

The dataset is organized as:

```text
<DATASET_DIR>/
├── requirements_full.json
└── data/
    └── <TASK_ID>/
        ├── icg.json
        ├── <artifact>.html
        └── ...
```

## Setup

```bash
git clone https://github.com/IIGROUP/WebRISE.git
cd WebRISE

pip install -r requirements.txt
python -m playwright install chromium

cp .env.example .env
```

Set API keys and judge model options in `.env` or in your shell. Common settings include:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_optional_base_url
MODEL_NAME=your_generation_model
WEBRISE_DATA_ROOT=path/to/dataset/data
WEB_EVAL_MODEL_AGENT=your_agent_model
WEB_EVAL_MODEL_SCORER=your_scoring_model
```

## Run Model Inference

Generate HTML artifacts from released modality inputs:

```bash
MODEL_NAME=your_model_name \
WEBRISE_DATA_ROOT=path/to/dataset/data \
TASK_MODE_FILTER=TASK_ID:Text \
python inference/generate_html.py
```

By default, outputs are written to:

```text
inference_outputs/<RUN_NAME>/<MODEL>/<MODALITY>/
```

## Run Evaluation

Evaluate a generated HTML artifact against its ICG:

```bash
python evaluation/eval_agentmode.py \
  --html path/to/generated.html \
  --icg path/to/icg.json \
  --output eval_runs/example
```

You can also use the lightweight shell wrapper:

```bash
bash evaluation/eval_agentmode.sh \
  path/to/generated.html \
  path/to/icg.json
```

## Utilities: Build Input Modalities

The released dataset already contains modality inputs. These scripts are mainly useful when regenerating or auditing modality assets.

```bash
DATA_ROOT=path/to/dataset/data
TASK_ID=your_task_id

python generation/gen_input/build_text_md_sketch_inputs.py \
  --data-root "$DATA_ROOT" \
  --tasks "$TASK_ID"

python generation/gen_input/build_image_inputs.py \
  "$TASK_ID" \
  --data-root "$DATA_ROOT"

python generation/gen_input/build_video_inputs.py \
  "$TASK_ID" \
  --data-root "$DATA_ROOT" \
  --passed-only
```

## Citation

If you use WebRISE in your research, please cite our paper:

```bibtex
```

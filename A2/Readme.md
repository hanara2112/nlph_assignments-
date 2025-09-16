## A2: NER + Concept Linking (Clean Minimal Baseline)

### Data
- Put CSVs in `A2/Assignment 2 Dataset/` with file names:
  - `training-notes.csv`, `train-annotations.csv`
  - `test-notes.csv`, `test-annotations.csv` (optional for evaluation)

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r A2/requirements.txt
```

### Quickstart
- Train NER (CRF) and run linker (hybrid dict+similarity):
```bash
python A2/baseline.py --data-dir "A2/Assignment 2 Dataset" --out-dir "A2/outputs" all
```

Outputs:
- `A2/outputs/ner_predictions.csv` (spans)
- `A2/outputs/linker_candidates.jsonl` (top-N candidates per span)
- `A2/outputs/linker_links.csv` (final concept per span)

### Run pieces
- NER only (and optionally attach training-surface concepts for sanity check):
```bash
python A2/baseline.py --data-dir "A2/Assignment 2 Dataset" --out-dir "A2/outputs" ner --attach-concept
```

- Linker only (choose mode):
  - Dictionary exact only: `--link-mode dict`
  - Similarity only: `--link-mode sim`
  - Hybrid (default): `--link-mode hybrid`
```bash
python A2/baseline.py --data-dir "A2/Assignment 2 Dataset" --out-dir "A2/outputs" link --link-mode hybrid --top-n 5
```

### Optional LLM modes (LoRA-ready)
- LLM for span extraction:
```bash
python A2/baseline.py --data-dir "A2/Assignment 2 Dataset" --out-dir "A2/outputs" ner \
  --ner-backend llm --llm-base-model mistralai/Mistral-7B-Instruct-v0.2 --device cuda --gpu-id 0
```

- FAISS retrieval + LLM selection:
```bash
python A2/baseline.py --data-dir "A2/Assignment 2 Dataset" --out-dir "A2/outputs" link \
  --link-mode llm --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --llm-base-model mistralai/Mistral-7B-Instruct-v0.2 --top-n 5 --device cuda --gpu-id 0 --faiss-gpu
```

- Quick LoRA fine-tune for span extraction (toy SFT):
```bash
python A2/baseline.py --data-dir "A2/Assignment 2 Dataset" --out-dir "A2/outputs" llm-train \
  --llm-base-model mistralai/Mistral-7B-Instruct-v0.2 --num-epochs 1 --batch-size 2 --max-examples 200
```

Notes:
- LLM modes require GPU for practicality and these extras: `sentence-transformers, faiss-cpu, transformers, torch, peft`.
- You can point to your own SNOMED dictionary CSV via `--snomed-csv` with columns `snomed_id,term,synonyms`.

### Evaluate spans (IoU)
```bash
python A2/evaluate.py --pred A2/outputs/ner_predictions.csv --gold "A2/Assignment 2 Dataset/test-annotations.csv"
```

### EDA (optional)
```bash
python A2/eda.py --data-dir "A2/Assignment 2 Dataset" --n 2
```

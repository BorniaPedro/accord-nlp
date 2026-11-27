# ACCORD-NLP: Joint Regulatory Rule Extraction with GLiNER-ReLEx

Extension of the [ACCORD-NLP](https://github.com/Accord-Project/accord-nlp) framework to perform **single-pass joint Named Entity Recognition (NER) and Relation Extraction (RE)** on building regulations from the **CODE-ACCORD** corpus.

Instead of relying on traditional cascading sequence-labeling pipelines (two separate `RoBERTa-large` models), this work adapts the span-based **GLiNER-ReLEx** (`DeBERTa-v3 Large` backbone) architecture combined with **Teacher-Student synthetic data augmentation** generated via **Gemini 3.1 Pro**, achieving over **+35 percentage points** improvement in end-to-end Strict Relation Macro-F1.

---

## Pre-trained Models (Hugging Face Hub)

Both optimized checkpoints (`seed=42`, selected via validation Strict RE Macro-F1 across a 216-configuration grid search) are publicly available:

| Model Variant | Training Set | Strict RE Macro-F1 | Relaxed RE Macro-F1 | Hugging Face Checkpoint |
| :--- | :--- | :---: | :---: | :--- |
| **2x RoBERTa-large** (Baseline Cascade) | 620 sentences | 21.88% | — | `ACCORD-NLP/*` |
| **GLiNER-ReLEx (`Standard`)** | 620 sentences | **58.08%** | 67.23% | [`bornia/gliner-relex-code-accord-standard`](https://huggingface.co/bornia/gliner-relex-code-accord-standard) |
| **GLiNER-ReLEx (`Augmented`)** | 1,220 sentences | 57.67% | **69.40%** | [`bornia/gliner-relex-code-accord-augmented`](https://huggingface.co/bornia/gliner-relex-code-accord-augmented) |

---

## Setup

1. Install [PyTorch](https://pytorch.org/get-started/locally/) matching your CUDA/ROCm/CPU platform.
2. Clone this repository and install the remaining dependencies:

```bash
git clone https://github.com/BorniaPedro/accord-nlp.git
cd accord-nlp
pip install -r requirements.txt
```

---

## Repository Structure

```text
├── accord_nlp/                         # Original ACCORD-NLP cascading baseline package
├── data/
│   └── relex/
│       ├── relex_train.json            # Original CODE-ACCORD train split (620 sentences)
│       ├── relex_train_augmented.json  # Augmented train split (620 original + 600 synthetic)
│       ├── relex_val.json              # Validation split (used for grid-search selection)
│       └── relex_test.json             # Isolated test split (173 sentences)
└── experiments/
    ├── ie.py                           # End-to-end baseline evaluation script
    └── relex/
        ├── augment_relex.py            # Synthetic batch validation, span alignment & merging
        ├── train_relex.py              # Single-pass fine-tuning + Strict & Relaxed evaluation
        ├── run_grid_relex.py           # Full 216-run grid search or final winners reproduction
        └── infer_relex.py              # Single-sentence inference & graph visualization
```

---

## Usage & Reproducibility

### 1. Run Inference & Generate Knowledge Graphs
Edit `SENTENCES` in `experiments/relex/infer_relex.py` and run:

```bash
python -m experiments.relex.infer_relex
```
Downloads the fine-tuned checkpoint from Hugging Face, prints extracted entities and relational triples to the terminal, and saves directed graph plots to `outputs/inference_graphs/`.

### 2. Reproduce Thesis Results (Final Winners)
Train and evaluate the two winning configurations (`Standard` and `Augmented`) on the isolated test split (`data/relex/relex_test.json`) with `seed=42`:

```bash
python -m experiments.relex.run_grid_relex --final_only
```
Outputs generated in `outputs/`:
* `per_class_results_seed42.json`: Strict and Boundary-Relaxed Precision, Recall, and F1-scores per entity and relation class.
* `final_models/<scenario>/test_predictions.json`: Sentence-by-sentence gold vs. predicted entities and triples for qualitative error analysis.

### 3. Run the Full Hyperparameter Grid Search (216 Configurations)

```bash
python -m experiments.relex.run_grid_relex
```

### 4. Custom Training
Run `train_relex.py` directly to fine-tune with custom hyperparameters:

```bash
python -m experiments.relex.train_relex \
  --model_path knowledgator/gliner-relex-large-v0.5 \
  --train_path data/relex/relex_train_augmented.json \
  --epochs 10 --batch_size 16 --lr 3e-5 --others_lr 5e-5 \
  --ner_thresholds 0.40 --rel_thresholds 0.60 \
  --fp16 --save_predictions --augmented_data
```

### 5. Validate & Merge New Synthetic Batches
To validate character offsets and token alignments of new LLM-generated JSON batches and merge them with the original training set:

```bash
python -m experiments.relex.augment_relex \
  --synthetic_raw data/relex/synthetic_raw_600.json \
  --train_path data/relex/relex_train.json \
  --output_path data/relex/relex_train_augmented.json
```

---

## Acknowledgments & Base Framework Citation

This codebase builds upon the open-source **ACCORD-NLP** framework developed under the Horizon Europe ACCORD project. The original cascading NER + RE pipeline (`accord_nlp/`) and evaluation utilities remain intact for baseline reproducibility.

```bibtex
@article{hettiarachchi-etal-2026-snowtec,
  title={{SNOWTEC}: Synthetic Natural language Oversampling With Transformer-based information ExtraCtion for automated compliance checking},
  author={Hettiarachchi, Hansi and Gaber, Mohamed Medhat and Parsafard, Pouyan and Vakaj, Edlira},
  journal={Machine Learning with Applications},
  volume = {24},
  pages={100911},
  year={2026},
  issn = {2666-8270},
  doi = {https://doi.org/10.1016/j.mlwa.2026.100911},
  publisher={Elsevier}
}
```
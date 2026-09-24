# 🎯 Smart MCQ Solver: Fine-Tuning DeBERTa-v3 for Top-3 Answer Ranking

> A fine-tuned **DeBERTa-v3-base** multiple-choice model that ranks the three most likely answers to challenging five-option questions. Built for the **Smart MCQ Solver Challenge** (Term 2, 2026) and evaluated with **MAP@3**.


---

## 📌 About the Challenge

Each question has a prompt and five options labeled **A, B, C, D, E**. Instead of a single answer, the model outputs a **ranked list of three labels**. The earlier the correct answer appears in the list, the higher the score.

| Detail | Description |
|--------|-------------|
| Task | Multiple choice question answering with ranked top-3 prediction |
| Input | `prompt` + options `A` to `E` |
| Output | Three labels per question `ID`, space-separated |
| Metric | Mean Average Precision at 3 (MAP@3) |
| Data | 2,000 training questions, 500 test questions |
| Competition window | May 22, 2026 to Aug 3, 2026 |

**MAP@3 example.** If the correct answer is `A`:

- `A B C` → highest score (correct answer ranked 1st)
- `B A C` → lower score (ranked 2nd)
- `C D A` → lowest score (ranked 3rd)

**Submission format**

```csv
ID,Prediction
1,A B C
2,C A D
3,B D A
```

## 🧠 Approach

The final solution (`notebook.ipynb`) fine-tunes `microsoft/deberta-v3-base` with Hugging Face's `AutoModelForMultipleChoice`.

1. **Input encoding:** the prompt is paired with each of the five options, so every question becomes five (prompt, option) sequences tokenised to a max length of 256.
2. **Scoring:** the model produces one logit per option, and a softmax across the five options gives the answer distribution.
3. **Top-3 ranking:** logits are sorted in descending order and the three highest-scoring labels are written to the submission file.
4. **Training tricks:**
   - Differential learning rates: `9e-6` for the transformer backbone and `1e-4` for the classifier/pooler head
   - AdamW with weight decay `0.01` (excluded for biases and LayerNorm weights)
   - Cosine learning-rate schedule with 10% warmup
   - Label smoothing of `0.08`
   - Gradient accumulation (batch 4 × 4 steps = effective batch 16) and gradient clipping at 1.0
   - Mixed-precision (FP16) training on GPU
5. **Checkpointing:** the best checkpoint (lowest epoch training loss) is saved and reloaded for inference.
6. **Experiment tracking:** loss, accuracy, and learning rate are logged to Weights & Biases.

### Training configuration

| Setting | Value |
|---------|-------|
| Base model | `microsoft/deberta-v3-base` |
| Max sequence length | 256 |
| Epochs | 12 |
| Batch size | 4 (accumulation 4, effective 16) |
| LR (backbone / head) | 9e-6 / 1e-4 |
| Warmup | 10% of steps |
| Label smoothing | 0.08 |
| Precision | FP16 (autocast) |

## 📊 Results

Training progressed steadily over 12 epochs (about 41 minutes on a Kaggle GPU):

| Epoch | Train loss | Train accuracy |
|-------|-----------|----------------|
| 1 | 1.5712 | 27.5% |
| 4 | 0.8662 | 72.9% |
| 8 | 0.5865 | 89.2% |
| 12 | 0.5400 | 91.3% |

| Metric | Score |
|--------|-------|
| Train MAP@3 (best checkpoint) | **0.9948** |
| Leaderboard MAP@3 | **0.75084** |

> ⚠️ **Note:** the MAP@3 above is measured on the **training set**, so it is optimistic. The leaderboard score is the reliable measure of generalisation.

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `notebook.ipynb` | Final solution: DeBERTa-v3 fine-tuning, inference, and submission generation |
| `eda.ipynb` | Exploratory data analysis |
| `LR_model.ipynb` | Linear Regression baseline experiment |
| `MLP_Scratch_model.ipynb` | MLP built from scratch experiment |
| `milestone_1.ipynb` | Milestone 1 submission |
| `milestone_2.ipynb` | Milestone 2 submission |

## 🛠️ Tech Stack

- Python 3, Jupyter / Kaggle Notebooks
- PyTorch, Hugging Face Transformers (with `sentencepiece` for the DeBERTa tokenizer)
- NumPy, Pandas
- Weights & Biases (experiment tracking)

## 🚀 Getting Started

The notebook is designed for **Kaggle** with a GPU accelerator.

1. Open `notebook.ipynb` in a Kaggle notebook and attach the *Smart MCQ Solver Challenge* competition data (`train.csv`, `test.csv`, `sample_submission.csv`).
2. Turn on a GPU and internet access (needed to download the model weights).
3. *(Optional)* Add your Weights & Biases key as a Kaggle secret named `WANDB_API_KEY`.
4. Run all cells. The notebook writes `submission_deberta_v3.csv` and `best_model.pt` to `/kaggle/working/`.

To run locally instead:

```bash
git clone https://github.com/Uma-Shettar/smart-mcq-solver-challenge.git
cd smart-mcq-solver-challenge
pip install torch transformers sentencepiece pandas numpy wandb jupyter
```

Then update `DATA_DIR` in the config cell to point at your local data folder.


## 📄 License

Created for academic purposes. Feel free to learn from it, and please credit if you build on it.

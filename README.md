# MindEase — Stress Level Detection System

AI-powered stress detection using physiological sensor data and natural language processing.

**Developed by:** 22CS014 & 22CS066 — Mehran University of Engineering & Technology

---

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| Stress-Lysis | Local CSV (`Stress-Lysis.csv`) | Sensor-based SVM training |
| Dreaddit | HuggingFace: `andreagasparini/dreaddit` | Text-based DistilBERT training |
| GoEmotions | HuggingFace: `SetFit/go_emotions` | Text-based DistilBERT training |

> Place `Stress-Lysis.csv` in the project root before running `train.py`.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train models

```bash
python train.py
```

This will:
- Load datasets (Dreaddit and GoEmotions are fetched automatically from HuggingFace)
- Run EDA and visualisations
- Train Logistic Regression, Random Forest, and SVM on sensor data
- Fine-tune DistilBERT on combined text data
- Save `stress_trained.sav`, `./final_stress_model/`, `./final_stress_tokenizer/`

### 2. Run the web app

```bash
streamlit run app.py
```

---

## Project Structure

```
├── train.py              # Full training pipeline
├── app.py                # MindEase Streamlit web app
├── requirements.txt      # Python dependencies
├── Stress-Lysis.csv      # Physiological sensor dataset (provide locally)
├── combined_dataset.csv  # Auto-generated after training
├── stress_trained.sav    # Saved SVM model
├── final_stress_model/   # Saved DistilBERT model
└── final_stress_tokenizer/
```

---

## Models

- **SVM** (`stress_trained.sav`): Classifies stress from Humidity, Temperature, Step Count into Low / Medium / High.
- **DistilBERT** (`final_stress_model`): Fine-tuned on combined Reddit and physiological text data for 3-class stress classification.

The app combines both models with equal weighting to produce a final stress score (0–10).

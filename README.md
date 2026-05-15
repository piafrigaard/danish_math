# Danish Math Reasoning Thesis Code

This repository contains the code used for the thesis:

**Cross-Lingual Generalization of Reasoning Models in Danish Mathematical Contexts**

The code was written for thesis experiments and is included for transparency and reproducibility. It is not intended as a general-purpose library.

## Contents

The repository contains scripts for:

- Translating and preparing the Danish GSM8K-style dataset
- Supervised fine-tuning (SFT)
- GRPO training
- Standard evaluation
- Test-time scaling experiments
- Plotting selected thesis figures

## Main scripts

The most important scripts are:

- `danish_sft.py` — Danish-only supervised fine-tuning
- `english_sft.py` — English-only supervised fine-tuning
- `sft.py` — bilingual paired supervised fine-tuning
- `evaluate.py` — standard model evaluation
- `lastdagrpo.py` — Danish GRPO training
- `daengrpo.py` — Danish GRPO with English support
- `improvedtranslate.py` — dataset translation script

Some scripts contain local paths, model names, or output folders used during the thesis. These may need to be changed before running the code in another environment.

## Dataset

The dataset file is derived from GSM8K and contains aligned English and Danish examples used in the thesis experiments.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
